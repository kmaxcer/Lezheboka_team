"""Объяснимое ранжирование раннего предупреждения для панели растительности.

Радар — функция представления: он использует уже обогащённые строки NDVI
и серии аномалий, но не меняет артефакт предсказаний. Все компоненты ограничены,
чтобы сравнение рейтинга не зависело от размера AOI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def early_warning_radar(frame: pd.DataFrame, periods: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ранжирует AOI по текущему стрессу и неопределённости с понятными факторами.

    ``risk_score`` is 0--100 and combines negative robust z-score, critical /
    suppressed observations, persistence of stress runs, and reconstruction
    uncertainty.  It is intentionally descriptive and does not alter NDVI.
    """
    cols = ["anon_polygon_id", "risk_score", "risk_level", "mean_zscore",
            "critical_share", "suppression_share", "max_stress_days",
            "reconstructed_share", "coverage", "top_factors"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)
    d = frame.copy()
    pid = d.get("anon_polygon_id", pd.Series("__unknown__", index=d.index)).astype(str)
    z = pd.to_numeric(d.get("ndvi_zscore", pd.Series(np.nan, index=d.index)), errors="coerce")
    status = d.get("status", pd.Series("unknown", index=d.index)).astype(str)
    observed = d.get("is_observed", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    reconstructed = d.get("is_reconstructed", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    # Ограниченные компоненты для каждого AOI. Компонента отрицательного z полезна,
    # когда исторический эталон доступен лишь для нескольких строк.
    tmp = pd.DataFrame({"pid": pid, "z": z, "critical": status.eq("critical"),
                        "suppression": status.eq("suppression"),
                        "reconstructed": reconstructed, "observed": observed,
                        "has_z": z.notna()})
    g = tmp.groupby("pid", sort=False)
    out = g.agg(rows=("pid", "size"), mean_zscore=("z", "mean"),
                critical_share=("critical", "mean"), suppression_share=("suppression", "mean"),
                reconstructed_share=("reconstructed", "mean"), observed_n=("observed", "sum"),
                clim_n=("has_z", "sum")).reset_index().rename(columns={"pid": "anon_polygon_id"})
    out["coverage"] = (out["clim_n"] / out["rows"].clip(lower=1)).clip(0, 1)
    if periods is None or periods.empty:
        runs = pd.DataFrame(columns=["anon_polygon_id", "n_days"])
    else:
        runs = periods.copy()
        runs["anon_polygon_id"] = runs["anon_polygon_id"].astype(str)
        runs["n_days"] = pd.to_numeric(runs.get("n_days", 0), errors="coerce").fillna(0)
    run_g = runs.groupby("anon_polygon_id", sort=False) if not runs.empty else None
    if run_g is None:
        out["max_stress_days"] = 0.0
        out["stress_runs"] = 0
    else:
        run_stats = run_g.agg(max_stress_days=("n_days", "max"), stress_runs=("n_days", "size")).reset_index()
        out = out.merge(run_stats, on="anon_polygon_id", how="left")
        out["max_stress_days"] = out["max_stress_days"].fillna(0.0)
        out["stress_runs"] = out["stress_runs"].fillna(0).astype(int)
    # Отсутствующая климатология повышает неопределённость, но ограничивается, чтобы
    # не перекрывать действительно сильную аномалию растительности.
    z_component = (-out["mean_zscore"].fillna(0) / 3.0).clip(0, 1)
    persistence = (out["max_stress_days"] / 14.0).clip(0, 1)
    uncertainty = (out["reconstructed_share"] * 0.55 + (1 - out["coverage"]) * 0.45).clip(0, 1)
    out["risk_score"] = (100 * (0.35 * z_component + 0.25 * out["critical_share"] +
                                0.15 * out["suppression_share"] + 0.15 * persistence +
                                0.10 * uncertainty)).round(1)
    out["risk_level"] = pd.cut(out["risk_score"], bins=[-np.inf, 20, 50, np.inf],
                               labels=["Низкий", "Наблюдать", "Высокий"]).astype(str)

    def factors(r: pd.Series) -> str:
        vals = {
            "критические точки": float(r.critical_share),
            "длинный стресс": float(min(r.max_stress_days / 14.0, 1.0)),
            "подавление": float(r.suppression_share),
            "отрицательный z-score": float(max(-r.mean_zscore / 3.0, 0.0)) if pd.notna(r.mean_zscore) else 0.0,
            "неопределённость восстановления": float(uncertainty.loc[r.name]),
        }
        ordered = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
        return ", ".join(name for name, value in ordered[:2] if value > 0.05) or "нет сильных сигналов"

    out["top_factors"] = out.apply(factors, axis=1)
    return out.sort_values(["risk_score", "anon_polygon_id"], ascending=[False, True]).reset_index(drop=True)[cols]


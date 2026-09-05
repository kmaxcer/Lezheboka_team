"""Leakage-aware post-correction sweep on the exact hidden-DOY proxy.

Uses the row-level predictions produced by ``exact_compare.py``.  Every
calibration is fit on years other than the scored year (leave-one-year-out),
so the result is a directional estimate rather than an in-sample claim.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "research" / "exact_compare_preds.csv"


def rmse(a: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - y) ** 2)))


def mae(a: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(a - y)))


def fit_affine(x: np.ndarray, y: np.ndarray, robust: bool = False) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if robust:
        # Remove only gross target tails for the calibration fit; validation
        # targets are never clipped when scoring.
        qx = np.quantile(x, [0.01, 0.99]); qy = np.quantile(y, [0.01, 0.99])
        keep = (x >= qx[0]) & (x <= qx[1]) & (y >= qy[0]) & (y <= qy[1])
        x, y = x[keep], y[keep]
    if len(x) < 20 or np.ptp(x) < 1e-8:
        return float(np.median(y) - np.median(x)), 1.0
    b, a = np.polyfit(x, y, 1)
    if not np.isfinite(a + b) or abs(b) > 3:
        return 0.0, 1.0
    return float(a), float(b)


def main() -> None:
    d = pd.read_csv(PRED, parse_dates=["date"])
    y = d["_truth"].to_numpy(float)
    methods = [c for c in ("hgb", "lag_k16_d3", "base_k8") if c in d]
    rows: list[dict[str, object]] = []

    # Raw component baselines.
    for m in methods:
        p = d[m].to_numpy(float)
        rows.append({"family": "raw", "method": m, "param": "none",
                     "rmse": rmse(p, y), "mae": mae(p, y), "n": len(y)})

    # Global blend weights, evaluated on the exact hidden-DOY proxy.
    h = d["hgb"].to_numpy(float); l = d["lag_k16_d3"].to_numpy(float)
    for w in np.arange(0, 0.61, 0.05):
        p = (1-w)*h + w*l
        rows.append({"family": "global_blend", "method": "hgb_lag",
                     "param": float(round(w, 2)), "rmse": rmse(p, y),
                     "mae": mae(p, y), "n": len(y)})

    # Leave-one-year-out calibration and clipping.  Calibration parameters are
    # estimated only from the other five years.
    years = sorted(d["year"].unique())
    for m in methods:
        x = d[m].to_numpy(float)
        for mode in ("affine", "robust_affine", "median_bias"):
            pred = np.full(len(d), np.nan)
            for year in years:
                train = d["year"].to_numpy() != year
                if mode == "median_bias":
                    shift = float(np.median(y[train] - x[train]))
                    a, b = shift, 1.0
                else:
                    a, b = fit_affine(x[train], y[train], robust=(mode == "robust_affine"))
                pred[~train] = a + b*x[~train]
            rows.append({"family": "loo_calibration", "method": m,
                         "param": mode, "rmse": rmse(pred, y),
                         "mae": mae(pred, y), "n": len(y)})

        # Fixed physical clipping is a post-correction that does not estimate
        # anything from validation labels.
        for lo, hi in ((-0.2, 1.1), (-0.1, 1.0), (0.0, 1.0), (-0.05, 0.95)):
            p = np.clip(x, lo, hi)
            rows.append({"family": "fixed_clip", "method": m,
                         "param": f"{lo}:{hi}", "rmse": rmse(p, y),
                         "mae": mae(p, y), "n": len(y)})

    # Leave-one-year-out blend weight: fit one scalar on the other years and
    # apply it to the held-out year.  This tests whether 0.2–0.3 is stable.
    loo = []
    for year in years:
        train = d["year"].to_numpy() != year
        delta = l[train] - h[train]
        w = float(np.sum(delta*(y[train]-h[train])) / np.sum(delta*delta)) if np.sum(delta*delta) else 0.0
        w = float(np.clip(w, 0.0, 1.0))
        test = ~train
        p = h[test] + w*(l[test]-h[test])
        loo.append({"year": int(year), "weight": w, "rmse": rmse(p, y[test]), "n": int(test.sum())})
        rows.append({"family": "loo_blend", "method": "hgb_lag",
                     "param": f"year={year};w={w:.4f}", "rmse": rmse(p, y[test]),
                     "mae": mae(p, y[test]), "n": int(test.sum())})

    out = pd.DataFrame(rows).sort_values("rmse")
    out.to_csv(ROOT / "research" / "teammate_sweep_root_postcorr.csv", index=False)
    pd.DataFrame(loo).to_csv(ROOT / "research" / "teammate_sweep_root_loo_weights.csv", index=False)
    best = out.head(12)
    best.to_csv(ROOT / "research" / "teammate_sweep_root_postcorr_top.csv", index=False)
    report = [
        "# Root post-correction sweep",
        "",
        "Источник: research/exact_compare_preds.csv (1114 строк, hidden-DOY proxy, 2019–2024).",
        "Все параметры LOO-калибровки подгонялись на других годах.",
        "",
        "## Лучшие варианты",
        "",
        "```text",
        best.to_string(index=False),
        "```",
        "",
        "## LOO blend weights",
        "",
        "```text",
        pd.DataFrame(loo).to_string(index=False),
        "```",
        "",
        "Вывод: fixed blend/клиппинг сравниваются с HGB без изменения production outputs;",
        "параметр не переносится автоматически без отдельной проверки на 2025.",
    ]
    (ROOT / "research" / "teammate_sweep_root_postcorr.md").write_text("\n".join(report), encoding="utf-8")
    print(best.to_string(index=False))
    print("LOO weights")
    print(pd.DataFrame(loo).to_string(index=False))


if __name__ == "__main__":
    main()

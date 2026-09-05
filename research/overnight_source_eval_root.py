"""Research-only source routing and residual/diversity evaluator.

This script uses the actual private synthetic day-of-year pattern projected onto
fully observed train years, plus per-AOI/year private-like random masks.  It
compares the existing soft source posterior with hard (modal) and oracle source
routing, then tests a strictly out-of-fold date/crop residual correction and a
small convex diversity blend.  It never writes ``outputs/`` or the input CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(ROOT / "src"))
from infer import (  # noqa: E402
    SOURCES,
    _fit_source_maps,
    _local_source_prediction,
    _mode_posteriors,
    _prepare,
    _query_posterior,
    _safe_bool,
    predict_private,
)
from infer_lag import predict_private_lag  # noqa: E402


DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def _source_labels(frame: pd.DataFrame) -> np.ndarray:
    s2 = np.isfinite(frame["s2_ndvi"].to_numpy(float))
    ls = np.isfinite(frame["landsat_ndvi"].to_numpy(float))
    md = np.isfinite(frame["modis_ndvi"].to_numpy(float))
    out = np.full(len(frame), "none", dtype=object)
    out[s2] = "s2"
    out[~s2 & ls] = "landsat"
    out[~s2 & ~ls & md] = "modis"
    return out


def _hidden_doys(private: pd.DataFrame) -> dict[str, set[int]]:
    p = private.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    flag = _safe_bool(p["is_synthetic_gap"])
    p = p.loc[flag].copy()
    p["_doy"] = p["date"].dt.dayofyear.astype(int)
    return p.groupby("anon_polygon_id")["_doy"].apply(set).to_dict()


def exact_doy_fold(train: pd.DataFrame, private: pd.DataFrame, year: int):
    """Project the organizer's hidden DOYs onto one observed train year."""
    d = train.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["_truth"] = pd.to_numeric(d["primary_ndvi"], errors="coerce")
    d["_true_src"] = _source_labels(d)
    d["is_synthetic_gap"] = False
    d["_year_orig"] = d["date"].dt.year.astype(int)
    d["_doy_orig"] = d["date"].dt.dayofyear.astype(int)
    doys = _hidden_doys(private)
    ids = d["anon_polygon_id"].astype(str)
    hide = (
        (d["_year_orig"].to_numpy() == int(year))
        & np.fromiter(
            (int(day) in doys.get(str(aoi), set())
             for aoi, day in zip(ids, d["_doy_orig"])),
            dtype=bool,
            count=len(d),
        )
        & d["_truth"].notna().to_numpy(bool)
    )
    for col in DYNAMIC:
        if col in d.columns:
            d.loc[hide, col] = np.nan
    d.loc[hide, "is_synthetic_gap"] = True
    return d, hide


def random_private_like_fold(frame: pd.DataFrame, seed: int, frac: float = 0.15):
    """Mask known rows independently within AOI/year, preserving schedules."""
    d = frame.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["_truth"] = pd.to_numeric(d["primary_ndvi"], errors="coerce")
    d["_true_src"] = _source_labels(d)
    d["is_synthetic_gap"] = False
    d["_year_orig"] = d["date"].dt.year.astype(int)
    d["_doy_orig"] = d["date"].dt.dayofyear.astype(int)
    rng = np.random.default_rng(int(seed))
    hide = np.zeros(len(d), dtype=bool)
    known = d["_truth"].notna().to_numpy(bool)
    for _, idx in d.groupby(["anon_polygon_id", "_year_orig"], sort=False).groups.items():
        ii = np.asarray(idx, dtype=int)
        ii = ii[known[ii]]
        n = int(float(frac) * len(ii))
        if n:
            hide[rng.choice(ii, n, replace=False)] = True
    for col in DYNAMIC:
        if col in d.columns:
            d.loc[hide, col] = np.nan
    d.loc[hide, "is_synthetic_gap"] = True
    return d, hide


def _routing_predictions(fold: pd.DataFrame, hide: np.ndarray, k: int = 8):
    """Return soft, hard, uniform and true-source local predictions."""
    d = _prepare(fold)
    syn = np.asarray(hide, dtype=bool)
    known = np.isfinite(d["primary_ndvi"].to_numpy(float))
    y = d["primary_ndvi"].to_numpy(float)
    x = d["_ord"].to_numpy(float)
    src = d["_src"].to_numpy(object)
    maps = _fit_source_maps(d, known, bin_days=30)
    aoi, crop, glob, date = _mode_posteriors(d, known)
    out = {"soft": np.full(len(d), np.nan), "hard": np.full(len(d), np.nan),
           "uniform": np.full(len(d), np.nan), "oracle_true": np.full(len(d), np.nan)}
    posterior_rows: list[np.ndarray] = []
    for _, idx in d.groupby(["anon_polygon_id", "_year"], sort=False).groups.items():
        ii = np.asarray(idx, dtype=int)
        kk = ii[known[ii]]
        for q in ii[syn[ii]]:
            p = _query_posterior(d, int(q), aoi, crop, glob, date,
                                 date_weight=1.0)
            vals = np.full(3, np.nan)
            for si, target in enumerate(SOURCES):
                vals[si] = _local_source_prediction(
                    x[q], kk, x, y, src, target, maps,
                    query_doy=int(d["_doy"].iat[q]), bin_days=30, k=k,
                )
            good = np.isfinite(vals)
            if good.any():
                pp = p.copy()
                pp[~good] = 0.0
                if pp.sum() <= 0:
                    pp[good] = 1.0 / float(good.sum())
                out["soft"][q] = float(np.average(vals[good], weights=pp[good]))
                out["uniform"][q] = float(np.nanmean(vals))
                mode = int(np.argmax(np.where(good, p, -np.inf)))
                out["hard"][q] = float(vals[mode])
                true_source = str(fold["_true_src"].iat[q])
                if true_source in SOURCES:
                    ti = SOURCES.index(true_source)
                    if np.isfinite(vals[ti]):
                        out["oracle_true"][q] = float(vals[ti])
            posterior_rows.append(p)
    # Fill any pathological group using the same nearest-observation fallback
    # as production, solely so metrics have a complete denominator.
    for name in out:
        miss = syn & ~np.isfinite(out[name])
        for q in np.flatnonzero(miss):
            same = np.flatnonzero(known & (d["anon_polygon_id"].to_numpy() == d["anon_polygon_id"].iat[q]))
            out[name][q] = y[same[np.argmin(np.abs(x[same] - x[q]))]] if len(same) else np.nanmedian(y[known])
    return out, d


def _loo_residuals(d: pd.DataFrame, hide: np.ndarray, k: int = 8) -> pd.DataFrame:
    """Compute source-soft LOO residuals for date/crop correction groups."""
    known = np.isfinite(d["primary_ndvi"].to_numpy(float))
    y = d["primary_ndvi"].to_numpy(float)
    x = d["_ord"].to_numpy(float)
    src = d["_src"].to_numpy(object)
    maps = _fit_source_maps(d, known, bin_days=30)
    aoi, crop, glob, date = _mode_posteriors(d, known)
    needed_doy = set(d.loc[hide, "_doy"].astype(int).tolist())
    candidates = np.flatnonzero(known & d["_doy"].isin(needed_doy).to_numpy(bool))
    rows: list[dict[str, Any]] = []
    groups = d.groupby(["anon_polygon_id", "_year"], sort=False).groups
    group_by_pos = {int(i): np.asarray(idx, dtype=int) for idx in groups.values() for i in idx}
    for i in candidates:
        ii = group_by_pos[int(i)]
        kk = ii[known[ii] & (ii != i)]
        if len(kk) == 0:
            continue
        p = _query_posterior(d, int(i), aoi, crop, glob, date, date_weight=1.0)
        vals = []
        for target, w in zip(SOURCES, p):
            v = _local_source_prediction(
                x[i], kk, x, y, src, target, maps,
                query_doy=int(d["_doy"].iat[i]), bin_days=30, k=k,
            )
            if np.isfinite(v):
                vals.append((v, float(w)))
        if vals:
            pred = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
            rows.append({
                "date": d["date"].iat[i], "_year": int(d["_year"].iat[i]),
                "_doy": int(d["_doy"].iat[i]), "crop_type": str(d["crop_type"].iat[i]),
                "aoi": str(d["anon_polygon_id"].iat[i]),
                "resid": float(y[i] - pred),
            })
    return pd.DataFrame(rows)


def _correct(pred: np.ndarray, q: pd.DataFrame, residuals: pd.DataFrame,
             strength: float) -> np.ndarray:
    """Apply shrunk LOO date/crop residual medians (no query truth)."""
    if residuals.empty:
        return pred.copy()
    exact: dict[tuple[Any, ...], tuple[float, int]] = {}
    seasonal: dict[tuple[Any, ...], tuple[float, int]] = {}
    for key, g in residuals.groupby(["date", "crop_type"], dropna=False):
        v = g["resid"].to_numpy(float)
        exact[key if isinstance(key, tuple) else (key,)] = (float(np.median(v)), len(v))
    for key, g in residuals.groupby(["_doy", "crop_type"], dropna=False):
        v = g["resid"].to_numpy(float)
        seasonal[key if isinstance(key, tuple) else (key,)] = (float(np.median(v)), len(v))
    z = pred.copy()
    for j, (_, row) in enumerate(q.iterrows()):
        e = exact.get((row["date"], str(row["crop_type"])))
        if e is None:
            e = seasonal.get((int(row["_doy"]), str(row["crop_type"])))
        if e is None:
            continue
        med, n = e
        # Conservative empirical-Bayes shrinkage: at least 12 observations
        # are needed for a substantial correction.
        alpha = float(np.clip(strength * n / (n + 12.0), 0.0, 0.5))
        z[j] = z[j] + alpha * float(np.clip(med, -0.15, 0.15))
    return z


def _score(a: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    e = np.asarray(a, float) - np.asarray(y, float)
    ok = np.isfinite(e)
    return float(np.sqrt(np.mean(e[ok] ** 2))), float(np.mean(np.abs(e[ok])))


def run(train: pd.DataFrame, private: pd.DataFrame, years: list[int], random_seeds: list[int]):
    rows: list[dict[str, Any]] = []
    pred_records: list[pd.DataFrame] = []
    folds: list[tuple[str, int, pd.DataFrame, np.ndarray]] = []
    for year in years:
        f, h = exact_doy_fold(train, private, year)
        folds.append(("exact_doy", year, f, h))
    for seed in random_seeds:
        f, h = random_private_like_fold(private, seed)
        folds.append(("private_random", seed, f, h))

    for mode, fold_id, fold, hide in folds:
        if not int(hide.sum()):
            continue
        routed, d = _routing_predictions(fold, hide, k=8)
        # Verify the independent implementation against production soft path.
        prod = predict_private(fold, k=8, bin_days=30, use_date_prior=True, date_weight=1.0)
        q = d.loc[hide, ["anon_polygon_id", "date", "_doy", "crop_type", "_truth", "_true_src"]].copy()
        q["date"] = pd.to_datetime(q["date"])
        pp = prod.copy(); pp["date"] = pd.to_datetime(pp["date"])
        pp = q[["anon_polygon_id", "date"]].merge(pp, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
        q["prod"] = pp["primary_ndvi_pred"].to_numpy(float)
        for name, a in routed.items():
            q[name] = a[hide]
        # Lag is included for the diversity diagnostic.
        lag = predict_private_lag(fold, k=16, degree=3, bin_days=30,
                                  use_date_prior=True, date_weight=1.0)
        lag["date"] = pd.to_datetime(lag["date"])
        q["lag"] = q[["anon_polygon_id", "date"]].merge(
            lag, on=["anon_polygon_id", "date"], how="left", validate="one_to_one"
        )["primary_ndvi_pred"].to_numpy(float)
        # The independently reconstructed soft output should be numerically
        # close to production; report the discrepancy instead of silently
        # conflating implementations.
        soft_delta = np.nanmax(np.abs(q["soft"].to_numpy(float) - q["prod"].to_numpy(float)))
        q["fold"] = str(fold_id); q["mode"] = mode
        pred_records.append(q)
        y = q["_truth"].to_numpy(float)
        for name in ["prod", "soft", "hard", "uniform", "oracle_true", "lag"]:
            rmse, mae = _score(q[name].to_numpy(float), y)
            rows.append({"mode": mode, "fold": fold_id, "method": name,
                         "n": len(q), "rmse": rmse, "mae": mae,
                         "soft_max_abs_delta": float(soft_delta)})
        # Source-stratified routing metrics.
        for source in SOURCES:
            z = q["_true_src"].astype(str).to_numpy() == source
            if not z.any():
                continue
            for name in ["prod", "soft", "hard", "oracle_true", "lag"]:
                rmse, mae = _score(q.loc[z, name].to_numpy(float), y[z])
                rows.append({"mode": mode, "fold": fold_id,
                             "method": f"{name}|true_{source}", "n": int(z.sum()),
                             "rmse": rmse, "mae": mae,
                             "soft_max_abs_delta": float(soft_delta)})

        # LOO residual correction is fit from known rows only.
        res = _loo_residuals(d, hide, k=8)
        for base_name in ["prod", "soft", "hard"]:
            for strength in [0.15, 0.30, 0.50, 0.75, 1.0]:
                corrected = _correct(q[base_name].to_numpy(float), q, res, strength)
                rmse, mae = _score(corrected, y)
                rows.append({"mode": mode, "fold": fold_id,
                             "method": f"{base_name}|loo_date_crop_{strength:.2f}",
                             "n": len(q), "rmse": rmse, "mae": mae,
                             "soft_max_abs_delta": float(soft_delta),
                             "loo_rows": int(len(res))})
        # Small convex blend grid, a low-rank/diversity proxy.  We only use
        # local candidates here and evaluate a fixed conservative grid.
        cand = ["prod", "hard", "uniform", "lag"]
        best = (float("inf"), None, None)
        for a in np.linspace(0.0, 1.0, 11):
            for b in np.linspace(0.0, 1.0 - a, 11):
                for c in np.linspace(0.0, 1.0 - a - b, 11):
                    w = np.array([a, b, c, 1.0 - a - b - c])
                    z = sum(wi * q[name].to_numpy(float) for wi, name in zip(w, cand))
                    rmse, mae = _score(z, y)
                    if rmse < best[0]:
                        best = (rmse, mae, w.copy())
        rows.append({"mode": mode, "fold": fold_id, "method": "convex_diversity_best",
                     "n": len(q), "rmse": float(best[0]), "mae": float(best[1]),
                     "weights": json.dumps(dict(zip(cand, best[2].tolist())))})

    pred = pd.concat(pred_records, ignore_index=True) if pred_records else pd.DataFrame()
    return pd.DataFrame(rows), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DATA / "train_dataset.csv")
    ap.add_argument("--private", type=Path, default=DATA / "private_features.csv")
    ap.add_argument("--years", default="2019,2020,2021,2022,2023,2024")
    ap.add_argument("--random-seeds", default="0,1")
    args = ap.parse_args()
    train = pd.read_csv(args.train, low_memory=False)
    private = pd.read_csv(args.private, low_memory=False)
    years = [int(x) for x in str(args.years).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.random_seeds).split(",") if x.strip()]
    results, preds = run(train, private, years, seeds)
    out_dir = ROOT / "research"
    results.to_csv(out_dir / "overnight_root_source_results.csv", index=False)
    preds.to_csv(out_dir / "overnight_root_source_preds.csv", index=False)
    # Aggregate only comparable full-fold methods (exclude source-stratified
    # and per-fold convex rows from the simple mean).
    simple = results[~results.method.str.contains("\\|true_|convex_diversity")].copy()
    agg = (simple.groupby(["mode", "method"], as_index=False)
           .agg(folds=("fold", "count"), n=("n", "sum"),
                rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean")))
    agg.to_csv(out_dir / "overnight_root_source_agg.csv", index=False)
    lines = [
        "# Overnight source-aware evaluator (research-only)", "",
        "Скрипт: `overnight_source_eval_root.py`. Входы и `outputs/` не изменяются.",
        "Проверены exact hidden-DOY folds 2019–2024 и private-like random folds (seeds "
        + ", ".join(map(str, seeds)) + ").", "",
        "## Интерпретация", "",
        "`soft` — независимая реконструкция текущего source-posterior пути; `hard` — "
        "modal source; `oracle_true` — диагностический верхний предел с истинным "
        "источником, недоступным на private; `loo_date_crop` использует только "
        "leave-one-out residuals известных строк; `convex_diversity_best` — "
        "фиксированная малая convex-сетка локальных предикторов.", "",
        "## Файлы", "",
        "- `overnight_root_source_results.csv` — row/fold/source metrics.",
        "- `overnight_root_source_agg.csv` — агрегаты по режиму и методу.",
        "- `overnight_root_source_preds.csv` — предикторы и truth для повторного анализа.",
        "",
        "Решение о замене production принимается только по устойчивому улучшению "
        "на нескольких годах и обоих типах маски; этот скрипт production не пишет.",
    ]
    (out_dir / "overnight_root_source_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(agg.sort_values(["mode", "rmse_mean"]).to_string(index=False))


if __name__ == "__main__":
    main()

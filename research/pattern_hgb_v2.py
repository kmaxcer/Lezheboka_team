"""Pattern-matched masked-training HGB experiment.

The competition gap mask is not iid in time: it is a fixed AOI/DOY pattern
sampled once.  The starter model trains its OOF rows with random folds.  This
research script asks whether training on pseudo-gaps with the *same observable
AOI/DOY pattern* improves the conditional mean.  It never uses validation
labels in features and writes only ``research/pattern_hgb_v2*`` artifacts.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
from agropulse.pipeline import build_features, FULL_FEATURES  # noqa: E402

TARGET = "primary_ndvi"
DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi",
    "era5_temp_c", "era5_precip_mm", "year", TARGET, "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def _clear(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    d = frame.copy()
    mask = np.asarray(mask, bool)
    for c in DYNAMIC:
        if c in d:
            d.loc[mask, c] = np.nan
    d["is_synthetic_gap"] = mask
    d["year"] = d["year"].fillna(d.date.dt.year).astype(int)
    d["doy"] = d["doy"].fillna(d.date.dt.dayofyear).astype(int)
    return d


def _pattern_mask(frame: pd.DataFrame, hidden_doys: dict[str, set[int]], years_exclude: set[int] | None = None) -> np.ndarray:
    dt = pd.to_datetime(frame["date"])
    ids = frame["anon_polygon_id"].astype(str).to_numpy()
    doy = dt.dt.dayofyear.to_numpy(int)
    yr = dt.dt.year.to_numpy(int)
    out = np.fromiter(
        (int((x not in (years_exclude or set())) and (dd in hidden_doys.get(pid, set())))
         for x, pid, dd in zip(yr, ids, doy)),
        dtype=bool, count=len(frame),
    )
    out &= frame[TARGET].notna().to_numpy(bool)
    return out


def _random_mask(frame: pd.DataFrame, seed: int, frac: float = 0.15, years_exclude: set[int] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = pd.to_datetime(frame["date"])
    pool = frame[TARGET].notna().to_numpy(bool).copy()
    if years_exclude:
        pool &= ~d.dt.year.isin(years_exclude).to_numpy()
    out = np.zeros(len(frame), bool)
    for _, ix in frame.loc[pool].groupby("anon_polygon_id", sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        # Keep the same per-AOI/year cadence as the canonical private-like CV.
        for _, jj in frame.iloc[ii].groupby(d.iloc[ii].dt.year, sort=False).groups.items():
            # ``frame`` has its original RangeIndex, and groupby returns those
            # labels even after the ``iloc`` subset.
            aa = np.asarray(jj, dtype=int)
            n = max(1, int(round(frac * len(aa))))
            out[rng.choice(aa, size=min(n, len(aa)), replace=False)] = True
    return out


def _model(kind: str, seed: int) -> HistGradientBoostingRegressor:
    specs = {
        "default": dict(learning_rate=.035, max_iter=300, max_leaf_nodes=48, min_samples_leaf=35, l2_regularization=8.0),
        "small": dict(learning_rate=.03, max_iter=400, max_leaf_nodes=31, min_samples_leaf=45, l2_regularization=12.0),
        "smooth": dict(learning_rate=.025, max_iter=450, max_leaf_nodes=24, min_samples_leaf=60, l2_regularization=16.0),
        "wide": dict(learning_rate=.03, max_iter=350, max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=8.0),
    }
    return HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **specs[kind])


def _metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    e = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    hidden = private.loc[private["is_synthetic_gap"].fillna(False).astype(bool)].copy()
    hidden["_doy"] = hidden.date.dt.dayofyear.astype(int)
    hd = hidden.groupby("anon_polygon_id") ["_doy"].apply(set).to_dict()
    # Baseline exact predictions are used only for a side-by-side comparison.
    exact_saved = pd.read_csv(RESEARCH / "exact_compare_preds.csv", parse_dates=["date"])
    rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for year in (2019, 2020, 2021, 2022, 2023, 2024):
        fold, truth = make_fold(train.copy(), private.copy(), year)
        fold["year"] = fold["year"].fillna(fold.date.dt.year).astype(int)
        fold["doy"] = fold["doy"].fillna(fold.date.dt.dayofyear).astype(int)
        qmask = fold["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
        yq = fold.loc[qmask, "_truth"].to_numpy(float)
        # Pattern rows from other years are the main training set.  Add two
        # independent random masks to avoid over-specialising to one DOY set.
        masks = [_pattern_mask(fold, hd, {year}), _random_mask(fold, 1900 + year, years_exclude={year}), _random_mask(fold, 2900 + year, years_exclude={year})]
        blocks, targets = [], []
        for pm in masks:
            if not pm.any():
                continue
            fr = _clear(fold, pm)
            obs = fr[TARGET].where(~pm)
            xx = build_features(fr, obs, pd.Series(pm, index=fr.index))
            blocks.append(xx.loc[pm, FULL_FEATURES]); targets.append(fold.loc[pm, "_truth"].astype(float))
        # Validation features are built with the exact hidden pattern.
        vf = _clear(fold, qmask)
        vx = build_features(vf, vf[TARGET].where(~qmask), pd.Series(qmask, index=vf.index)).loc[qmask, FULL_FEATURES]
        print("year", year, "train", sum(len(x) for x in blocks), "query", len(yq), flush=True)
        for kind in ("default", "small", "smooth", "wide"):
            model = _model(kind, 42)
            model.fit(pd.concat(blocks, ignore_index=True), pd.concat(targets, ignore_index=True))
            pred = np.clip(model.predict(vx), -0.2, 1.1)
            rm, ma = _metric(yq, pred)
            rows.append({"protocol": "exact", "year": year, "kind": kind, "rmse": rm, "mae": ma, "n": len(yq), "train_n": sum(len(x) for x in blocks)})
            pred_rows.append(pd.DataFrame({"year": year, "anon_polygon_id": fold.loc[qmask, "anon_polygon_id"].to_numpy(), "date": fold.loc[qmask, "date"].to_numpy(), "truth": yq, "method": kind, "pred": pred}))
        print("  done", year, "elapsed", round(time.time() - t0, 1), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESEARCH / "pattern_hgb_v2_results.csv", index=False)
    if pred_rows:
        pp = pd.concat(pred_rows, ignore_index=True)
        pp.to_csv(RESEARCH / "pattern_hgb_v2_predictions.csv", index=False)
    agg = out.groupby("kind", as_index=False).apply(lambda g: pd.Series({"n": int(g.n.sum()), "rmse_pooled": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))), "mae_pooled": float(np.average(g.mae, weights=g.n))}), include_groups=False).reset_index(drop=True).sort_values("rmse_pooled")
    agg.to_csv(RESEARCH / "pattern_hgb_v2_aggregate.csv", index=False)
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

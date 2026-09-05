"""Quick validation of the bundled HistGradientBoosting candidate.

The archive model is evaluated on random masks of observed private rows.  It
is intentionally separate from the tuned production build because its feature
builder is heavier and has a different validation protocol.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
ARCHIVE_SRC = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
sys.path.insert(0, str(ARCHIVE_SRC))
from agropulse.pipeline import fit_final_model, predict_submission  # noqa: E402

DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def make_masked(private: pd.DataFrame, seed: int, frac: float = 0.15):
    d = private.copy().sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(d), dtype=bool)
    pool = d["primary_ndvi"].notna()
    years = d["date"].dt.year
    for _, ix in d.loc[pool].groupby(["anon_polygon_id", years], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        n = max(1, int(round(frac * len(ii))))
        mask[rng.choice(ii, size=min(n, len(ii)), replace=False)] = True
    for col in DYNAMIC:
        if col in d.columns:
            d.loc[mask, col] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    return d, mask


def score(pred: pd.DataFrame, q: pd.DataFrame, mask: np.ndarray):
    truth = q.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy()
    truth["date"] = pd.to_datetime(truth["date"])
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    z = truth.merge(pred, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    e = z["primary_ndvi_pred"].to_numpy(float) - z["_truth"].to_numpy(float)
    return float(np.sqrt(np.nanmean(e * e))), float(np.nanmean(np.abs(e))), int(np.isfinite(e).sum())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated mask seeds")
    args = ap.parse_args()
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    rows = []
    for seed in seeds:
        masked, mask = make_masked(private, seed)
        tr = train.copy(); tr["_origin"] = "train"; tr["_test_order"] = np.nan
        te = masked.copy(); te["_origin"] = "test"; te["_test_order"] = np.arange(len(te))
        # Match the archive loader's sorted combined reference.
        ref = pd.concat([tr, te], ignore_index=True, sort=False)
        ref = ref.sort_values(["anon_polygon_id", "date", "_origin"]).reset_index(drop=True)
        ref["year"] = ref["year"].fillna(ref["date"].dt.year).astype(int)
        ref["doy"] = ref["doy"].fillna(ref["date"].dt.dayofyear).astype(int)
        model, _ = fit_final_model(ref, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            path = Path(tf.name)
        pred = predict_submission(ref, model, path)
        path.unlink(missing_ok=True)
        pred.to_csv(ROOT / "research" / f"hgb_cv_pred_seed{seed}.csv", index=False)
        rmse, mae, n = score(pred, masked, mask)
        rows.append({"seed": seed, "rmse": rmse, "mae": mae, "n": n})
        print(seed, rmse, mae, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "research" / "hgb_cv_results.csv", index=False)
    print("mean", out[["rmse", "mae"]].mean().to_dict())


if __name__ == "__main__":
    main()

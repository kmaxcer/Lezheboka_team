"""Scratch pseudo-CV on private known rows.

This file is intentionally separate from production code.  It masks a
deterministic 15% of known rows per AOI/year (the observed synthetic-gap rate),
then compares source-aware local interpolation variants with/without adding
historical train rows to the local neighbour frame.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from infer import predict_private  # noqa: E402


DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def make_masked(private: pd.DataFrame, seed: int, frac: float = 0.15):
    d = private.copy().sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(d), dtype=bool)
    pool = d["primary_ndvi"].notna()
    for _, ix in d.loc[pool].groupby(["anon_polygon_id", d.loc[pool, "date"].dt.year], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        n = max(1, int(round(frac * len(ii))))
        mask[rng.choice(ii, size=min(n, len(ii)), replace=False)] = True
    for col in DYNAMIC:
        if col in d.columns:
            d.loc[mask, col] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    return d, mask


def with_history(masked: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    # train and private have disjoint date keys for shared IDs; concatenate
    # only common columns to avoid accidental schema-dependent behaviour.
    tr = train.copy()
    tr["is_synthetic_gap"] = False
    cols = [c for c in masked.columns if c in tr.columns]
    return pd.concat([tr[cols], masked[cols]], ignore_index=True, sort=False)


def score(out: pd.DataFrame, q: pd.DataFrame, mask: np.ndarray):
    qk = q.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy()
    qk["date"] = pd.to_datetime(qk["date"])
    out = out.copy()
    out["date"] = pd.to_datetime(out["date"])
    z = qk.merge(out, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    e = z["primary_ndvi_pred"].to_numpy(float) - z["_truth"].to_numpy(float)
    ok = np.isfinite(e)
    return float(np.sqrt(np.mean(e[ok] ** 2))), float(np.mean(np.abs(e[ok]))), int(ok.sum())


def main():
    tr = pd.read_csv(ROOT / "train_dataset.csv", parse_dates=["date"])
    pr = pd.read_csv(ROOT / "private_features.csv", parse_dates=["date"])
    train_ids = set(tr.anon_polygon_id)
    rows = []
    # Three seeds are enough for a quick directional screen; the hidden mask
    # itself is approximately 15% and per-seed variation is substantial.
    for seed in [0, 1, 2]:
        masked, mask = make_masked(pr, seed)
        q = masked
        shared_q = q["anon_polygon_id"].isin(train_ids).to_numpy() & mask
        new_q = (~q["anon_polygon_id"].isin(train_ids)).to_numpy() & mask
        for hist_name, frame in [("private", masked), ("history", with_history(masked, tr))]:
            for k in [6, 8, 10]:
                for dw in [0.0, 1.0]:
                    out = predict_private(frame, train=None, k=k, bin_days=30,
                                           use_date_prior=True, date_weight=dw)
                    rm, mae, n = score(out, q, mask)
                    r2, _, _ = score(out, q, mask & shared_q)
                    r3, _, _ = score(out, q, mask & new_q)
                    rows.append((seed, hist_name, k, dw, rm, mae, n, r2, r3,
                                 int(shared_q.sum()), int(new_q.sum())))
                    print(seed, hist_name, k, dw, rm, r2, r3, flush=True)
    out = pd.DataFrame(rows, columns=["seed", "frame", "k", "date_weight", "rmse", "mae", "n",
                                      "rmse_shared", "rmse_new", "n_shared", "n_new"])
    out.to_csv(HERE / "research" / "_private_cv_results.csv", index=False)
    print("AGGREGATE")
    print(out.groupby(["frame", "k", "date_weight"])[["rmse", "rmse_shared", "rmse_new"]].mean().sort_values("rmse").head(30).to_string())


if __name__ == "__main__":
    main()

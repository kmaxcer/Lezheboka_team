"""Evaluate the downloaded teammate archive model on independent private-like masks.

This is an audit-only script. It never modifies production outputs; predictions are
written under research/ with a distinct prefix.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
ARCHIVE_SRC = ROOT / "_teammate_agropulse" / "agropulse_max_score" / "src"
sys.path.insert(0, str(ARCHIVE_SRC))
from agropulse.pipeline import fit_final_model, predict_submission  # noqa: E402

DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def make_masked(private: pd.DataFrame, seed: int, frac: float = 0.15,
                year: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    d = private.copy().sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(d), dtype=bool)
    pool = d["primary_ndvi"].notna()
    if year is not None:
        pool &= d.date.dt.year.eq(int(year))
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


def build_ref(train: pd.DataFrame, masked: pd.DataFrame) -> pd.DataFrame:
    tr = train.copy()
    tr["_origin"] = "train"
    tr["_test_order"] = np.nan
    te = masked.copy()
    te["_origin"] = "test"
    te["_test_order"] = np.arange(len(te))
    ref = pd.concat([tr, te], ignore_index=True, sort=False)
    ref = ref.sort_values(["anon_polygon_id", "date", "_origin"]).reset_index(drop=True)
    ref["year"] = ref["year"].fillna(ref["date"].dt.year).astype(int)
    ref["doy"] = ref["doy"].fillna(ref["date"].dt.dayofyear).astype(int)
    return ref


def score(pred: pd.DataFrame, q: pd.DataFrame, mask: np.ndarray) -> tuple[float, float, int]:
    truth = q.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy()
    truth["date"] = pd.to_datetime(truth["date"])
    pp = pred.copy()
    pp["date"] = pd.to_datetime(pp["date"])
    z = truth.merge(pp, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    y = z["_truth"].to_numpy(float)
    p = z["primary_ndvi_pred"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), float(np.mean(np.abs(p[ok] - y[ok]))), int(ok.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--years", default="all,2025")
    ap.add_argument("--frac", type=float, default=0.15)
    args = ap.parse_args()
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    rows: list[dict[str, object]] = []
    for ys in str(args.years).split(","):
        year = None if ys.strip().lower() in {"all", "none", ""} else int(ys)
        for ss in str(args.seeds).split(","):
            if not ss.strip():
                continue
            seed = int(ss)
            t0 = time.time()
            masked, mask = make_masked(private, seed, args.frac, year)
            ref = build_ref(train, masked)
            model, _ = fit_final_model(ref, seed=42)
            out = ROOT / "research" / f"archive_model_pred_{('all' if year is None else year)}_seed{seed}.csv"
            predict_submission(ref, model, out)
            pp = pd.read_csv(out)
            rmse, mae, n = score(pp, masked, mask)
            rec = {"scenario": "all" if year is None else f"year{year}", "seed": seed,
                   "frac": args.frac, "n": n, "rmse": rmse, "mae": mae,
                   "seconds": round(time.time() - t0, 2), "path": str(out)}
            rows.append(rec)
            print(rec, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "research" / "archive_model_mask_results.csv", index=False)
    print(res.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Compare production-safe candidate imputers on private-style random masks.

This is a diagnostic script only.  It creates masks from observed private rows,
keeps the truth in a side column, and scores only those rows.  The script is
deliberately self-contained so the final tuned build can cite the same numbers
without changing the competition inputs.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(ROOT / "src"))
from infer import predict_private  # noqa: E402
from infer_lag import predict_private_lag  # noqa: E402

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
    d["_true_src"] = np.select(
        [d["s2_ndvi"].notna(), d["landsat_ndvi"].notna(), d["modis_ndvi"].notna()],
        ["s2", "landsat", "modis"], default="none",
    )
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


def with_history(masked: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    tr = train.copy()
    tr["is_synthetic_gap"] = False
    cols = [c for c in masked.columns if c in tr.columns]
    return pd.concat([tr[cols], masked[cols]], ignore_index=True, sort=False)


def score(out: pd.DataFrame, q: pd.DataFrame, mask: np.ndarray):
    truth = q.loc[mask, ["anon_polygon_id", "date", "_truth", "_true_src"]].copy()
    truth["date"] = pd.to_datetime(truth["date"])
    pred = out.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    z = truth.merge(pred, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    e = z["primary_ndvi_pred"].to_numpy(float) - z["_truth"].to_numpy(float)
    ok = np.isfinite(e)
    result = {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean(e[ok] ** 2))),
              "mae": float(np.mean(np.abs(e[ok])))}
    for src in ("s2", "landsat", "modis"):
        take = ok & z["_true_src"].eq(src).to_numpy()
        result[f"rmse_{src}"] = float(np.sqrt(np.mean(e[take] ** 2))) if take.any() else np.nan
        result[f"n_{src}"] = int(take.sum())
    return result


def main() -> None:
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    train_ids = set(train["anon_polygon_id"])
    rows: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        masked, mask = make_masked(private, seed)
        hist = with_history(masked, train)
        # The history frame is intentionally passed without a second train
        # argument: train rows are now eligible local anchors and calibration
        # observations, while hidden private rows remain masked.
        candidates = {
            "base_k6": predict_private(masked, train=train, k=6, bin_days=30, date_weight=1.0),
            "base_k8": predict_private(masked, train=train, k=8, bin_days=30, date_weight=1.0),
            "history_k6": predict_private(hist, k=6, bin_days=30, date_weight=1.0),
            "history_k8": predict_private(hist, k=8, bin_days=30, date_weight=1.0),
            "lag_k16_d2": predict_private_lag(masked, train=train, k=16, degree=2, bin_days=30, date_weight=1.0),
            "lag_k16_d3": predict_private_lag(masked, train=train, k=16, degree=3, bin_days=30, date_weight=1.0),
            "lag_history_k16_d2": predict_private_lag(hist, k=16, degree=2, bin_days=30, date_weight=1.0),
            "lag_history_k16_d3": predict_private_lag(hist, k=16, degree=3, bin_days=30, date_weight=1.0),
        }
        for name, pred in candidates.items():
            r = score(pred, masked, mask)
            r.update(seed=seed, method=name)
            rows.append(r)
            print(seed, name, r["rmse"], flush=True)

        # Conservative blends are included for stability screening.  Use the
        # same key order for all candidates and blend only finite predictions.
        keyed = {}
        for name, pred in candidates.items():
            z = pred.copy(); z["date"] = pd.to_datetime(z["date"])
            keyed[name] = masked[["anon_polygon_id", "date"]].merge(
                z, on=["anon_polygon_id", "date"], how="left", validate="one_to_one"
            )["primary_ndvi_pred"].to_numpy(float)
        blend_specs = {
            "blend_hist_lag25": ("history_k6", "lag_history_k16_d2", 0.25),
            "blend_hist_lag50": ("history_k6", "lag_history_k16_d2", 0.50),
            "blend_hist_lag25_d3": ("history_k6", "lag_history_k16_d3", 0.25),
        }
        for name, (a, b, wb) in blend_specs.items():
            arr = (1.0 - wb) * keyed[a] + wb * keyed[b]
            pred = pd.DataFrame({"anon_polygon_id": masked.loc[mask, "anon_polygon_id"].to_numpy(),
                                 "date": masked.loc[mask, "date"].dt.strftime("%Y-%m-%d").to_numpy(),
                                 "primary_ndvi_pred": arr[mask]})
            r = score(pred, masked, mask); r.update(seed=seed, method=name)
            rows.append(r)
            print(seed, name, r["rmse"], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "research" / "tuned_cv_results.csv", index=False)
    agg = out.groupby("method", as_index=False).agg(
        rmse=("rmse", "mean"), mae=("mae", "mean"), n=("n", "sum"),
        rmse_s2=("rmse_s2", "mean"), rmse_landsat=("rmse_landsat", "mean"),
        rmse_modis=("rmse_modis", "mean"),
    ).sort_values("rmse")
    agg.to_csv(ROOT / "research" / "tuned_cv_aggregate.csv", index=False)
    print("AGGREGATE")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()

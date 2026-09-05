"""2025 hidden-date proxy for fast local/lag candidate screening.

The real private mask contains 925 hidden rows in 2025.  For each hidden
calendar date this script samples the same number of *known* AOI rows at that
date, masks their dynamic fields, and scores against their retained truth.
This preserves the date multiplicity and acquisition schedule while avoiding
the unavailable labels of the actual hidden rows.
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


def make_mask(private: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    d = private.copy().sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["_true_src"] = np.select(
        [d["s2_ndvi"].notna(), d["landsat_ndvi"].notna(), d["modis_ndvi"].notna()],
        ["s2", "landsat", "modis"], default="none",
    )
    d["_hold"] = False
    d["is_synthetic_gap"] = d["is_synthetic_gap"].fillna(False).astype(bool)
    y25 = d["date"].dt.year.eq(2025)
    real_hidden = d["is_synthetic_gap"] & y25
    hidden_counts = d.loc[real_hidden].groupby("date").size().to_dict()
    candidates = d[y25 & ~d["is_synthetic_gap"] & d["primary_ndvi"].notna()]
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for date, count in hidden_counts.items():
        ix = candidates.index[candidates["date"].eq(date)].to_numpy()
        if len(ix):
            take = min(int(count), len(ix))
            selected.extend(rng.choice(ix, size=take, replace=False).tolist())
    # There is a single hidden date without known peers; fill any shortfall
    # from known rows in the same 16-day phase, preserving the 925-row rate.
    need = int(real_hidden.sum()) - len(selected)
    if need > 0:
        pool = candidates.index[~candidates.index.isin(selected)].to_numpy()
        if len(pool):
            selected.extend(rng.choice(pool, size=min(need, len(pool)), replace=False).tolist())
    hold = np.zeros(len(d), dtype=bool)
    hold[np.asarray(selected, dtype=int)] = True
    # Keep the actual hidden rows hidden too, but score only sampled known rows.
    mask = hold | real_hidden.to_numpy()
    for col in DYNAMIC:
        if col in d.columns:
            d.loc[mask, col] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    d.loc[hold, "_hold"] = True
    return d, hold


def score(pred: pd.DataFrame, d: pd.DataFrame, hold: np.ndarray) -> dict[str, object]:
    q = d.loc[hold, ["anon_polygon_id", "date", "_truth", "_true_src"]].copy()
    p = pred.copy(); p["date"] = pd.to_datetime(p["date"])
    q["date"] = pd.to_datetime(q["date"])
    z = q.merge(p, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    e = z["primary_ndvi_pred"].to_numpy(float) - z["_truth"].to_numpy(float)
    ok = np.isfinite(e)
    out: dict[str, object] = {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean(e[ok]**2))), "mae": float(np.mean(np.abs(e[ok])))}
    for s in ("s2", "landsat", "modis"):
        take = ok & z["_true_src"].eq(s).to_numpy()
        out[f"n_{s}"] = int(take.sum())
        out[f"rmse_{s}"] = float(np.sqrt(np.mean(e[take]**2))) if take.any() else np.nan
    return out


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    rows: list[dict[str, object]] = []
    row_parts: list[pd.DataFrame] = []
    for seed in (0, 1, 2):
        d, hold = make_mask(pr, seed)
        candidates = {
            "base_k6": predict_private(d, train=tr, k=6, bin_days=30, date_weight=1.0),
            "base_k8": predict_private(d, train=tr, k=8, bin_days=30, date_weight=1.0),
            "lag_k12_d2": predict_private_lag(d, train=tr, k=12, degree=2, bin_days=30, date_weight=1.0),
            "lag_k16_d3": predict_private_lag(d, train=tr, k=16, degree=3, bin_days=30, date_weight=1.0),
            "lag_k24_d2": predict_private_lag(d, train=tr, k=24, degree=2, bin_days=30, date_weight=1.0),
        }
        for name, pred in candidates.items():
            r = score(pred, d, hold); r.update(seed=seed, cohort="2025_same_hidden_dates", method=name); rows.append(r)
            print(seed, name, r["rmse"], flush=True)
        # Save one compact row-level table for source-aware diagnostics.
        q = d.loc[hold, ["anon_polygon_id", "date", "_truth", "_true_src"]].copy()
        q["seed"] = seed
        for name, pred in candidates.items():
            z = pred.copy(); z["date"] = pd.to_datetime(z["date"])
            q = q.merge(z.rename(columns={"primary_ndvi_pred": name}), on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
        row_parts.append(q)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "research" / "teammate_sweep_root_2025.csv", index=False)
    rows_all = pd.concat(row_parts, ignore_index=True)
    rows_all.to_csv(ROOT / "research" / "teammate_sweep_root_2025_rows.csv", index=False)
    agg = out.groupby("method", as_index=False).agg(rmse=("rmse", "mean"), mae=("mae", "mean"), n=("n", "sum"), rmse_s2=("rmse_s2", "mean"), rmse_landsat=("rmse_landsat", "mean"), rmse_modis=("rmse_modis", "mean")).sort_values("rmse")
    agg.to_csv(ROOT / "research" / "teammate_sweep_root_2025_aggregate.csv", index=False)
    (ROOT / "research" / "teammate_sweep_root_2025.md").write_text(
        "# Root 2025 hidden-date proxy\n\n"
        "For each actual hidden 2025 date, sampled the same number of known AOI rows; 3 seeds.\n\n"
        + agg.to_string(index=False) + "\n\n"
        "This is a proxy: actual hidden labels remain unavailable. No production outputs were changed.\n",
        encoding="utf-8",
    )
    print("AGGREGATE\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()

"""Probe observable source-schedule classifiers (no model fitting).

This diagnostic compares same-date spatial evidence with temporal/AOI and
day-of-year schedule evidence on four private-like masks.  It is intentionally
separate from production source-route files; target/source labels are retained
only as an evaluation sidecar.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(R))
import source_expert_route_v2 as route  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
SRCN = {"s2": 0, "landsat": 1, "modis": 2}


def srcs(d: pd.DataFrame) -> np.ndarray:
    return np.select([d.s2_ndvi.notna(), d.landsat_ndvi.notna(), d.modis_ndvi.notna()], [0, 1, 2], -1).astype(int)


def mode_weighted(vals: list[tuple[int, float]], prior: np.ndarray | None = None) -> np.ndarray:
    z = np.zeros(3, float)
    for s, w in vals:
        if s >= 0: z[s] += float(w)
    if z.sum() <= 0:
        return np.asarray(prior if prior is not None else np.ones(3) / 3, float)
    return z / z.sum()


def classify(train: pd.DataFrame, private: pd.DataFrame, hold: np.ndarray) -> pd.DataFrame:
    pm, gaps = route._masked_private(private, hold)
    tr = train.copy(); tr[DATE] = pd.to_datetime(tr[DATE]); pm[DATE] = pd.to_datetime(pm[DATE])
    tr["_src"] = srcs(tr); pm["_src"] = srcs(pm)
    tr["_yr"] = tr[DATE].dt.year.astype(int); pm["_yr"] = pm[DATE].dt.year.astype(int)
    tr["_doy"] = tr[DATE].dt.dayofyear.astype(int); pm["_doy"] = pm[DATE].dt.dayofyear.astype(int)
    tr["_num"] = pd.to_numeric(tr[ID].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(-1).astype(int)
    pm["_num"] = pd.to_numeric(pm[ID].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(-1).astype(int)
    tr["_crop"] = tr.crop_type.fillna("unknown").astype(str); pm["_crop"] = pm.crop_type.fillna("unknown").astype(str)
    # Visible population includes train and non-gap private rows.  Queries are
    # private rows only; no target values are needed for this classifier.
    vis = pd.concat([tr, pm.loc[~gaps]], ignore_index=True, sort=False)
    vis = vis[vis._src >= 0].reset_index(drop=True)
    by_date = {k: g for k, g in vis.groupby(DATE, sort=False)}
    by_idyr = {(str(a), int(y)): g for (a, y), g in vis.groupby([ID, "_yr"], sort=False)}
    by_id = {str(a): g for a, g in vis.groupby(ID, sort=False)}
    by_doy = {int(k): g for k, g in vis.groupby("_doy", sort=False)}
    q = private.loc[hold, [ID, DATE, "crop_type"]].copy().reset_index(drop=True); q[DATE] = pd.to_datetime(q[DATE]); q["_num"] = pd.to_numeric(q[ID].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(-1).astype(int); q["_crop"] = q.crop_type.fillna("unknown").astype(str); q["_yr"] = q[DATE].dt.year.astype(int); q["_doy"] = q[DATE].dt.dayofyear.astype(int)
    true = srcs(private)[hold]
    records = []
    # Existing crop-neighbour modes at each radius, plus temporal and schedule
    # modes.  All features are computed after masking ``hold``.
    for _, row in q.iterrows():
        aid, num, crop, dt, yr, doy = str(row[ID]), int(row._num), str(row._crop), pd.Timestamp(row[DATE]), int(row._yr), int(row._doy)
        vals: dict[str, np.ndarray] = {}
        gd = by_date.get(dt, vis.iloc[0:0])
        for radius in (1, 2, 4, 8, 16, 32):
            z = gd[(np.abs(gd._num - num) <= radius) & (gd._crop == crop)]
            vals[f"sp_crop_{radius}"] = np.bincount(z._src.to_numpy(int), minlength=3).astype(float) if len(z) else np.zeros(3)
            z = gd[np.abs(gd._num - num) <= radius]
            vals[f"sp_all_{radius}"] = np.bincount(z._src.to_numpy(int), minlength=3).astype(float) if len(z) else np.zeros(3)
        zy = by_idyr.get((aid, yr), vis.iloc[0:0])
        if len(zy):
            dist = np.abs((zy[DATE] - dt).dt.days.to_numpy(float)); order = np.argsort(dist)
            for k in (1, 2, 4, 8):
                ii = order[: min(k, len(order))]; vals[f"temp_{k}"] = np.bincount(zy.iloc[ii]._src.to_numpy(int), minlength=3).astype(float)
        else:
            for k in (1, 2, 4, 8): vals[f"temp_{k}"] = np.zeros(3)
        za = by_id.get(aid, vis.iloc[0:0])
        if len(za):
            dd = np.abs(za._doy.to_numpy(int) - doy); dd = np.minimum(dd, 366 - dd); order = np.argsort(dd)
            for k in (2, 4, 8, 16):
                ii = order[: min(k, len(order))]; vals[f"cross_{k}"] = np.bincount(za.iloc[ii]._src.to_numpy(int), minlength=3).astype(float)
        else:
            for k in (2, 4, 8, 16): vals[f"cross_{k}"] = np.zeros(3)
        zd = by_doy.get(doy, vis.iloc[0:0]); vals["doy"] = np.bincount(zd._src.to_numpy(int), minlength=3).astype(float) if len(zd) else np.zeros(3)
        # Crop-specific day-of-year schedule.
        zdc = zd[zd._crop == crop]; vals["doy_crop"] = np.bincount(zdc._src.to_numpy(int), minlength=3).astype(float) if len(zdc) else np.zeros(3)
        rec = {"true": int(true[len(records)]), ID: aid, DATE: dt}
        for n, c in vals.items(): rec[n] = int(np.argmax(c)) if c.sum() else -1; rec[n+"_n"] = int(c.sum())
        # A few weighted combinations; counts are normalized per evidence.
        def prob(n):
            c = vals[n]; return c / c.sum() if c.sum() else np.ones(3) / 3
        for name, specs in {
            "sp1_temp1": [("sp_crop_1", 2), ("temp_4", 1)],
            "sp2_temp2": [("sp_crop_2", 2), ("temp_4", 1), ("cross_4", .5)],
            "sp4_sched": [("sp_crop_4", 2), ("temp_4", 1), ("cross_8", 1), ("doy_crop", .5)],
            "sp1_cross": [("sp_crop_1", 2), ("cross_8", 1)],
            "sp1_doy": [("sp_crop_1", 2), ("doy", 1)],
        }.items():
            pp = np.zeros(3)
            for n, w in specs: pp += w * prob(n)
            rec[name] = int(np.argmax(pp))
        records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False); pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False); pr[GAP] = pr[GAP].fillna(False).astype(bool)
    allr = []
    for seed in (0, 1, 2, 70404):
        print("seed", seed, flush=True); q = classify(tr, pr, make_holdout(pr, seed)); q["seed"] = seed; allr.append(q)
    d = pd.concat(allr, ignore_index=True); methods = [c for c in d.columns if c not in {"true", ID, DATE, "seed"} and not c.endswith("_n")]
    rows = []
    for m in methods:
        for s, g in d.groupby("seed"):
            v = g[m].to_numpy(int); ok = v >= 0; rows.append({"method": m, "seed": int(s), "n": int(ok.sum()), "accuracy": float(np.mean(v[ok] == g.true.to_numpy(int)[ok])) if ok.any() else np.nan})
        v = d[m].to_numpy(int); ok = v >= 0; rows.append({"method": m, "seed": "pooled", "n": int(ok.sum()), "accuracy": float(np.mean(v[ok] == d.true.to_numpy(int)[ok])) if ok.any() else np.nan})
    out = pd.DataFrame(rows); out.to_csv(R / "source_schedule_route_probe.csv", index=False); d.to_csv(R / "source_schedule_route_probe_rows.csv", index=False)
    print(out[out.seed.eq("pooled")].sort_values("accuracy", ascending=False).head(30).to_string(index=False), flush=True)


if __name__ == "__main__": main()

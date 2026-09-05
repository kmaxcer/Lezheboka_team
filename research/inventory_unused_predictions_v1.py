"""Bounded inventory of existing prediction sidecars.

Aligns existing research prediction artifacts to the four leakage-safe
private-like masks and reports RMSE/error correlations against the current
source-route and local baselines.  This audit never writes submissions or
overwrites existing files.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
SEEDS = (0, 1, 2, 70404)
ID, DATE = "anon_polygon_id", "date"


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def corr_error(y: np.ndarray, p: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    e1, e2 = p[ok] - y[ok], b[ok] - y[ok]
    if np.std(e1) == 0 or np.std(e2) == 0:
        return np.nan
    return float(np.corrcoef(e1, e2)[0, 1])


def norm_key(d: pd.DataFrame) -> pd.Series:
    return d[ID].astype(str).str.strip() + "|" + pd.to_datetime(d[DATE], errors="coerce").dt.strftime("%Y-%m-%d")


def build_reference() -> pd.DataFrame:
    """One row per (mask seed,key), including observable baseline variants."""
    route = pd.read_csv(R / "source_expert_route_v2_fixed_radius_trainaug_rows.csv", low_memory=False)
    route[DATE] = pd.to_datetime(route[DATE])
    sched = pd.read_csv(R / "source_schedule_route_probe_rows.csv", usecols=[ID, DATE, "seed", "sp_crop_2_n", "sp_crop_8_n"], low_memory=False)
    sched[DATE] = pd.to_datetime(sched[DATE])
    q = route.merge(sched, on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    n2 = q.sp_crop_2_n.fillna(0).to_numpy(float)
    n8 = q.sp_crop_8_n.fillna(0).to_numpy(float)
    near, mid = n2 > 0, (n2 <= 0) & (n8 > 0)
    year, cohort = q.year.to_numpy(int), q.cohort.astype(str).to_numpy()
    a = np.where(near, .50, np.where(mid, .40, .30))
    a = np.where((cohort == "new") & (year == 2025), .60, a)
    a = np.where((cohort == "shared") & (year == 2025), .35, a)
    q["route_base"] = (1 - a) * q.baseline.to_numpy(float) + a * q.expert_trainaug_r2.to_numpy(float)
    pth = R / "paired_aoi_trainaug_local_audit_v1_rows.csv"
    if pth.exists():
        p = pd.read_csv(pth, usecols=[ID, DATE, "seed", "base_local", "n12_c40_r100_k2"], low_memory=False)
        p[DATE] = pd.to_datetime(p[DATE])
        q = q.merge(p, on=[ID, DATE, "seed"], how="left", validate="one_to_one")
        q["paired_best"] = q.base_local + .08 * (q["n12_c40_r100_k2"] - q.base_local)
    else:
        q["base_local"] = np.nan
        q["paired_best"] = np.nan
    q["key"] = norm_key(q)
    q["seed"] = q.seed.astype(int)
    q["truth"] = pd.to_numeric(q.truth, errors="coerce")
    return q[["key", ID, DATE, "seed", "truth", "route_base", "base_local", "paired_best"]]


def partition_seed(x: object) -> int | None:
    s = str(x).lower()
    m = re.search(r"(?:random[_-]?|mask[_-]?)(0|1|2|70404)$", s)
    if m:
        return int(m.group(1))
    return int(s) if s in {"0", "1", "2", "70404"} else None


def numeric_prediction_columns(d: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    ex = set(excluded)
    out: list[str] = []
    for c in d.columns:
        if c in ex or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        lc = c.lower()
        if any(t in lc for t in ["truth", "baseline", "shock", "state", "span", "count", "year", "doy", "seed", "index", "coverage", "n_"]):
            continue
        if lc == "pred" or lc.startswith(("pred", "p_", "hgb", "lag", "blend", "joint", "expert", "route", "interp", "crossyr", "global_doy", "date_peer", "factor_", "ridge_", "ext", "spectral", "base", "peer", "prod", "soft", "hard", "uniform", "oracle", "v3", "b0.")):
            out.append(c)
    return out

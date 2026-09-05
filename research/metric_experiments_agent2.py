"""Fast time-series imputation experiments for Agropuls.

This is deliberately separate from :mod:`src.infer`: it evaluates candidate
estimators on pseudo-private masks and writes a compact CSV/Markdown report.
The hidden rows have no dynamic features, so all methods use only observations
available in the masked frame plus source schedules learned from other AOIs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import (  # noqa: E402
    SOURCES,
    _fit_source_maps,
    _local_source_prediction,
    _mode_posteriors,
    _prepare,
    _query_posterior,
    predict_private,
)
from validate import make_fold  # noqa: E402


def _source(df: pd.DataFrame) -> np.ndarray:
    s2 = np.isfinite(df["s2_ndvi"].to_numpy(float))
    ls = np.isfinite(df["landsat_ndvi"].to_numpy(float))
    md = np.isfinite(df["modis_ndvi"].to_numpy(float))
    out = np.full(len(df), "none", dtype=object)
    out[s2] = "s2"
    out[~s2 & ls] = "landsat"
    out[~s2 & ~ls & md] = "modis"
    return out


def _linear_local(xq, kk, x, y, k=8, degree=1, robust=True):
    """Local polynomial in date, centered at xq; no sensor conversion."""
    if len(kk) == 0:
        return np.nan
    d = np.abs(x[kk] - xq)
    js = kk[np.argsort(d)[: min(k, len(kk))]]
    yy = y[js].astype(float)
    good = np.isfinite(yy)
    js, yy, d = js[good], yy[good], d[np.argsort(d)[: min(k, len(kk))]][good]
    if len(yy) == 0:
        return np.nan
    if len(yy) <= degree:
        return float(np.average(yy, weights=1.0 / (1.0 + d)))
    scale = max(1.0, float(np.max(d)))
    z = (x[js] - xq) / scale
    w = 1.0 / (1.0 + 2.0 * np.abs(z))
    try:
        coef = np.polynomial.polynomial.polyfit(z, yy, degree, w=w)
        val = float(coef[0])
    except Exception:
        val = float(np.average(yy, weights=w))
    if robust and len(yy) >= 4:
        lo, hi = np.quantile(yy, [0.05, 0.95])
        val = float(np.clip(val, lo - 0.04, hi + 0.04))
    return val


def _predict_mode(fold: pd.DataFrame, mode: str = "mode", k: int = 8,
                  date_weight: float = 1.0, bin_days: int = 30) -> pd.DataFrame:
    """Source-aware estimator with alternative source posterior decisions.

    ``weighted`` reproduces production; ``mode`` picks the most likely source;
    ``date_mode`` uses only the same-year/date acquisition schedule.
    """
    df = _prepare(fold)
    syn = fold["is_synthetic_gap"].astype(bool).to_numpy()
    known = np.isfinite(df.primary_ndvi.to_numpy(float))
    maps = _fit_source_maps(df, known, bin_days=bin_days)
    aoi, crop, glob, date = _mode_posteriors(df, known)
    y = df.primary_ndvi.to_numpy(float)
    x = df._ord.to_numpy(float)
    src = df._src.to_numpy(object)
    out = np.full(len(df), np.nan)
    groups = df.groupby(["anon_polygon_id", "_year"], sort=False).groups
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        kk = ii[known[ii]]
        for q in ii[syn[ii]]:
            if mode == "date_mode":
                p = date.get((int(df._year.iat[q]), int(df._doy.iat[q])))
                if p is None:
                    p = _query_posterior(df, int(q), aoi, crop, glob, date,
                                         date_weight=date_weight)
            else:
                p = _query_posterior(df, int(q), aoi, crop, glob, date,
                                     date_weight=date_weight)
            vals = []
            for s, w in zip(SOURCES, p):
                v = _local_source_prediction(
                    x[q], kk, x, y, src, s, maps,
                    query_doy=int(df._doy.iat[q]),
                    bin_days=bin_days, k=k,
                )
                if np.isfinite(v):
                    vals.append((v, float(w), s))
            if vals:
                if mode in ("mode", "date_mode"):
                    # Stable tie-break follows the task source priority.
                    best = max(vals, key=lambda z: (z[1], -SOURCES.index(z[2])))
                    out[q] = best[0]
                else:
                    out[q] = float(np.average([z[0] for z in vals], weights=[z[1] for z in vals]))
    # same fallback as production
    for q in np.flatnonzero(syn & ~np.isfinite(out)):
        same = np.flatnonzero(known & (df.anon_polygon_id.to_numpy() == df.anon_polygon_id.iat[q]))
        out[q] = y[same[np.argmin(np.abs(x[same] - x[q]))]] if len(same) else np.nanmedian(y[known])
    return pd.DataFrame({"pred": out[syn]})


def _predict_local(fold: pd.DataFrame, k=8, degree=1, cross_year=False):
    df = _prepare(fold)
    syn = fold["is_synthetic_gap"].astype(bool).to_numpy()
    y = df.primary_ndvi.to_numpy(float)
    x = df._ord.to_numpy(float)
    known = np.isfinite(y)
    out = np.full(len(df), np.nan)
    if cross_year:
        groups = df.groupby("anon_polygon_id", sort=False).groups
    else:
        groups = df.groupby(["anon_polygon_id", "_year"], sort=False).groups
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        kk = ii[known[ii]]
        for q in ii[syn[ii]]:
            out[q] = _linear_local(x[q], kk, x, y, k=k, degree=degree)
    for q in np.flatnonzero(syn & ~np.isfinite(out)):
        same = np.flatnonzero(known & (df.anon_polygon_id.to_numpy() == df.anon_polygon_id.iat[q]))
        out[q] = y[same[np.argmin(np.abs(x[same] - x[q]))]] if len(same) else np.nanmedian(y[known])
    return pd.DataFrame({"pred": out[syn]})


def _predict_sensor_mode(fold: pd.DataFrame, k=8, date_weight=1.0):
    """Infer the acquisition sensor from date schedule, then interpolate only it."""
    df = _prepare(fold)
    syn = fold["is_synthetic_gap"].astype(bool).to_numpy()
    y = df.primary_ndvi.to_numpy(float)
    x = df._ord.to_numpy(float)
    src = df._src.to_numpy(object)
    known = np.isfinite(y)
    maps = _fit_source_maps(df, known, bin_days=30)
    aoi, crop, glob, date = _mode_posteriors(df, known)
    out = np.full(len(df), np.nan)
    groups = df.groupby(["anon_polygon_id", "_year"], sort=False).groups
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        for q in ii[syn[ii]]:
            p = date.get((int(df._year.iat[q]), int(df._doy.iat[q])))
            if p is None:
                p = _query_posterior(df, int(q), aoi, crop, glob, date, date_weight=date_weight)
            s = SOURCES[int(np.argmax(p))]
            # Prefer neighbours that were actually measured by the selected sensor.
            kk = ii[known[ii] & (src[ii] == s)]
            v = _local_source_prediction(x[q], kk, x, y, src, s, maps,
                                         query_doy=int(df._doy.iat[q]),
                                         bin_days=30, k=k)
            if not np.isfinite(v):
                kk = ii[known[ii]]
                v = _local_source_prediction(x[q], kk, x, y, src, s, maps,
                                             query_doy=int(df._doy.iat[q]),
                                             bin_days=30, k=k)
            out[q] = v
    for q in np.flatnonzero(syn & ~np.isfinite(out)):
        same = np.flatnonzero(known & (df.anon_polygon_id.to_numpy() == df.anon_polygon_id.iat[q]))
        out[q] = y[same[np.argmin(np.abs(x[same] - x[q]))]] if len(same) else np.nanmedian(y[known])
    return pd.DataFrame({"pred": out[syn]})


def _predict_date_effect(fold: pd.DataFrame, weight: float = 1.0,
                         bin_days: int = 30, k: int = 8) -> pd.DataFrame:
    """Production prediction plus a robust same-date cross-AOI shock.

    Satellite observations on one acquisition date share cloud/atmosphere
    effects across polygons.  We estimate that effect in a canonical S2
    domain after removing each AOI's seasonal median, then add a shrunken
    exact-date residual to the local prediction.  This is an experiment only;
    the query row itself is excluded because it is masked.
    """
    df = _prepare(fold)
    syn = fold["is_synthetic_gap"].astype(bool).to_numpy()
    y = df.primary_ndvi.to_numpy(float)
    known = np.isfinite(y)
    src = df._src.to_numpy(object)
    maps = _fit_source_maps(df, known, bin_days=bin_days)
    # Convert each observed primary to a canonical S2-like domain.
    z = np.full(len(df), np.nan)
    for i in np.flatnonzero(known):
        s = str(src[i])
        a, b = maps.get(("s2", s, int(df._doy.iat[i] // bin_days)),
                        maps.get(("s2", s, "g"), (0.0, 1.0)))
        z[i] = a + b * y[i]
    # AOI seasonal baselines.  Use a broad bin to keep enough multi-year rows.
    tmp = pd.DataFrame({"pid": df.anon_polygon_id.to_numpy(),
                        "db": (df._doy.to_numpy(int) // max(1, bin_days)),
                        "year": df._year.to_numpy(int), "doy": df._doy.to_numpy(int),
                        "z": z})
    base = tmp.loc[known].groupby(["pid", "db"], observed=True).z.median()
    key = pd.MultiIndex.from_arrays([tmp.pid, tmp.db])
    bvals = base.reindex(key).to_numpy(float)
    # Global seasonal fallback, then robust date shock by exact year/DOY.
    glob = tmp.loc[known].groupby("db", observed=True).z.median()
    missing = ~np.isfinite(bvals)
    if missing.any():
        bvals[missing] = tmp.loc[missing, "db"].map(glob).to_numpy(float)
    resid = z - bvals
    tmp["resid"] = resid
    shock = tmp.loc[known].groupby(["year", "doy"], observed=True).resid.median()
    # Also keep an uncertainty/shrinkage count per date.
    cnt = tmp.loc[known].groupby(["year", "doy"], observed=True).resid.count()
    # Base production prediction.
    prod = predict_private(fold, k=k, bin_days=bin_days).primary_ndvi_pred.to_numpy(float)
    qidx = np.flatnonzero(syn)
    add = np.zeros(len(qidx), dtype=float)
    for n, q in enumerate(qidx):
        sh = shock.get((int(df._year.iat[q]), int(df._doy.iat[q])), np.nan)
        if np.isfinite(sh):
            # More peers justify a stronger correction; cap to avoid unstable
            # tails.  ``weight`` is the user-tunable global multiplier.
            npeer = float(cnt.get((int(df._year.iat[q]), int(df._doy.iat[q])), 0))
            shrink = min(1.0, npeer / 8.0)
            add[n] = float(np.clip(weight * shrink * sh, -0.20, 0.20))
    return pd.DataFrame({"pred": np.clip(prod + add, -0.5, 1.2)})


def evaluate(train: pd.DataFrame, private: pd.DataFrame, years: Iterable[int],
             methods: dict[str, callable]) -> pd.DataFrame:
    rows = []
    for year in years:
        fold, truth = make_fold(train, private, int(year))
        if len(truth) == 0:
            continue
        yy = truth.to_numpy(float)
        for name, fn in methods.items():
            pred = fn(fold).pred.to_numpy(float)
            e = pred - yy
            rows.append({"year": int(year), "method": name, "n": len(yy),
                         "rmse": float(np.sqrt(np.mean(e * e))),
                         "mae": float(np.mean(np.abs(e)))})
    out = pd.DataFrame(rows)
    if len(out):
        allrows = []
        for name, g in out.groupby("method"):
            allrows.append({"year": "all", "method": name, "n": int(g.n.sum()),
                            "rmse": float(np.sqrt(np.average(g.rmse**2, weights=g.n))),
                            "mae": float(np.average(g.mae, weights=g.n))})
        out = pd.concat([out, pd.DataFrame(allrows)], ignore_index=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    train = pd.read_csv(args.train, parse_dates=["date"])
    private = pd.read_csv(args.private, parse_dates=["date"])
    methods = {
        "production_weighted": lambda f: pd.DataFrame({"pred": predict_private(f).primary_ndvi_pred}),
        "source_mode": lambda f: _predict_mode(f, mode="mode"),
        "date_mode": lambda f: _predict_mode(f, mode="date_mode"),
        "sensor_schedule": _predict_sensor_mode,
        "local_linear_k4": lambda f: _predict_local(f, k=4, degree=1),
        "local_linear_k8": lambda f: _predict_local(f, k=8, degree=1),
        "local_poly2_k8": lambda f: _predict_local(f, k=8, degree=2),
        "local_poly2_k12": lambda f: _predict_local(f, k=12, degree=2),
        "crossyear_linear": lambda f: _predict_local(f, k=8, degree=1, cross_year=True),
    }
    res = evaluate(train, private, [2019, 2020, 2021, 2022, 2023, 2024], methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)
    print(res.sort_values(["year", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()

"""Evaluate direct per-sensor interpolation versus primary-source bridge.

This is an experiment only; it does not modify production inference.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import SOURCES, SENSOR_COL, _mode_posteriors, _prepare, _query_posterior
from validate import make_fold


def local_poly(xq, xx, yy, k=16, degree=3):
    good = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[good], yy[good]
    if len(yy) == 0:
        return np.nan
    d = np.abs(xx - xq)
    ii = np.argsort(d)[:min(k, len(d))]
    xx, yy, d = xx[ii], yy[ii], d[ii]
    if len(yy) <= degree:
        return float(yy[np.argmin(d)])
    z = (xx - xq) / max(1.0, float(np.max(d)))
    w = 1.0 / (1.0 + 2.0 * np.abs(z))
    try:
        val = float(np.polynomial.polynomial.polyfit(z, yy, min(degree, len(yy)-1), w=w)[0])
    except Exception:
        val = float(np.average(yy, weights=w))
    lo, hi = np.quantile(yy, [0.05, 0.95])
    return float(np.clip(val, lo - .04, hi + .04))


def predict_direct(frame, k=16, degree=3):
    df = _prepare(frame)
    syn = frame["is_synthetic_gap"].astype(bool).to_numpy()
    x = df["_ord"].to_numpy(float)
    # Source schedule from all available observed rows (including other AOIs).
    known = np.isfinite(df.primary_ndvi.to_numpy(float))
    aoi, crop, glob, date = _mode_posteriors(df, known)
    pred = np.full(len(df), np.nan)
    groups = df.groupby(["anon_polygon_id", "_year"], sort=False).groups
    sensor = {s: df[SENSOR_COL[s]].to_numpy(float) for s in SOURCES}
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        for q in ii[syn[ii]]:
            p = _query_posterior(df, int(q), aoi, crop, glob, date)
            vals = []
            for s, w in zip(SOURCES, p):
                kk = ii[np.isfinite(sensor[s][ii]) & known[ii]]
                v = local_poly(x[q], x[kk], sensor[s][kk], k=k, degree=degree)
                if np.isfinite(v):
                    vals.append((v, float(w)))
            if vals:
                pred[q] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    # same-AOI nearest fallback
    ids = df.anon_polygon_id.to_numpy()
    for q in np.flatnonzero(syn & ~np.isfinite(pred)):
        same = np.flatnonzero(known & (ids == ids[q]))
        pred[q] = df.primary_ndvi.to_numpy(float)[same[np.argmin(np.abs(x[same]-x[q]))]] if len(same) else np.nanmedian(df.primary_ndvi.to_numpy(float)[known])
    return pred[syn]


def main():
    root = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
    tr = pd.read_csv(root / "train_dataset.csv", parse_dates=["date"])
    pr = pd.read_csv(root / "private_features.csv", parse_dates=["date"])
    rows = []
    for k in [8, 12, 16, 24]:
      for deg in [1, 2, 3]:
        es=[]; n=0
        for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
            fold, truth = make_fold(tr, pr, yr)
            yh = predict_direct(fold, k=k, degree=deg)
            e = yh - truth.to_numpy(float); es.extend(e.tolist()); n += len(e)
        rows.append((k,deg,n,float(np.sqrt(np.mean(np.asarray(es)**2))),float(np.mean(np.abs(es)))))
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows,columns=['k','degree','n','rmse','mae']).sort_values('rmse').to_string(index=False))


if __name__ == "__main__": main()

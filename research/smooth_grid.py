"""Fast grid of robust temporal smoothers for the NDVI gap task.

This is research-only.  It evaluates source-normalized kernel/local-polynomial
curves on the same hidden-date folds as the real private file and on random
private-like masks, then blends the strongest candidates with saved HGB/lag
predictions when available.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
from infer import (  # noqa: E402
    SOURCES, _fit_source_maps, _mode_posteriors, _prepare, _query_posterior,
)
from validate import make_fold  # noqa: E402


def source_arrays(d: pd.DataFrame):
    z = _prepare(d)
    y = z.primary_ndvi.to_numpy(float)
    known = np.isfinite(y)
    src = z._src.to_numpy(object)
    maps = _fit_source_maps(z, known, bin_days=30)
    aoi, crop, glob, date = _mode_posteriors(z, known)
    return z, y, known, src, maps, (aoi, crop, glob, date)


def _map_to_can(v: np.ndarray, s: str, maps, doy: int) -> np.ndarray:
    """Convert source-domain values to an S2-like canonical domain."""
    b = int(doy // 30)
    a, k = maps.get(("s2", s, b), maps.get(("s2", s, "g"), (0.0, 1.0)))
    return float(a) + float(k) * v


def _map_from_can(v: float, target: str, maps, doy: int) -> float:
    b = int(doy // 30)
    a, k = maps.get(("s2", target, b), maps.get(("s2", target, "g"), (0.0, 1.0)))
    if abs(float(k)) < 1e-8:
        return float(v)
    return (float(v) - float(a)) / float(k)


def _robust_kernel(xq, xx, yy, bandwidth, kind="gauss", degree=0):
    if len(xx) == 0:
        return np.nan
    d = np.abs(xx - float(xq))
    h = max(0.5, float(bandwidth))
    u = d / h
    if kind == "tricube":
        w = np.where(u < 1, (1 - u**3) ** 3, 0.0)
    elif kind == "epan":
        w = np.where(u < 1, 1 - u**2, 0.0)
    else:
        w = np.exp(-0.5 * u * u)
    good = np.isfinite(yy) & np.isfinite(w) & (w > 1e-10)
    if good.sum() == 0:
        # nearest-neighbour fallback
        return float(yy[np.nanargmin(d)]) if np.isfinite(yy).any() else np.nan
    xx = xx[good]; yy = yy[good]; w = w[good]
    if len(yy) == 1 or degree <= 0:
        # A weighted median is useful against isolated cloud outliers at very
        # short bandwidths; use weighted mean for the smooth Gaussian mode.
        if kind == "median":
            order = np.argsort(yy); cs = np.cumsum(w[order]); return float(yy[order[np.searchsorted(cs, cs[-1] / 2)]])
        return float(np.sum(w * yy) / np.sum(w))
    scale = max(1.0, h)
    z = (xx - float(xq)) / scale
    deg = min(int(degree), len(yy) - 1)
    try:
        coef = np.polynomial.polynomial.polyfit(z, yy, deg, w=np.sqrt(w))
        val = float(coef[0])
    except Exception:
        val = float(np.sum(w * yy) / np.sum(w))
    # conservative robust range guard
    lo, hi = np.quantile(yy, [0.03, 0.97]) if len(yy) >= 5 else (np.min(yy), np.max(yy))
    return float(np.clip(val, lo - 0.05, hi + 0.05))


def predict_smoother(frame: pd.DataFrame, *, bandwidth=8.0, kind="gauss",
                     degree=0, cross_year=False, canonical=True,
                     source_mode="posterior") -> pd.DataFrame:
    """Predict synthetic rows; only visible target values are consumed."""
    z, y, known, src, maps, posts = source_arrays(frame)
    syn = frame.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    x = z._ord.to_numpy(float)
    doy = z._doy.to_numpy(int)
    ids = z.anon_polygon_id.to_numpy(object)
    years = z._year.to_numpy(int)
    # canonical observations, preserving seasonal source conversion
    can = np.full(len(z), np.nan)
    for i in np.flatnonzero(known):
        can[i] = _map_to_can(np.array([y[i]]), str(src[i]), maps, int(doy[i]))[0]
    out = np.full(len(z), np.nan)
    groups = z.groupby("anon_polygon_id" if cross_year else ["anon_polygon_id", "_year"], sort=False).groups
    aoi, crop, glob, date = posts
    for _, ii0 in groups.items():
        ii = np.asarray(ii0, dtype=int)
        kk = ii[known[ii] & np.isfinite(can[ii])]
        if len(kk) == 0:
            continue
        for q in ii[syn[ii]]:
            if cross_year:
                # Align all years by DOY; use a modest effective day window.
                xx = doy[kk].astype(float)
                xq = float(doy[q])
            else:
                xx = x[kk]; xq = x[q]
            # For a calendar-aligned cross-year curve, scale bandwidth in days.
            val = _robust_kernel(xq, xx, can[kk], bandwidth, kind=kind, degree=degree)
            if not np.isfinite(val):
                continue
            if source_mode == "oracle":
                # diagnostics only; production does not know this label
                target = str(frame.get("_true_src", pd.Series("s2", index=frame.index)).iat[q])
                if target not in SOURCES: target = "s2"
                out[q] = _map_from_can(val, target, maps, int(doy[q]))
            else:
                p = _query_posterior(z, int(q), aoi, crop, glob, date, date_weight=1.0)
                vals = [_map_from_can(val, s, maps, int(doy[q])) for s in SOURCES]
                if source_mode == "hard":
                    out[q] = vals[int(np.argmax(p))]
                else:
                    out[q] = float(np.average(vals, weights=p))
    # nearest visible fallback
    for q in np.flatnonzero(syn & ~np.isfinite(out)):
        same = np.flatnonzero(known & (ids == ids[q]))
        out[q] = y[same[np.argmin(np.abs(x[same] - x[q]))]] if len(same) else np.nanmedian(y[known])
    return pd.DataFrame({"pred": out[syn]})


def random_private_mask(pr: pd.DataFrame, seed: int, frac=.15, year=None):
    d = pr.copy().reset_index(drop=True)
    if year is not None:
        d = d[d.date.dt.year.eq(year)].copy().reset_index(drop=True)
    d["_truth"] = d.primary_ndvi.astype(float)
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(d), bool)
    pool = d.primary_ndvi.notna()
    for _, g in d.loc[pool].groupby("anon_polygon_id"):
        ix = g.index.to_numpy(); n = max(1, int(round(frac * len(ix))))
        mask[rng.choice(ix, n, replace=False)] = True
    dynamic = ["s2_ndvi","s2_evi","s2_ndwi","landsat_ndvi","landsat_evi","landsat_ndwi","modis_ndvi","modis_evi","era5_temp_c","era5_precip_mm","year","primary_ndvi","doy","ndvi_climatology_mean","ndvi_climatology_std","ndvi_zscore","n_reference_years","status"]
    for c in dynamic:
        if c in d: d.loc[mask, c] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    return d, mask


def score(p, y):
    e = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def main():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    records=[]; predrows=[]
    # exact hidden-date train folds are the strongest structural proxy
    for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
        f, truth = make_fold(tr, pr, yr)
        if len(truth)==0: continue
        qmask=f.is_synthetic_gap.to_numpy(bool); ytrue=truth.to_numpy(float)
        for kind in ["gauss","tricube","epan","median"]:
            for bw in [2,3,4,5,6,8,10,12,16,20,30]:
                for deg in ([0,1,2] if kind not in ("median",) else [0]):
                    p=predict_smoother(f,bandwidth=bw,kind=kind,degree=deg,cross_year=False).pred.to_numpy(float)
                    rm,ma=score(p,ytrue); records.append(dict(protocol="exact",year=yr,kind=kind,bw=bw,deg=deg,rmse=rm,mae=ma,n=len(ytrue)))
                    if (kind,bw,deg) in [("gauss",8,0),("gauss",5,1),("tricube",8,1),("median",8,0)]:
                        predrows.append(pd.DataFrame({"protocol":"exact","year":yr,"idx":np.flatnonzero(qmask),"truth":ytrue,"method":f"{kind}_{bw}_{deg}","pred":p}))
        print('exact year',yr,'done',flush=True)
    # private-like random masks, all rows and 2025 stress
    for seed in [0,1,2]:
        for yy in [None,2025]:
            f,mask=random_private_mask(pr,seed,frac=.15,year=yy)
            truth=f.loc[mask,"_truth"].to_numpy(float)
            for kind,bw,deg in [("gauss",3,0),("gauss",5,0),("gauss",8,0),("gauss",12,0),("gauss",8,1),("tricube",8,1),("median",8,0),("gauss",16,1)]:
                p=predict_smoother(f,bandwidth=bw,kind=kind,degree=deg,cross_year=False).pred.to_numpy(float); rm,ma=score(p,truth)
                records.append(dict(protocol="random2025" if yy else "random",year=yy or 0,seed=seed,kind=kind,bw=bw,deg=deg,rmse=rm,mae=ma,n=len(truth)))
            print('random',seed,yy,'done',flush=True)
    out=pd.DataFrame(records); out.to_csv(ROOT/'research'/'smooth_grid_results.csv',index=False)
    print(out.sort_values('rmse').head(40).to_string(index=False))
    if predrows: pd.concat(predrows,ignore_index=True).to_csv(ROOT/'research'/'smooth_grid_preds.csv',index=False)

if __name__=='__main__': main()

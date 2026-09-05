"""Evaluate a multi-sensor local smoother on realistic hidden-date folds.

The target is the priority sensor value, but secondary sensors provide useful
nearby observations.  This experiment maps each raw sensor into a hypothetical
target-sensor domain using overlap calibration, then fits a local polynomial
around each query.  It is deliberately isolated from production code.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold

SENS = ("s2", "landsat", "modis")
COL = {"s2": "s2_ndvi", "landsat": "landsat_ndvi", "modis": "modis_ndvi"}
DYN = ["s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
       "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
       "era5_precip_mm", "year", "primary_ndvi", "doy",
       "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
       "n_reference_years", "status"]


def _trim(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 2) & (np.abs(y) < 2)
    x, y = x[ok], y[ok]
    if len(x) < 20 or np.ptp(x) < 1e-8:
        return 0.0, 1.0
    qx = np.quantile(x, [0.02, .98]); qy = np.quantile(y, [0.02, .98])
    ok = (x >= qx[0]) & (x <= qx[1]) & (y >= qy[0]) & (y <= qy[1])
    x, y = x[ok], y[ok]
    if len(x) < 12 or np.ptp(x) < 1e-8:
        return float(np.median(y) - np.median(x)), 1.0
    b, a = np.polyfit(x, y, 1)
    return (float(a), float(b)) if np.isfinite(a+b) and abs(b) < 3 else (0.0, 1.0)


def fit_maps(d: pd.DataFrame, known: np.ndarray, bin_days: int = 30):
    """Map raw source -> hypothetical target source, globally/seasonally."""
    raw = {s: d[COL[s]].to_numpy(float) for s in SENS}
    doy = d.date.dt.dayofyear.to_numpy(int)
    out = {}
    for target in SENS:
        for source in SENS:
            if target == source:
                out[(target, source, "g")] = (0., 1.)
                continue
            both = np.isfinite(raw[target]) & np.isfinite(raw[source])
            # Do not require primary target here: overlap is more plentiful.
            out[(target, source, "g")] = _trim(raw[source][both], raw[target][both])
            for b in range(14):
                z = both & ((doy // bin_days) == b)
                if z.sum() >= 30:
                    out[(target, source, b)] = _trim(raw[source][z], raw[target][z])
    return out


def _map_val(v, target, source, qdoy, maps, bin_days):
    a, b = maps.get((target, source, int(qdoy // bin_days)),
                    maps.get((target, source, "g"), (0., 1.)))
    return a + b * v


def predict_fold(frame: pd.DataFrame, method: str = "all", k: int = 12,
                 degree: int = 1, bandwidth: float = 12.,
                 source_decay: float = 1., cross_year: bool = False) -> np.ndarray:
    d = frame.copy().reset_index(drop=True)
    d.date = pd.to_datetime(d.date)
    qmask = d.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    y = d.primary_ndvi.to_numpy(float)
    known = np.isfinite(y)
    maps = fit_maps(d, known)
    # Query source posterior from visible schedule.  For evaluation the true
    # source is intentionally not used.
    src_obs = np.select([d.s2_ndvi.notna(), d.landsat_ndvi.notna(), d.modis_ndvi.notna()],
                        ["s2", "landsat", "modis"], default="none")
    pred = np.full(len(d), np.nan)
    d['_yr'] = d.date.dt.year
    d['_ord'] = d.date.map(pd.Timestamp.toordinal).to_numpy(float)
    d['_doy'] = d.date.dt.dayofyear
    # source posterior by AOI+DOY, then global DOY; all based on rows whose
    # target remains visible in this fold.
    z = pd.DataFrame({'pid': d.anon_polygon_id, 'doy': d['_doy'], 'src': src_obs,
                      'known': known})
    z = z[z.known & z.src.isin(SENS)]
    tab = z.groupby(['pid', 'doy', 'src']).size().unstack('src', fill_value=0)
    for s in SENS:
        if s not in tab: tab[s] = 0
    tab = (tab.loc[:, list(SENS)].to_numpy(float) + .5)
    tab /= tab.sum(axis=1, keepdims=True)
    post = {key: val for key, val in zip(z.groupby(['pid', 'doy']).size().index, tab)}
    gtab = z.groupby(['doy', 'src']).size().unstack('src', fill_value=0)
    for s in SENS:
        if s not in gtab: gtab[s] = 0
    gidx = gtab.index
    gtab = (gtab.loc[:, list(SENS)].to_numpy(float) + .5); gtab /= gtab.sum(axis=1, keepdims=True)
    gpost = {key: val for key, val in zip(gidx, gtab)}

    for (pid, yr), gi in d.groupby(['anon_polygon_id', '_yr'], sort=False).groups.items():
        ii = np.asarray(list(gi), dtype=int)
        qq = ii[qmask[ii]]
        # observations from current AOI/year, or optionally all years at same
        # seasonal phase when a sparse one-year AOI has too few anchors.
        aa = ii[known[ii]]
        if cross_year and len(aa) < 8:
            aa = np.flatnonzero(known & (d.anon_polygon_id.to_numpy() == pid))
        if not len(aa):
            continue
        # Build raw sensor anchor table.  Primary-only is retained as a useful
        # control; all-mode uses every finite raw NDVI value.
        anchors = []
        for j in aa:
            for s in SENS:
                v = d.at[j, COL[s]]
                if np.isfinite(v) and (method == 'all' or src_obs[j] == s):
                    anchors.append((float(d.at[j, '_ord']), s, float(v), int(d.at[j, '_doy'])))
        if not anchors:
            continue
        ax = np.array([a[0] for a in anchors]); ass = np.array([a[1] for a in anchors], object)
        av = np.array([a[2] for a in anchors]); ad = np.array([a[3] for a in anchors])
        for q in qq:
            qx = float(d.at[q, '_ord']); qd = int(d.at[q, '_doy'])
            # Candidate target source domains, then posterior average.
            pp = post.get((pid, qd), gpost.get((qd,), np.array([.4, .4, .2])))
            vals = []
            for target, pw in zip(SENS, pp):
                vv = np.array([_map_val(v, target, s, qd, maps, 30)
                               for s, v in zip(ass, av)])
                dist = np.abs(ax - qx)
                # Keep nearby anchors; source observations at the same date
                # are especially informative, but avoid overwhelming the fit.
                order = np.argsort(dist)
                take = order[:min(int(k), len(order))]
                if source_decay != 1.:
                    sw = np.where(ass[take] == target, 1., float(source_decay))
                else: sw = np.ones(len(take))
                dd = dist[take]
                w = np.exp(-dd / max(.5, float(bandwidth))) * sw
                zz = (ax[take] - qx) / max(1., float(np.max(dd)) if len(dd) else 1.)
                good = np.isfinite(vv[take]) & np.isfinite(w)
                if good.sum() == 0: continue
                try:
                    coef = np.polynomial.polynomial.polyfit(zz[good], vv[take][good],
                                                            min(int(degree), good.sum()-1), w=w[good])
                    val = float(coef[0])
                except Exception:
                    val = float(np.average(vv[take][good], weights=w[good]))
                # robust guard against a single sensor bridge explosion
                lo, hi = np.quantile(vv[take][good], [.03, .97])
                vals.append((float(np.clip(val, lo-.05, hi+.05)), float(pw)))
            if vals:
                pred[q] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    # broad fallback
    for q in np.flatnonzero(qmask & ~np.isfinite(pred)):
        same = np.flatnonzero(known & (d.anon_polygon_id.to_numpy() == d.anon_polygon_id.iat[q]))
        pred[q] = y[same[np.argmin(abs(d.loc[same, '_ord'].to_numpy()-d.at[q, '_ord']))]] if len(same) else np.nanmedian(y[known])
    return pred[qmask]


def mask_private(pr: pd.DataFrame, seed: int, frac=.15, year=None):
    d = pr.copy().reset_index(drop=True); d.date = pd.to_datetime(d.date)
    if year is not None: d = d[d.date.dt.year.eq(year)].copy().reset_index(drop=True)
    d['_truth'] = d.primary_ndvi.astype(float); d.is_synthetic_gap = False
    rng = np.random.default_rng(seed); mask = np.zeros(len(d), bool)
    for _, g in d[d.primary_ndvi.notna()].groupby(['anon_polygon_id', d.date.dt.year]):
        ix = g.index.to_numpy(); n=max(1, int(round(frac*len(ix)))); mask[rng.choice(ix,size=min(n,len(ix)),replace=False)] = True
    for c in DYN:
        if c in d: d.loc[mask,c] = np.nan
    d.loc[mask,'is_synthetic_gap'] = True
    return d, mask


def main():
    tr = pd.read_csv(DATA/'train_dataset.csv', parse_dates=['date'], low_memory=False)
    pr = pd.read_csv(DATA/'private_features.csv', parse_dates=['date'], low_memory=False)
    rows=[]
    configs=[]
    for method in ['primary','all']:
        for k in [4,6,8,12,16,24,32]:
            for deg in [0,1,2,3]:
                for bw in [2,4,8,12,20,30]:
                    if method=='primary' and (k not in [8,16] or deg not in [0,1,2]): continue
                    configs.append((method,k,deg,bw,1.0,False))
    # A small cross-year variant for sparse one-year groups.
    configs += [('all',8,1,8,1.,True),('all',12,1,12,1.,True),('all',16,2,12,1.,True)]
    for yr in [2019,2020,2021,2022,2023,2024]:
        f,t=make_fold(tr,pr,yr); truth=t.to_numpy(float)
        for cfg in configs:
            p=predict_fold(f,*cfg); e=p-truth
            rows.append(dict(protocol='exact',year=yr,method=cfg[0],k=cfg[1],degree=cfg[2],bw=cfg[3],decay=cfg[4],cross=cfg[5],rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
        print('exact',yr,flush=True)
    for seed in [0,1,2]:
        for yr in [None,2025]:
            f,m=mask_private(pr,seed,year=yr); truth=f.loc[m,'_truth'].to_numpy(float)
            for cfg in configs:
                p=predict_fold(f,*cfg);e=p-truth
                rows.append(dict(protocol='random2025' if yr else 'random',year=yr or 0,seed=seed,method=cfg[0],k=cfg[1],degree=cfg[2],bw=cfg[3],decay=cfg[4],cross=cfg[5],rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
            print('random',seed,yr,flush=True)
    out=pd.DataFrame(rows);out.to_csv(ROOT/'research/multisensor_results.csv',index=False)
    agg=[]
    for key,g in out.groupby(['protocol','method','k','degree','bw','decay','cross']):
        agg.append((*key,np.sqrt(np.average(g.rmse**2,weights=g.n)),np.average(g.mae,weights=g.n),g.n.sum()))
    a=pd.DataFrame(agg,columns=['protocol','method','k','degree','bw','decay','cross','rmse','mae','n'])
    a.to_csv(ROOT/'research/multisensor_aggregate.csv',index=False)
    print(a.sort_values(['protocol','rmse']).groupby('protocol').head(30).to_string(index=False))


if __name__=='__main__': main()

"""Research screening of source-conditioned temporal curves.

The target is exactly the first available sensor.  This script tests whether
separate sensor curves plus an inferred acquisition-source posterior beat the
generic HGB/lag model on the exact private-like hidden-DOY folds.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402

SENS = ["s2_ndvi", "landsat_ndvi", "modis_ndvi"]


def _src(d: pd.DataFrame) -> np.ndarray:
    return np.select([d[SENS[0]].notna(), d[SENS[1]].notna(), d[SENS[2]].notna()], [0, 1, 2], -1)


def _interp(x: np.ndarray, y: np.ndarray, q: np.ndarray, method: str = "linear") -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.full(len(q), np.nan)
    order = np.argsort(x); x, y = x[order], y[order]
    # Collapse duplicate abscissae using a median.
    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) != len(x):
        y = pd.Series(y).groupby(inv).median().to_numpy(); x = ux
    if len(x) == 1:
        return np.full(len(q), y[0])
    z = np.interp(q, x, y)
    if method == "nearest":
        pos = np.searchsorted(x, q, side="left").clip(0, len(x)-1)
        left = np.maximum(pos-1, 0); right = pos
        take = np.where(np.abs(q-x[left]) <= np.abs(x[right]-q), left, right)
        z = y[take]
    return z


def _harmonic(x: np.ndarray, y: np.ndarray, q: np.ndarray, period: float = 366., k: int = 4, ridge: float = 1e-2) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.full(len(q), np.nan)
    def design(t):
        cols=[np.ones(len(t)), t/366.]
        for j in range(1,k+1):
            cols += [np.sin(2*np.pi*j*t/period), np.cos(2*np.pi*j*t/period)]
        return np.column_stack(cols)
    X=design(x); Q=design(q)
    # Penalize all but intercept/linear terms.
    pen=np.eye(X.shape[1])*ridge; pen[:2,:2]=0
    try: b=np.linalg.solve(X.T@X+pen,X.T@y)
    except np.linalg.LinAlgError: b=np.linalg.lstsq(X,y,rcond=None)[0]
    return Q@b


def predict(frame: pd.DataFrame, query: np.ndarray, variant: str) -> np.ndarray:
    d=frame.copy().reset_index(drop=True); d.date=pd.to_datetime(d.date)
    doy=d.date.dt.dayofyear.to_numpy(float); year=d.date.dt.year.to_numpy(int)
    ids=d.anon_polygon_id.astype(str).to_numpy(); src=_src(d)
    known=d.primary_ndvi.notna().to_numpy(bool)
    # source schedule posterior from visible source rows; combine AOI+doy,
    # AOI+month-bin and global DOY with progressively stronger shrinkage.
    tab=pd.DataFrame({'id':ids,'doy':doy.astype(int),'src':src,'known':known})
    tab=tab[tab.known & (tab.src>=0)]
    g_id=tab.groupby(['id','doy','src']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    g_bin=tab.assign(bin=(tab.doy//8).astype(int)).groupby(['id','bin','src']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    g_doy=tab.groupby(['doy','src']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    g_global=tab.groupby('src').size().reindex([0,1,2],fill_value=0)
    def probs(i):
        key=(ids[i],int(doy[i]));
        if variant.startswith('mode_doy') or variant.startswith('soft_doy'):
            if key in g_id.index: c=g_id.loc[key].to_numpy(float)
            else: c=np.zeros(3)
            # shrink counts toward same-AOI bin and global DOY
            bk=(ids[i],int(doy[i]//8)); b=g_bin.loc[bk].to_numpy(float) if bk in g_bin.index else np.zeros(3)
            dd=g_doy.loc[int(doy[i])].to_numpy(float) if int(doy[i]) in g_doy.index else np.zeros(3)
            c = c + (0.5*b/ max(1,b.sum())) + 0.25*(dd/max(1,dd.sum()))
        elif variant.startswith('global_doy'):
            c=g_doy.loc[int(doy[i])].to_numpy(float) if int(doy[i]) in g_doy.index else g_global.to_numpy(float)
        else:
            c=g_global.to_numpy(float)
        c=c+0.5
        return c/c.sum()
    pred=np.full(len(d),np.nan)
    # Curves per AOI/year/source, with fallback to AOI all-year source curves.
    for (aid,yr), ix0 in d.groupby([ids,year], sort=False).groups.items():
        ix=np.asarray(ix0,dtype=int); kk=ix[known[ix]]; qq=ix[query[ix]]
        if len(qq)==0: continue
        # Build source curves from current AOI/year visible observations.
        curves=[]
        for s,col in enumerate(SENS):
            sel=kk[src[kk]==s]; x=doy[sel]; y=d.loc[sel,'primary_ndvi'].to_numpy(float)
            curves.append((x,y))
        # fallback across all visible years for this AOI
        allix=np.where(known & (ids==aid))[0]
        for s in range(3):
            if len(curves[s][0])<3:
                sel=allix[src[allix]==s]; curves[s]= (doy[sel],d.loc[sel,'primary_ndvi'].to_numpy(float))
        for i in qq:
            pp=probs(i); vals=[]
            for s in range(3):
                x,y=curves[s]
                if variant.endswith('harm2'): z=_harmonic(x,y,np.array([doy[i]]),k=2,ridge=.1)[0]
                elif variant.endswith('harm4'): z=_harmonic(x,y,np.array([doy[i]]),k=4,ridge=.1)[0]
                elif variant.endswith('nearest'): z=_interp(x,y,np.array([doy[i]]),'nearest')[0]
                else: z=_interp(x,y,np.array([doy[i]]),'linear')[0]
                vals.append(z)
            vals=np.asarray(vals,float); good=np.isfinite(vals)
            if not good.any(): continue
            pp=np.where(good,pp,0); pp/=pp.sum()
            if variant.startswith('mode_'):
                pred[i]=vals[int(np.argmax(pp))]
            else: pred[i]=np.sum(pp*vals)
    return pred[query]


def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False)
    pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    variants=['soft_doy_linear','mode_doy_linear','soft_doy_harm2','soft_doy_harm4','soft_doy_nearest','global_doy_linear','soft_global_linear']
    rows=[]
    for yr in [2019,2020,2021,2022,2023,2024]:
        fold,truth=make_fold(tr.copy(),pr.copy(),yr); q=fold.is_synthetic_gap.fillna(False).to_numpy(bool)
        y=truth.to_numpy(float)
        for v in variants:
            p=predict(fold,q,v);e=p-y
            rows.append({'year':yr,'variant':v,'n':len(y),'rmse':float(np.sqrt(np.nanmean(e*e))),'mae':float(np.nanmean(abs(e)))})
        print('done',yr,flush=True)
    out=pd.DataFrame(rows);out.to_csv(ROOT/'research/sensor_curve_results.csv',index=False)
    def _agg(g):
        return pd.Series({'n': g['n'].sum(), 'rmse': np.sqrt(np.average(g['rmse']**2, weights=g['n'])), 'mae': np.average(g['mae'], weights=g['n'])})
    agg=out.groupby('variant',as_index=False).apply(_agg,include_groups=False).reset_index(drop=True).sort_values('rmse');agg.to_csv(ROOT/'research/sensor_curve_aggregate.csv',index=False);print(agg.to_string(index=False))

if __name__=='__main__': main()

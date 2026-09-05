"""Pseudo-CV for climatology-normalised (z-score) interpolation.

The supplied rows satisfy ``primary_ndvi = climatology_mean + zscore*std``.
The z-score sequence is substantially smoother than raw NDVI, so this script
tests reconstructing the masked value in that domain.  It never reads the
masked row's climatology fields; all query baselines are interpolated from
unmasked rows and, optionally, historical years.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private
from validate import make_fold

def _local(xq,x,y,k=8,degree=1,weights=True):
    good=np.isfinite(x)&np.isfinite(y)
    x=x[good];y=y[good]
    if not len(x):return np.nan
    ix=np.argsort(abs(x-xq))[:min(k,len(x))];xx=x[ix];yy=y[ix]
    if len(yy)==1:return float(yy[0])
    scale=max(1.,float(np.max(abs(xx-xq))));z=(xx-xq)/scale
    w=1/(1+2*abs(z)) if weights else None;deg=min(degree,len(yy)-1)
    try:v=float(np.polynomial.polynomial.polyfit(z,yy,deg,w=w)[0])
    except Exception:v=float(np.average(yy,weights=w) if w is not None else np.mean(yy))
    lo,hi=np.quantile(yy,[.05,.95]);return float(np.clip(v,lo-.5,hi+.5))

def z_predict(frame, train=None, *, k=8, degree=1, crossyear=True, radius=None):
    """Return predictions aligned to synthetic rows in frame."""
    d=frame.copy().reset_index(drop=True);d['date']=pd.to_datetime(d.date);d['_year']=d.date.dt.year;d['_doy']=d.date.dt.dayofyear
    syn=d.is_synthetic_gap.astype(bool).to_numpy();y=d.primary_ndvi.to_numpy(float);known=np.isfinite(y)&~syn
    # Historical rows are context only.  We retain all columns needed for
    # climatology and target; synthetic private rows remain masked.
    if train is not None:
        h=train.copy();h['date']=pd.to_datetime(h.date);h['is_synthetic_gap']=False
        h['_year']=h.date.dt.year;h['_doy']=h.date.dt.dayofyear
        cols=['anon_polygon_id','date','primary_ndvi','ndvi_climatology_mean','ndvi_climatology_std','crop_type','is_synthetic_gap']
        allf=pd.concat([h[[c for c in cols if c in h]],d[[c for c in cols if c in d]]],ignore_index=True,sort=False)
    else:
        allf=d[['anon_polygon_id','date','primary_ndvi','ndvi_climatology_mean','ndvi_climatology_std','crop_type','is_synthetic_gap']].copy()
    allf['_year']=allf.date.dt.year;allf['_doy']=allf.date.dt.dayofyear
    amask=np.isfinite(allf.primary_ndvi.to_numpy(float)) & ~allf.is_synthetic_gap.astype(bool).to_numpy()
    mu=allf.ndvi_climatology_mean.to_numpy(float);sd=allf.ndvi_climatology_std.to_numpy(float);yy=allf.primary_ndvi.to_numpy(float)
    zz=(yy-mu)/sd; goodz=amask & np.isfinite(zz) & np.isfinite(mu)&np.isfinite(sd)&(sd>1e-4)&(abs(zz)<8)
    ids=allf.anon_polygon_id.to_numpy(object);yrs=allf._year.to_numpy(int);doys=allf._doy.to_numpy(int)
    ords=allf.date.map(pd.Timestamp.toordinal).to_numpy(float)
    # private rows are final segment; output hidden order
    start=len(allf)-len(d); qall=np.arange(start,len(allf)); qall=qall[syn]
    out=np.full(len(qall),np.nan); qpos={int(ix):j for j,ix in enumerate(qall)}
    # Base climatology profile and z-score source set by AOI/year.
    for q in qall:
        pid=ids[q];yr=yrs[q];dd=doys[q]
        same=np.flatnonzero(goodz & (ids==pid) & (yrs==yr))
        hist=np.flatnonzero(goodz & (ids==pid) & (yrs!=yr)) if crossyear else np.array([],dtype=int)
        # Query baseline: interpolate climatology from same-year known rows,
        # then historical AOI rows, then global/crop profiles.
        base_idx=np.flatnonzero(amask & (ids==pid) & (yrs==yr) & np.isfinite(mu) & np.isfinite(sd))
        if len(base_idx)<3 and crossyear:
            base_idx=np.flatnonzero(amask & (ids==pid) & np.isfinite(mu) & np.isfinite(sd))
        if len(base_idx):
            m=_local(dd,doys[base_idx].astype(float),mu[base_idx],k=24,degree=2)
            s=_local(dd,doys[base_idx].astype(float),sd[base_idx],k=24,degree=1)
        else:m=np.nan;s=np.nan
        # z interpolation primarily uses same-year anchors.  For sparse or
        # edge groups blend a cross-year seasonal z profile.
        zp=np.nan
        if len(same):
            zp=_local(float(ords[q]),ords[same],zz[same],k=k,degree=degree)
        if (not np.isfinite(zp) or len(same)<3) and len(hist):
            zp2=_local(dd,doys[hist].astype(float),zz[hist],k=max(k,12),degree=min(2,degree))
            if np.isfinite(zp2):zp=zp2 if not np.isfinite(zp) else .7*zp+.3*zp2
        if np.isfinite(m) and np.isfinite(s) and np.isfinite(zp):out[qpos[int(q)]]=np.clip(m+s*np.clip(zp,-5,5),-0.5,1.2)
    # fallback to production estimator for unresolved rows
    prod=predict_private(d,train=train,k=8,bin_days=30)
    keys=d.loc[syn,['anon_polygon_id','date']].copy();keys.date=keys.date.dt.strftime('%Y-%m-%d');pp=keys.merge(prod,on=['anon_polygon_id','date'],how='left').primary_ndvi_pred.to_numpy(float)
    out[~np.isfinite(out)]=pp[~np.isfinite(out)]
    return out,pp

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rows=[]
    for year in range(2019,2025):
        f,t=make_fold(tr,pr,year);q=f[f.is_synthetic_gap].copy();truth=t.to_numpy(float)
        for k in [3,4,6,8,12,16,24]:
          for deg in [0,1,2,3]:
            p,base=z_predict(f,None,k=k,degree=deg,crossyear=False);e=p-truth;rows.append((year,k,deg,'same',np.sqrt(np.mean(e*e)),np.mean(abs(e))))
            p,base=z_predict(f,None,k=k,degree=deg,crossyear=True);e=p-truth;rows.append((year,k,deg,'cross',np.sqrt(np.mean(e*e)),np.mean(abs(e))))
    out=pd.DataFrame(rows,columns=['year','k','degree','mode','rmse','mae']);agg=out.groupby(['k','degree','mode']).apply(lambda z:np.sqrt(np.average(z.rmse**2,weights=np.ones(len(z))))).reset_index(name='rmse').sort_values('rmse');print(agg.head(30).to_string(index=False));out.to_csv(ROOT/'research'/'zscore_eval_results.csv',index=False)
if __name__=='__main__':main()

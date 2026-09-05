"""Local Gaussian-process/state-space style imputers.

Research-only evaluator.  It uses source-calibrated observations and a small
RBF/Matérn kernel within each AOI/year.  Hyperparameters are deliberately
tested on hidden-date and private-like masks before any deployment change.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src'))
from infer import SOURCES,_prepare,_fit_source_maps,_mode_posteriors,_query_posterior
from validate import make_fold


def _map(v,s,maps,doy,forward=True):
    b=int(doy//30); a,k=maps.get(('s2',s,b),maps.get(('s2',s,'g'),(0.,1.)))
    if forward:return a+k*v
    return (v-a)/k if abs(k)>1e-8 else v


def _kernel(d,l,kind):
    u=np.abs(d)/max(1e-6,float(l))
    if kind=='matern32': return (1+np.sqrt(3)*u)*np.exp(-np.sqrt(3)*u)
    if kind=='matern52': return (1+np.sqrt(5)*u+5*u*u/3)*np.exp(-np.sqrt(5)*u)
    return np.exp(-.5*u*u)


def predict_gp(frame:pd.DataFrame, *, length=8., nugget=.05, signal=.20,
               window=60, kind='rbf', mean_mode='linear', source_mode='posterior'):
    z=_prepare(frame); y=z.primary_ndvi.to_numpy(float); known=np.isfinite(y)
    syn=frame.is_synthetic_gap.fillna(False).astype(bool).to_numpy(); src=z._src.to_numpy(object)
    doy=z._doy.to_numpy(int); year=z._year.to_numpy(int); x=z._ord.to_numpy(float); ids=z.anon_polygon_id.to_numpy(object)
    maps=_fit_source_maps(z,known,30); aoi,crop,glob,date=_mode_posteriors(z,known)
    can=np.full(len(z),np.nan)
    for i in np.flatnonzero(known): can[i]=_map(y[i],str(src[i]),maps,int(doy[i]),True)
    out=np.full(len(z),np.nan)
    groups=z.groupby(['anon_polygon_id','_year'],sort=False).groups
    # broad seasonal means as a stable prior; derive from visible canonical obs
    tmp=pd.DataFrame({'id':ids,'doy':doy,'can':can,'known':known})
    prof=tmp.loc[known].groupby(['id','doy']).can.median()
    globprof=tmp.loc[known].groupby('doy').can.median()
    for _,ii0 in groups.items():
        ii=np.asarray(ii0,dtype=int); kk=ii[known[ii]&np.isfinite(can[ii])]
        if not len(kk): continue
        # unique observations; avoid ill-conditioned duplicate dates
        for q in ii[syn[ii]]:
            dabs=np.abs(x[kk]-x[q]); take=np.argsort(dabs)
            if window is not None and window>0:
                take=take[dabs[take]<=float(window)]
            if len(take)<3: take=np.argsort(dabs)[:min(12,len(kk))]
            if len(take)==0: continue
            jo=kk[take]; xx=x[jo]; yy=can[jo]
            # prior mean: local linear interpolation when possible, otherwise
            # seasonal medians.  Constant prior is safer at sparse edges.
            if mean_mode=='seasonal':
                mm=np.array([prof.get((ids[j],int(doy[j])),globprof.get(int(doy[j]),np.nan)) for j in jo],float)
                mq=prof.get((ids[q],int(doy[q])),globprof.get(int(doy[q]),np.nan))
            else:
                mm=np.full(len(jo),np.nanmedian(yy));mq=np.nanmedian(yy)
                left=jo[x[jo]<x[q]]; right=jo[x[jo]>x[q]]
                if len(left) and len(right):
                    il=left[np.argmax(x[left])]; ir=right[np.argmin(x[right])]
                    if x[ir]>x[il]:
                        mq=can[il]+(can[ir]-can[il])*(x[q]-x[il])/(x[ir]-x[il])
                        # interpolate the same baseline at observations only
                        mm=np.array([mq + 0.0 for _ in jo])
                if not np.isfinite(mq): mq=np.nanmedian(yy)
            if not np.isfinite(mm).all(): mm=np.full(len(jo),np.nanmedian(yy))
            # Use a smooth prior if requested, but center around query prior.
            if mean_mode=='seasonal' and not np.isfinite(mq): mq=np.nanmedian(yy)
            dist=np.abs(xx[:,None]-xx[None,:]); K=(float(signal)**2)*_kernel(dist,length,kind)+(float(nugget)**2)*np.eye(len(jo))
            kq=(float(signal)**2)*_kernel(xx-float(x[q]),length,kind)
            try:
                coef=np.linalg.solve(K,yy-mm)
                val=float(mq+kq@coef)
            except np.linalg.LinAlgError:
                val=float(np.average(yy,weights=np.exp(-dabs[jo*0] if False else -np.abs(xx-x[q])/max(1.,length))))
            # Conservative range guard around local observations.
            lo,hi=np.quantile(yy,[.03,.97]) if len(yy)>=5 else (np.min(yy),np.max(yy)); val=float(np.clip(val,lo-.08,hi+.08))
            if source_mode=='oracle':
                target=str(frame.get('_true_src',pd.Series('s2',index=frame.index)).iat[q]); target=target if target in SOURCES else 's2';out[q]=_map(val,target,maps,int(doy[q]),False)
            else:
                p=_query_posterior(z,int(q),aoi,crop,glob,date,date_weight=1.0)
                vals=[_map(val,s,maps,int(doy[q]),False) for s in SOURCES]
                out[q]=vals[int(np.argmax(p))] if source_mode=='hard' else float(np.average(vals,weights=p))
    for q in np.flatnonzero(syn&~np.isfinite(out)):
        same=np.flatnonzero(known&(ids==ids[q]));out[q]=y[same[np.argmin(np.abs(x[same]-x[q]))]] if len(same) else np.nanmedian(y[known])
    return out[syn]


def random_mask(pr,seed,frac=.15,year=None):
    d=pr.copy().reset_index(drop=True); d.date=pd.to_datetime(d.date)
    if year is not None:d=d[d.date.dt.year.eq(year)].copy().reset_index(drop=True)
    d['_truth']=d.primary_ndvi.astype(float);d.is_synthetic_gap=False
    rng=np.random.default_rng(seed);mask=np.zeros(len(d),bool);pool=d.primary_ndvi.notna()
    for _,g in d.loc[pool].groupby('anon_polygon_id'):
        ix=g.index.to_numpy();mask[rng.choice(ix,max(1,int(round(frac*len(ix)))),replace=False)]=True
    dynamic=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','era5_temp_c','era5_precip_mm','year','primary_ndvi','doy','ndvi_climatology_mean','ndvi_climatology_std','ndvi_zscore','n_reference_years','status']
    for c in dynamic:
        if c in d:d.loc[mask,c]=np.nan
    d.loc[mask,'is_synthetic_gap']=True;return d,mask

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rec=[]
    configs=[(l,n,s,w,k,m) for l in [3,5,8,12,16,24] for n in [.02,.04,.06,.08,.12] for s in [.12,.20,.30] for w in [30,60,120] for k in ['rbf','matern32'] for m in ['constant','seasonal']]
    # Keep grid manageable: broad promising configurations first; all folds.
    configs=[(3,.04,.20,60,'rbf','constant'),(5,.04,.20,60,'rbf','constant'),(8,.04,.20,60,'rbf','constant'),(12,.04,.20,60,'rbf','constant'),(8,.06,.20,60,'rbf','constant'),(12,.06,.20,60,'rbf','constant'),(8,.08,.20,60,'rbf','constant'),(12,.08,.20,60,'rbf','constant'),(8,.04,.20,120,'rbf','seasonal'),(12,.04,.20,120,'rbf','seasonal'),(16,.06,.20,120,'matern32','seasonal'),(8,.06,.30,60,'matern32','constant'),(12,.06,.30,60,'matern32','constant')]
    for yr in [2019,2020,2021,2022,2023,2024]:
        f,t=make_fold(tr,pr,yr);y=t.to_numpy(float)
        for l,n,s,w,k,m in configs:
            p=predict_gp(f,length=l,nugget=n,signal=s,window=w,kind=k,mean_mode=m);e=p-y;rec.append(dict(protocol='exact',year=yr,length=l,nugget=n,signal=s,window=w,kernel=k,mean=m,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(y)))
        print('exact',yr,flush=True)
    for seed in [0,1,2]:
      for yr in [None,2025]:
        f,mask=random_mask(pr,seed,year=yr);y=f.loc[mask,'_truth'].to_numpy(float)
        for l,n,s,w,k,m in configs[:8]+configs[8:]:
            p=predict_gp(f,length=l,nugget=n,signal=s,window=w,kind=k,mean_mode=m);e=p-y;rec.append(dict(protocol='random2025' if yr else 'random',year=yr or 0,seed=seed,length=l,nugget=n,signal=s,window=w,kernel=k,mean=m,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(y)))
        print('random',seed,yr,flush=True)
    out=pd.DataFrame(rec);out.to_csv(ROOT/'research/gp_eval_results.csv',index=False)
    print(out.groupby(['protocol','length','nugget','signal','window','kernel','mean'],as_index=False).apply(lambda g:pd.Series(rmse=np.sqrt(np.average(g.rmse**2,weights=g.n)),mae=np.average(g.mae,weights=g.n)),include_groups=False).sort_values(['protocol','rmse']).groupby('protocol').head(30).to_string(index=False))
if __name__=='__main__':main()

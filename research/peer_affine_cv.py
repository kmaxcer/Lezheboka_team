"""Cross-AOI peer transfer experiment.

Private rows from different AOIs share acquisition dates.  This script fits
target-AOI-specific affine maps from peer AOIs using only unmasked rows, then
tests same-date transfer on realistic random masks.  It is intentionally
standalone and writes diagnostics under ``research``.
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
from infer_lag import predict_private_lag

DYN=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','era5_temp_c','era5_precip_mm','year','primary_ndvi','doy','ndvi_climatology_mean','ndvi_climatology_std','n_reference_years','status']

def masked_private(pr,seed,frac=.15):
    d=pr.copy().sort_values(['anon_polygon_id','date']).reset_index(drop=True)
    d['_truth']=d.primary_ndvi.astype(float);d['_true_src']=np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],['s2','landsat','modis'],'none');d['is_synthetic_gap']=False
    rng=np.random.default_rng(seed);mask=np.zeros(len(d),bool);pool=d.primary_ndvi.notna();yrs=d.date.dt.year
    for _,ix in d.loc[pool].groupby(['anon_polygon_id',yrs],sort=False).groups.items():
        ix=np.asarray(ix,dtype=int); n=max(1,int(round(frac*len(ix))));mask[rng.choice(ix,min(n,len(ix)),replace=False)]=True
    for c in DYN:
        if c in d:d.loc[mask,c]=np.nan
    d.loc[mask,'is_synthetic_gap']=True
    return d,mask

def robust_aff(x,y):
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
    if len(x)<12:return (float(np.nanmedian(y)) if len(y) else .3),0.,0.,len(x)
    keep=(x>=np.quantile(x,.03))&(x<=np.quantile(x,.97))&(y>=np.quantile(y,.03))&(y<=np.quantile(y,.97))
    if keep.sum()<10:keep=np.ones(len(x),bool)
    b,a=np.polyfit(x[keep],y[keep],1);pred=a+b*x;rm=np.sqrt(np.mean((pred-y)**2))
    if not np.isfinite(a+b) or abs(b)>4:return float(np.median(y)),0.,float(rm),len(x)
    return float(a),float(b),float(rm),len(x)

def peer_predict(d,mask, *, topk=5, min_n=20, same_crop=False):
    # Matrix of target values.  Dates are exact; peers at a hidden date are
    # available only if their own target is observed.
    obs=~mask & d.primary_ndvi.notna()
    z=d.loc[:,['date','anon_polygon_id','primary_ndvi','crop_type']].copy();z['_obs']=obs
    piv=z.pivot_table(index='date',columns='anon_polygon_id',values='primary_ndvi',aggfunc='first')
    avail=z[z._obs].pivot_table(index='date',columns='anon_polygon_id',values='primary_ndvi',aggfunc='first')
    ids=list(piv.columns); crops=z.groupby('anon_polygon_id').crop_type.first().astype(str).to_dict()
    models={}
    for a in ids:
        ya=avail[a] if a in avail else pd.Series(dtype=float)
        for b in ids:
            if a==b or b not in avail:continue
            xy=pd.concat([ya,avail[b]],axis=1,keys=['y','x']).dropna()
            if same_crop and crops.get(a)!=crops.get(b):continue
            if len(xy)<min_n:continue
            aa,bb,rm,n=robust_aff(xy.x.to_numpy(float),xy.y.to_numpy(float));models[(a,b)]=(aa,bb,rm,n)
    out=np.full(mask.sum(),np.nan); qrows=d.index[mask].to_numpy(); qpos={int(ix):j for j,ix in enumerate(qrows)}
    # Iterate hidden rows; peer count is small and exact-date lookup is fast.
    for ix in qrows:
        dt=d.date.iat[ix];a=d.anon_polygon_id.iat[ix];
        if dt not in avail.index:continue
        row=avail.loc[dt];cand=[]
        for b,val in row.items():
            if not np.isfinite(val) or (a,b) not in models:continue
            aa,bb,rm,n=models[(a,b)];cand.append((rm,n,aa+bb*float(val),b))
        if not cand:continue
        cand.sort(key=lambda q:(q[0],-q[1]));cand=cand[:topk]
        # Inverse-MSE weights plus a tiny n preference.
        w=np.array([1/(c[0]**2+.001) * min(1.,c[1]/80) for c in cand]);
        out[qpos[int(ix)]]=np.average([c[2] for c in cand],weights=w)
    return out

def score(arr,q,mask):
    # ``q`` already contains only the hidden rows; retain the mask argument
    # for call-site compatibility with earlier experiments.
    y=q['_truth'].to_numpy(float);e=arr-y;ok=np.isfinite(e)
    return np.sqrt(np.mean(e[ok]**2)) if ok.any() else np.nan, np.mean(np.abs(e[ok])) if ok.any() else np.nan, int(ok.sum())

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rows=[]
    for seed in [0,1,2]:
        d,m=masked_private(pr,seed);q=d.loc[m].copy();
        base=predict_private(d,train=tr,k=8,bin_days=30);lag=predict_private_lag(d,train=tr,k=16,degree=3,bin_days=30)
        keys=q[['anon_polygon_id','date']].copy();
        def align(o):
            x=o.copy();x.date=pd.to_datetime(x.date);return keys.merge(x,on=['anon_polygon_id','date'],how='left',validate='one_to_one').primary_ndvi_pred.to_numpy(float)
        bp=align(base);lp=align(lag)
        for crop in [False,True]:
          for k in [1,3,5,10]:
            pp=peer_predict(d,m,topk=k,same_crop=crop)
            for name,arr in [('peer',pp),('blend25',.75*bp+.25*pp),('blend50',.5*bp+.5*pp),('lagblend25',.75*lp+.25*pp)]:
                rm,ma,n=score(arr,q,m);rows.append((seed,crop,k,name,rm,ma,n));print(seed,crop,k,name,rm,flush=True)
    out=pd.DataFrame(rows,columns=['seed','same_crop','k','method','rmse','mae','n']);out.to_csv(ROOT/'research'/'peer_affine_cv_results.csv',index=False)
    agg=out.groupby(['same_crop','k','method']).apply(lambda z:pd.Series(rmse=np.sqrt(np.average(z.rmse**2,weights=z.n)),mae=np.average(z.mae,weights=z.n))).reset_index().sort_values('rmse');print(agg.head(30).to_string(index=False));agg.to_csv(ROOT/'research'/'peer_affine_cv_agg.csv',index=False)
if __name__=='__main__':main()

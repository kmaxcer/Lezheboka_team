"""Evaluate anomaly-residual interpolation using visible climatology fields.

The private file leaves climatology values on observed rows, while masking them
on synthetic gaps.  This diagnostic asks whether a local residual/z-score
model can recover suppression/anomaly episodes that a plain target interpolator
shrinks away.  No input or production output is modified.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold

DYN=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','era5_temp_c','era5_precip_mm','year','primary_ndvi','doy','ndvi_climatology_mean','ndvi_climatology_std','ndvi_zscore','n_reference_years','status']

def _mask_private(pr,seed,frac=.15,year=None):
 d=pr.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date)
 if year is not None:d=d[d.date.dt.year.eq(year)].copy().reset_index(drop=True)
 d['_truth']=d.primary_ndvi.astype(float);d.is_synthetic_gap=False
 rng=np.random.default_rng(seed);mask=np.zeros(len(d),bool)
 for _,g in d[d.primary_ndvi.notna()].groupby('anon_polygon_id'):
  ix=g.index.to_numpy();mask[rng.choice(ix,max(1,int(round(frac*len(ix)))),replace=False)]=True
 for c in DYN:
  if c in d:d.loc[mask,c]=np.nan
 d.loc[mask,'is_synthetic_gap']=True;return d,mask

def predict(frame,method='rawres',band=8.,k=8,shrink=1.0):
 d=frame.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date);d['_yr']=d.date.dt.year.to_numpy();d['_doy']=d.date.dt.dayofyear.to_numpy();d['_q']=d.is_synthetic_gap.fillna(False).astype(bool).to_numpy();y=d.primary_ndvi.to_numpy(float);known=np.isfinite(y);out=np.full(len(d),np.nan)
 for _,g in d.groupby(['anon_polygon_id','_yr'],sort=False):
  ix=g.index.to_numpy(); q=ix[d.loc[ix,'_q'].to_numpy()]; ok=ix[known[ix]]
  if len(ok)==0:continue
  x=d.loc[ix,'_doy'].to_numpy(float); xo=d.loc[ok,'_doy'].to_numpy(float); yy=y[ok]
  # Visible climatology fields; interpolate by day within this AOI/year.
  cm=d.loc[ok,'ndvi_climatology_mean'].to_numpy(float); cs=d.loc[ok,'ndvi_climatology_std'].to_numpy(float)
  goodm=np.isfinite(cm); goods=np.isfinite(cs)
  if goodm.sum()>=2:
   order=np.argsort(xo[goodm]); xm=xo[goodm][order]; vm=cm[goodm][order]
  else: xm=np.array([]);vm=np.array([])
  if goods.sum()>=2:
   order=np.argsort(xo[goods]); xs=xo[goods][order]; vs=cs[goods][order]
  else: xs=np.array([]);vs=np.array([])
  # Residuals/z scores at observed points; use a robust fallback baseline.
  if len(xm): base_o=np.interp(xo,xm,vm)
  else: base_o=np.full(len(ok),np.nanmedian(yy))
  if len(xs): std_o=np.interp(xo,xs,vs)
  else: std_o=np.full(len(ok),np.nanstd(yy) if np.nanstd(yy)>1e-3 else .15)
  res=yy-base_o; zres=res/np.maximum(std_o,.02)
  for qi in q:
   xq=float(d.loc[qi,'_doy'])
   if len(xm): mq=float(np.interp(xq,xm,vm))
   else: mq=float(np.nanmedian(yy))
   if len(xs): sq=float(np.interp(xq,xs,vs))
   else: sq=float(np.nanmedian(std_o))
   dist=np.abs(xo-xq); sel=np.argsort(dist)[:min(int(k),len(ok))]
   w=np.exp(-dist[sel]/max(.5,float(band)))
   if method=='zscore': adj=float(np.average(zres[sel],weights=w))*sq
   elif method=='median': adj=float(np.median(res[sel]))
   elif method=='near': adj=float(res[sel[0]])
   else: adj=float(np.average(res[sel],weights=w))
   if method=='none': val=mq
   else: val=mq+float(shrink)*adj
   out[qi]=np.clip(val,-.5,1.2)
 # cross-year fallback uses target nearest; uncommon in exact folds
 for qi in np.flatnonzero(d['_q'].to_numpy()&~np.isfinite(out)):
  same=np.flatnonzero(known&(d.anon_polygon_id.to_numpy()==d.anon_polygon_id.iat[qi]));out[qi]=y[same[np.argmin(abs(d.loc[same,'_doy'].to_numpy()-d.loc[qi,'_doy']))]] if len(same) else np.nanmedian(y[known])
 return out[d['_q'].to_numpy()]

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
 rec=[]
 methods=[]
 for meth in ['none','rawres','zscore','median','near']:
  for band in [2,4,6,8,12,20,30]:
   for k in [1,2,3,5,8,12,20]:
    if meth=='none' and (band!=8 or k!=8):continue
    methods.append((meth,band,k))
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr);yt=t.to_numpy(float)
  for meth,b,k in methods:
   p=predict(f,meth,b,k);e=p-yt;rec.append(dict(protocol='exact',year=yr,method=meth,band=b,k=k,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(yt)))
  print('exact',yr,flush=True)
 for seed in [0,1,2]:
  for yr in [None,2025]:
   f,m=_mask_private(pr,seed,year=yr);yt=f.loc[m,'_truth'].to_numpy(float)
   for meth,b,k in methods:
    p=predict(f,meth,b,k);e=p-yt;rec.append(dict(protocol='random2025' if yr else 'random',year=yr or 0,seed=seed,method=meth,band=b,k=k,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(yt)))
   print('random',seed,yr,flush=True)
 out=pd.DataFrame(rec);out.to_csv(ROOT/'research/anomaly_eval_results.csv',index=False)
 rows=[]
 for key,g in out.groupby(['protocol','method','band','k']): rows.append((*key,np.sqrt(np.average(g.rmse**2,weights=g.n)),np.average(g.mae,weights=g.n),g.n.sum()))
 print(pd.DataFrame(rows,columns=['protocol','method','band','k','rmse','mae','n']).sort_values(['protocol','rmse']).groupby('protocol').head(30).to_string(index=False))
if __name__=='__main__':main()

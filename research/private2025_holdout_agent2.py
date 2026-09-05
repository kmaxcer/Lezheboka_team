"""Pseudo-holdout on observed private 2025 rows (same shape as hidden data)."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private
from validate import make_fold

DYN=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','era5_temp_c','era5_precip_mm','year','primary_ndvi','doy','ndvi_climatology_mean','ndvi_climatology_std','ndvi_zscore','n_reference_years','status']
def holdout(pr, seed=0, frac=0.18):
 rng=np.random.default_rng(seed); d=pr[pr.date.dt.year==2025].copy().reset_index(drop=True); d['is_synthetic_gap']=False; d['_truth']=d.primary_ndvi
 # sample only observed target rows; preserve roughly the observed hidden rate
 mask=np.zeros(len(d),bool)
 for pid,idx in d[d.primary_ndvi.notna()].groupby('anon_polygon_id').groups.items():
  idx=np.asarray(idx); n=max(1,int(round(len(idx)*frac))); mask[rng.choice(idx,n,replace=False)]=True
 for c in DYN:
  if c in d: d.loc[mask,c]=np.nan
 d.loc[mask,'is_synthetic_gap']=True
 return d,mask
def with_history(d,tr):
 hist=tr[tr.date.dt.year<2025].copy(); hist['is_synthetic_gap']=False
 cols=d.columns.intersection(hist.columns).tolist(); return pd.concat([hist[cols],d[cols]],ignore_index=True,sort=False)
def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 for seed in [0,1,2]:
  d,m=holdout(pr,seed); q=d[m]; y=q._truth.to_numpy(float)
  for label,frame in [('private',d),('history',with_history(d,tr))]:
   out=predict_private(frame,train=None); keys=q[['anon_polygon_id','date']].copy();keys.date=keys.date.dt.strftime('%Y-%m-%d'); pred=keys.merge(out,on=['anon_polygon_id','date'],how='left').primary_ndvi_pred.to_numpy(float);e=pred-y; print(seed,label,'all',len(y),np.sqrt(np.mean(e*e)),'overlap',np.mean(q.anon_polygon_id.isin(tr.anon_polygon_id)), 'new_rmse',np.sqrt(np.mean(e[~q.anon_polygon_id.isin(tr.anon_polygon_id)]**2)) if (~q.anon_polygon_id.isin(tr.anon_polygon_id)).any() else np.nan)
  # baseline by observed/private group only and history frame comparisons
if __name__=='__main__':main()

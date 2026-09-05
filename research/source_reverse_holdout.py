from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); sys.path[:0]=[str(ROOT/'research'),str(ROOT/'src')]
from evaluate_private_cohort_blend import make_holdout
DYNAMIC=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','modis_ndwi','era5_temp_c','era5_precip_mm','year','doy','ndvi_climatology_mean','ndvi_climatology_std','n_reference_years','primary_ndvi']
from overnight_source_eval import _predict_matrix

def route_nc(d,q,w=6):
 d=d.reset_index(drop=True); q=np.asarray(q); s=np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],[0,1,2],-1); d['_idn']=d.anon_polygon_id.str.extract(r'(\d+)',expand=False).astype(int); dates=d.date.to_numpy(); ids=d._idn.to_numpy(); cr=d.crop_type.astype(str).to_numpy(); vis=np.flatnonzero((~q)&(s>=0)); out=[]
 for i in np.flatnonzero(q):
  z=vis[(dates[vis]==dates[i])&(np.abs(ids[vis]-ids[i])<=w)]; zz=z[cr[z]==cr[i]]
  if len(zz)==0: zz=z
  out.append(np.bincount(s[zz],minlength=3).argmax() if len(zz) else -1)
 return np.array(out)

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False); pr['is_synthetic_gap']=pr.is_synthetic_gap.fillna(False).astype(bool); hold=make_holdout(pr); actual=pr.is_synthetic_gap.to_numpy(bool); gaps=hold|actual; pm=pr.copy();
 for c in DYNAMIC:
  if c in pm: pm.loc[gaps,c]=np.nan
 pm.loc[gaps,'is_synthetic_gap']=True
 # source experts on private + train calibration
 print('predict base',flush=True); b,_=_predict_matrix(pm,train=tr,family='base',k=8,degree=1,bin_days=30,date_weight=1.0)
 print('predict lag',flush=True); l,_=_predict_matrix(pm,train=tr,family='lag',k=16,degree=3,bin_days=30,date_weight=1.0)
 qi=np.flatnonzero(hold); qbase=b.iloc[np.searchsorted(np.flatnonzero(gaps),qi)] if False else b.set_index('row_index').loc[qi]; qlag=l.set_index('row_index').loc[qi]
 route=route_nc(pm,gaps,6); route=route[np.isin(np.flatnonzero(gaps),qi)] # same order qi because hold subset; safer below
 # exact route in qi order
 route=route_nc(pm,hold,6)
 matb=qbase[['pred_s2','pred_landsat','pred_modis']].to_numpy(float); matl=qlag[['pred_s2','pred_landsat','pred_modis']].to_numpy(float); ext40=pd.read_csv(ROOT/'research/private_cohort_blend_holdout_predictions.csv',parse_dates=['date']); y=pr.loc[hold,'primary_ndvi'].to_numpy(float); ext40=ext40.set_index(['anon_polygon_id','date']).loc[pd.MultiIndex.from_frame(pr.loc[hold,['anon_polygon_id','date']]),'ext40'].to_numpy(float)
 r=[]
 for a in [0,.1,.2,.3,.4,.5, .7,1.0]:
  src=np.array([matb[i,j] if j>=0 and np.isfinite(matb[i,j]) else qbase.hard.iloc[i] for i,j in enumerate(route)]); p=(1-a)*ext40+a*src; r.append({'w':a,'rmse':float(np.sqrt(np.mean((p-y)**2)))})
 print(pd.DataFrame(r).to_string(index=False)); out=pd.DataFrame({'anon_polygon_id':pr.loc[hold,'anon_polygon_id'].to_numpy(),'date':pr.loc[hold,'date'].to_numpy(),'truth':y,'ext40':ext40,'route_base': [matb[i,j] if j>=0 and np.isfinite(matb[i,j]) else qbase.hard.iloc[i] for i,j in enumerate(route)],'route':route}); out.to_csv(ROOT/'research/source_reverse_holdout.csv',index=False)
 # cohorts
 out['year']=pd.to_datetime(out.date).dt.year; out['cohort']=np.where(out.anon_polygon_id.isin(set(tr.anon_polygon_id)),'shared','new')
 for key,g in out.groupby('cohort'):
  print(key,len(g),[(a,float(np.sqrt(np.mean((((1-a)*g.ext40+a*g.route_base)-g.truth)**2)))) for a in [0,.2,.4,.6,1]])
if __name__=='__main__':main()

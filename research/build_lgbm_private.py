"""Build a LightGBM extended-feature component for the real private gaps."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); OUT=ROOT/'outputs'; RES=ROOT/'research'; sys.path.insert(0,str(RES))
from build_extended_hgb_private import _clear, _matrix
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 t=time.time(); tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
 tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool); tr['_origin']='train'; pr['_origin']='private'; d=pd.concat([tr,pr],ignore_index=True,sort=False).sort_values([ID,DATE,'_origin']).reset_index(drop=True); d[DATE]=pd.to_datetime(d[DATE]); d['year']=d['year'].fillna(d[DATE].dt.year).astype(int); d['doy']=d['doy'].fillna(d[DATE].dt.dayofyear).astype(int); d['_truth']=pd.to_numeric(d[TARGET],errors='coerce'); hidden=d[GAP].to_numpy(bool); qi=np.flatnonzero(hidden); blocks=[]; ys=[]; known=d[TARGET].notna().to_numpy(bool)&~hidden; years=d[DATE].dt.year.to_numpy(int)
 for rep in range(2):
  rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(d),bool); tab=pd.DataFrame({'id':d[ID].astype(str),'year':years})
  for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
   ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
  comb=hidden|pm; obs=d[TARGET].where(~comb); print('features train',rep,int(pm.sum()),flush=True); x=_matrix(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True))
 obs=d[TARGET].where(~hidden); print('features query',len(qi),flush=True); qx=_matrix(d,obs,hidden).loc[hidden].reset_index(drop=True); X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float)
 m=LGBMRegressor(objective='regression',verbosity=-1,random_state=42,n_jobs=-1,subsample=.9,colsample_bytree=.9,n_estimators=700,learning_rate=.018,num_leaves=127,min_child_samples=80,reg_lambda=24.,reg_alpha=1.,max_bin=127).fit(X,y); p=np.clip(m.predict(qx),-.2,1.1); keys=d.loc[hidden,[ID,DATE]].copy(); keys['primary_ndvi_pred']=p; out=OUT/'model_dani_lgbm_extended.csv'; keys.to_csv(out,index=False,float_format='%.8f')
 meta={'rows':len(keys),'features':X.shape[1],'pseudo_masks':2,'model':'LGBMRegressor deep','private_sha256':sha(DATA/'private_features.csv'),'sha256':sha(out),'seconds':round(time.time()-t,1),'production_baseline_overwritten':False}; (OUT/'model_dani_lgbm_extended_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(meta,flush=True)
if __name__=='__main__': main()

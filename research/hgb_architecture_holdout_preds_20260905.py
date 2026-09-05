from pathlib import Path
import numpy as np,pandas as pd,sys,time,json
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path('.'); R=ROOT/'research'; D=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); sys.path.insert(0,str(ROOT/'_archive_inspect'/'agropulse_max_score'/'src')); from agropulse.pipeline import build_features; sys.path.insert(0,str(R)); from feature_hgb_v2 import _clear,extra_features
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
def mat(d,obs,m):
 fr=_clear(d,m); bx=build_features(fr,obs,pd.Series(np.asarray(m,bool))); ex=extra_features(fr,obs,np.asarray(m,bool)); return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)
tr=pd.read_csv(D/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/'private_features.csv',parse_dates=[DATE],low_memory=False); gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=[DATE]).rename(columns={'primary_ndvi_true':TARGET}); pr[GAP]=pr[GAP].fillna(False).astype(bool); hidden=pr[GAP].to_numpy(bool); qkeys=pr.loc[hidden,[ID,DATE]]; truth=qkeys.merge(gt[[ID,DATE,TARGET]],on=[ID,DATE],validate='one_to_one')[TARGET].to_numpy(float)
pr.loc[hidden,TARGET]=np.nan
for f in (tr,pr): f[GAP]=False; f['_origin']='train' if f is tr else 'private'; f['date']=pd.to_datetime(f.date); f['year']=f.year.fillna(f.date.dt.year).astype(int); f['doy']=f.doy.fillna(f.date.dt.dayofyear).astype(int)
d=pd.concat([tr,pr],ignore_index=True,sort=False); d['_truth']=pd.to_numeric(d[TARGET],errors='coerce'); known=d[TARGET].notna().to_numpy(bool); hm=np.r_[np.zeros(len(tr),bool),hidden]; ids=d[ID].astype(str).to_numpy(); yrs=d.date.dt.year.to_numpy(int); blocks=[]; ys=[]
for rep in range(3):
 rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(d),bool); tab=pd.DataFrame({'id':ids,'year':yrs})
 for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
  ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
 comb=hm|pm; obs=d[TARGET].where(~comb); print('block',rep+1,int(pm.sum()),flush=True); x=mat(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True))
obs=d[TARGET].where(~hm); print('query',len(hidden),flush=True); qx=mat(d,obs,hm).loc[hm].reset_index(drop=True); X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float)
specs=[('regular',dict(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12)),('wide',dict(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=8)),('abs',dict(loss='absolute_error',learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12)),('smooth',dict(loss='squared_error',learning_rate=.02,max_iter=500,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=20))]
rec=[]; preds={} 
for name,sp in specs:
 t=time.time(); m=HistGradientBoostingRegressor(random_state=42,**sp); m.fit(X,y); p=np.clip(m.predict(qx),-.2,1.1); preds[name]=p; rr=float(np.sqrt(np.mean((p-truth)**2))); rec.append({'model':name,'rmse':rr,'mae':float(np.mean(abs(p-truth))),'mean':float(p.mean()),'seconds':round(time.time()-t,1)}); print(rec[-1],flush=True)
pd.DataFrame(rec).to_csv(R/'hgb_architecture_holdout_corrected_20260905.csv',index=False); np.savez(R/'hgb_architecture_holdout_preds_20260905.npz',truth=truth,regular=preds['regular'],wide=preds['wide'],abs=preds['abs'],smooth=preds['smooth']); print(rec)

import os,sys,time,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from threadpoolctl import threadpool_limits
ROOT=Path('.'); DATA=Path(r'C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904'); R=ROOT/'research'; sys.path.insert(0,str(R)); import hgb_54_safe_probe_20260905_1810 as H
from agropulse.pipeline import load_competition_data
ID,DATE,TARGET,GAP=H.ID,H.DATE,H.TARGET,H.GAP
tr,private,ref=load_competition_data(DATA/'train_dataset.csv',DATA/'private_features.csv'); actual=(ref['_origin'].eq('test') & ref[GAP].fillna(False)).to_numpy(bool); known=ref[TARGET].notna().to_numpy(bool)&~actual
outer=H.stratified_mask(ref,known & ref['_origin'].eq('test').to_numpy(),20260905,.15);train_pool=known&~outer;hidden=actual|outer
blocks=[];ys=[]
for seed in (11,29,47):
 pm=H.stratified_mask(ref,train_pool,seed); fx=H.safe_features(ref,hidden|pm);blocks.append(fx.loc[pm]);ys.append(ref.loc[pm,TARGET])
X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).to_numpy(float); q=H.safe_features(ref,hidden); xo=q.loc[outer]; xg=q.loc[actual]
gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=[DATE]); keys=ref.loc[actual,[ID,DATE]];yg=keys.merge(gt,on=[ID,DATE],validate='one_to_one').primary_ndvi_true.to_numpy(float); yo=ref.loc[outer,TARGET].to_numpy(float)
base=pd.read_csv(ROOT/'outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv',parse_dates=[DATE]); baseo=base.merge(ref.loc[outer,[ID,DATE]],on=[ID,DATE],validate='one_to_one').primary_ndvi_pred.to_numpy(float); baseg=base.merge(keys,on=[ID,DATE],validate='one_to_one').primary_ndvi_pred.to_numpy(float)
def rm(y,p):return float(np.sqrt(np.mean((y-p)**2)))
rows=[]
for name,spec in [('et',dict(n_estimators=300,max_features=.7,min_samples_leaf=5,max_depth=None)),('et10',dict(n_estimators=300,max_features=1.0,min_samples_leaf=10,max_depth=None)),('rf',dict(n_estimators=250,max_features=.7,min_samples_leaf=5,max_depth=None))]:
 cls=ExtraTreesRegressor if name.startswith('et') else RandomForestRegressor; m=cls(random_state=42,n_jobs=3,**spec); t=time.time();
 with threadpool_limits(3):m.fit(X,y)
 po=np.clip(m.predict(xo),-.2,1.1);pg=np.clip(m.predict(xg),-.2,1.1); rows.append({'model':name,'outer_rmse':rm(yo,po),'released_rmse':rm(yg,pg),'base_outer':float('nan'),'base_released':rm(yg,baseg),'seconds':round(time.time()-t,1)}); print(rows[-1],flush=True)
 for w in [.05,.1,.2,.3]: rows.append({'model':name+f'_blend{w}','outer_rmse':rm(yo,(1-w)*baseo+w*po),'released_rmse':rm(yg,(1-w)*baseg+w*pg),'base_outer':float('nan'),'base_released':rm(yg,baseg),'seconds':0})
out=R/'compact_tree_probe_20260905_results.csv';pd.DataFrame(rows).to_csv(out,index=False); rep=R/'compact_tree_probe_20260905_report.md';rep.write_text('# Compact tree probe\n\nLeakage-safe 3x AOI/year pseudo masks; all dynamic fields masked on query and pseudo rows. ExtraTrees/RandomForest on 69 build_features+extra_features columns.\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nNo candidate materialized because this probe is diagnostic until independent improvement is confirmed. Upload not performed.\n',encoding='utf8');print(pd.DataFrame(rows).to_string(index=False))


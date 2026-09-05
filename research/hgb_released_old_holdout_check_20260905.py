from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'research'; D=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
import sys;sys.path.insert(0,str(ROOT/'_archive_inspect'/'agropulse_max_score'/'src')); from agropulse.pipeline import build_features
sys.path.insert(0,str(R)); from feature_hgb_v2 import _clear,extra_features
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
def mat(d,obs,m):
 fr=_clear(d,m); bx=build_features(fr,obs,pd.Series(np.asarray(m,bool))); ex=extra_features(fr,obs,np.asarray(m,bool)); return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)
def main():
 t0=time.time(); tr=pd.read_csv(D/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/'private_features.csv',parse_dates=[DATE],low_memory=False); gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=[DATE]); gt=gt.rename(columns={'primary_ndvi_true':TARGET}); pr[GAP]=pr[GAP].fillna(False).astype(bool); hidden=pr[GAP].to_numpy(bool); truth=pr.loc[hidden,[ID,DATE]].merge(gt[[ID,DATE,TARGET]],on=[ID,DATE],how='left',validate='one_to_one')[TARGET].to_numpy(float)
 # Exclude released old-gap labels from all fit/context features.
 pr.loc[hidden,TARGET]=np.nan; tr[GAP]=False; tr['_origin']='train'; pr['_origin']='private';
 for f in (tr,pr): f['date']=pd.to_datetime(f.date); f['year']=f.year.fillna(f.date.dt.year).astype(int); f['doy']=f.doy.fillna(f.date.dt.dayofyear).astype(int)
 d=pd.concat([tr,pr],ignore_index=True,sort=False); d['_truth']=pd.to_numeric(d[TARGET],errors='coerce'); known=d[TARGET].notna().to_numpy(bool)&~np.r_[np.zeros(len(tr),bool),hidden]; qi=np.flatnonzero(np.r_[np.zeros(len(tr),bool),hidden]); ids=d[ID].astype(str).to_numpy(); yrs=d.date.dt.year.to_numpy(int); blocks=[]; ys=[]
 for rep in range(3):
  rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(d),bool); tab=pd.DataFrame({'id':ids,'year':yrs})
  for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
   ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
  comb=np.r_[np.zeros(len(tr),bool),hidden]|pm; obs=d[TARGET].where(~comb); print('block',rep+1,int(pm.sum()),flush=True); x=mat(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True))
 comb=np.r_[np.zeros(len(tr),bool),hidden]; obs=d[TARGET].where(~comb); print('query',len(qi),flush=True); qx=mat(d,obs,comb).loc[qi].reset_index(drop=True); X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float); rec=[]
 for name,spec in [('regular',dict(learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.)),('wide',dict(learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=8.))]:
  m=HistGradientBoostingRegressor(loss='squared_error',random_state=42,**spec); m.fit(X,y); p=np.clip(m.predict(qx),-.2,1.1); rec.append({'model':name,'n':len(truth),'rmse':float(np.sqrt(np.mean((p-truth)**2))),'mae':float(np.mean(abs(p-truth))),'pred_mean':float(p.mean())}); print(rec[-1],flush=True)
 out=pd.DataFrame(rec); out.to_csv(R/'hgb_released_old_holdout_check_20260905.csv',index=False,float_format='%.10f'); report={'protocol':'old private synthetic gaps held out; released labels excluded from training/context; old train + old private visible only; 3x18% AOI-year pseudo-mask blocks','metrics':rec,'seconds':round(time.time()-t0,1),'no_upload':True}; (R/'hgb_released_old_holdout_check_20260905_report.md').write_text('# HGB architecture holdout\n\n'+json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()

from pathlib import Path
import sys,time,json
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); R=ROOT/'research'; ARCH=ROOT/'_archive_inspect'/'agropulse_max_score'/'src'; sys.path.insert(0,str(ARCH)); sys.path.insert(0,str(R))
from agropulse.pipeline import build_features
from feature_hgb_v2 import _clear
from feature_hgb_v3 import extra_features_v3
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
def make_holdout(pr,seed):
 known=pr[TARGET].notna().to_numpy(bool)&~pr[GAP].fillna(False).to_numpy(bool); out=np.zeros(len(pr),bool); rng=np.random.default_rng(seed); yy=pd.to_datetime(pr[DATE]).dt.year
 for _,ix0 in pr.loc[known].groupby([ID,yy],sort=False).groups.items():
  ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.15*len(ix)))); out[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
 return out
def matrix(d,obs,mask):
 fr=_clear(d,mask); bx=build_features(fr,obs,pd.Series(mask,index=fr.index)); ex=extra_features_v3(fr,obs,mask); return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)
def main(seed=2,n_masks=2):
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False); tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool)
 hold=make_holdout(pr,seed); gaps_pr=pr[GAP].to_numpy(bool)|hold; tr['_origin']='train'; pr['_origin']='private'; ref=pd.concat([tr,pr],ignore_index=True,sort=False); ref[DATE]=pd.to_datetime(ref[DATE]); ref['year']=ref['year'].fillna(ref[DATE].dt.year).astype(int); ref['doy']=ref['doy'].fillna(ref[DATE].dt.dayofyear).astype(int); ref['_truth']=pd.to_numeric(ref[TARGET],errors='coerce')
 hk=set(map(tuple,pr.loc[gaps_pr,[ID,DATE]].to_numpy())); gaps=np.array([tuple(x) in hk for x in ref[[ID,DATE]].to_numpy()],bool); ref.loc[gaps,TARGET]=np.nan; known=ref['_truth'].notna().to_numpy(bool)&~gaps; tab=pd.DataFrame({'id':ref[ID].astype(str),'year':ref[DATE].dt.year.to_numpy(int)}); blocks=[]; ys=[]; t0=time.time()
 for rep in range(n_masks):
  rng=np.random.default_rng(202600+seed*10+rep); pm=np.zeros(len(ref),bool)
  for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
   ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
  comb=gaps|pm; obs=ref[TARGET].where(~comb); print('features',rep,pm.sum(),flush=True); x=matrix(ref,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(ref.loc[pm,'_truth'].reset_index(drop=True))
 obs=ref[TARGET].where(~gaps); print('query features',gaps.sum(),flush=True); qx=matrix(ref,obs,gaps).loc[gaps].reset_index(drop=True); X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float); yy=np.clip(y,-.10,1.0); print('fit',X.shape,flush=True)
 m=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.,random_state=42); m.fit(X,yy); p=np.clip(m.predict(qx),-.2,1.1)
 q=ref.loc[gaps,[ID,DATE,'_truth']].copy().reset_index(drop=True); q['year']=pd.to_datetime(q[DATE]).dt.year; q['cohort']=np.where(q[ID].astype(str).isin(set(tr[ID].astype(str))),'shared','new'); q['sq_clip']=p
 q.to_csv(R/f'hgb_robust_seed{seed}_predictions.csv',index=False); print('wrote',len(q),'seconds',round(time.time()-t0,1))
if __name__=='__main__': main()

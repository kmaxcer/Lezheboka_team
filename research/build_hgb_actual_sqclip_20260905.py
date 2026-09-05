"""Fit the robust pseudo-gap HGB once for the real synthetic gaps.

This is a research artifact. It uses only train labels and visible private
labels; organiser gaps remain query-only. The feature recipe matches the
exact-mask validation in ``validate_hgb_exact_mask_20260905.py``.
"""
from pathlib import Path
import sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); R=ROOT/'research'; ARCH=ROOT/'_archive_inspect'/'agropulse_max_score'/'src'; sys.path.insert(0,str(ARCH)); sys.path.insert(0,str(R))
from agropulse.pipeline import build_features
from feature_hgb_v2 import _clear
from feature_hgb_v3 import extra_features_v3
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'

def matrix(d,obs,mask):
    fr=_clear(d,mask); bx=build_features(fr,obs,pd.Series(mask,index=fr.index)); ex=extra_features_v3(fr,obs,mask)
    return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)

def main(n_masks=3):
    t0=time.time(); tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
    tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool); tr['_origin']='train'; pr['_origin']='private'
    ref=pd.concat([tr,pr],ignore_index=True,sort=False); ref[DATE]=pd.to_datetime(ref[DATE]); ref['year']=ref['year'].fillna(ref[DATE].dt.year).astype(int); ref['doy']=ref['doy'].fillna(ref[DATE].dt.dayofyear).astype(int); ref['_truth']=pd.to_numeric(ref[TARGET],errors='coerce')
    gaps=np.r_[np.zeros(len(tr),bool),pr[GAP].to_numpy(bool)]; ref.loc[gaps,TARGET]=np.nan; known=ref['_truth'].notna().to_numpy(bool)&~gaps; tab=pd.DataFrame({'id':ref[ID].astype(str),'year':ref[DATE].dt.year.to_numpy(int)})
    blocks=[]; ys=[]
    for rep in range(n_masks):
        rng=np.random.default_rng(20261000+rep); pm=np.zeros(len(ref),bool)
        for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
            ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
        comb=gaps|pm; obs=ref[TARGET].where(~comb); print('features',rep,pm.sum(),flush=True); x=matrix(ref,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(ref.loc[pm,'_truth'].reset_index(drop=True))
    obs=ref[TARGET].where(~gaps); print('query features',int(gaps.sum()),flush=True); qx=matrix(ref,obs,gaps).loc[gaps].reset_index(drop=True); X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float); print('fit',X.shape,flush=True)
    m=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.,random_state=42); m.fit(X,np.clip(y,-.10,1.0)); p=np.clip(m.predict(qx),-.2,1.1)
    q=pr.loc[pr[GAP],[ID,DATE]].copy().reset_index(drop=True); q['hgb_sq_clip']=p; q.to_csv(R/'hgb_actual_sqclip_predictions_20260905.csv',index=False,float_format='%.10f'); print('wrote',len(q),'seconds',round(time.time()-t0,1),flush=True)

if __name__=='__main__': main()

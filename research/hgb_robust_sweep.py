from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
ARCH=ROOT/'_archive_inspect'/'agropulse_max_score'/'src'; R=ROOT/'research'
sys.path.insert(0,str(ARCH)); sys.path.insert(0,str(R))
from agropulse.pipeline import build_features
from feature_hgb_v2 import _clear
from feature_hgb_v3 import extra_features_v3
ID='anon_polygon_id'; DATE='date'; TARGET='primary_ndvi'; GAP='is_synthetic_gap'

def make_holdout(pr,seed):
    known=pr[TARGET].notna().to_numpy(bool)&~pr[GAP].fillna(False).to_numpy(bool); out=np.zeros(len(pr),bool)
    rng=np.random.default_rng(seed); yy=pd.to_datetime(pr[DATE]).dt.year
    for _,ix0 in pr.loc[known].groupby([ID,yy],sort=False).groups.items():
        ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.15*len(ix)))); out[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
    return out

def matrix(d,obs,mask):
    fr=_clear(d,mask); bx=build_features(fr,obs,pd.Series(mask,index=fr.index)); ex=extra_features_v3(fr,obs,mask)
    return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)

def run(seed,n_masks):
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
    tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool)
    hold=make_holdout(pr,seed); gaps_pr=pr[GAP].to_numpy(bool)|hold
    tr2=tr.copy(); p2=pr.copy(); tr2['_origin']='train'; p2['_origin']='private'; ref=pd.concat([tr2,p2],ignore_index=True,sort=False)
    ref[DATE]=pd.to_datetime(ref[DATE]); ref['year']=ref['year'].fillna(ref[DATE].dt.year).astype(int); ref['doy']=ref['doy'].fillna(ref[DATE].dt.dayofyear).astype(int); ref['_truth']=pd.to_numeric(ref[TARGET],errors='coerce')
    hk=set(map(tuple,pr.loc[gaps_pr,[ID,DATE]].to_numpy())); gaps=np.array([tuple(x) in hk for x in ref[[ID,DATE]].to_numpy()],bool); ref.loc[gaps,TARGET]=np.nan
    known=ref['_truth'].notna().to_numpy(bool)&~gaps; years=ref[DATE].dt.year.to_numpy(int); tab=pd.DataFrame({'id':ref[ID].astype(str),'year':years})
    blocks=[]; ys=[]; t0=time.time()
    for rep in range(n_masks):
        rng=np.random.default_rng(202600+seed*10+rep); pm=np.zeros(len(ref),bool)
        for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
            ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
        comb=gaps|pm; obs=ref[TARGET].where(~comb); print('features',seed,rep,pm.sum(),flush=True); x=matrix(ref,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(ref.loc[pm,'_truth'].reset_index(drop=True))
    obs=ref[TARGET].where(~gaps); print('query features',seed,gaps.sum(),flush=True); qx=matrix(ref,obs,gaps).loc[gaps].reset_index(drop=True)
    X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float); print('fit matrix',X.shape,'secs',round(time.time()-t0,1),flush=True)
    specs={
      'sq_regular':dict(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.),
      'sq_wide':dict(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=8.),
      'abs_regular':dict(loss='absolute_error',learning_rate=.03,max_iter=300,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.),
      'sq_clip':dict(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.),
    }
    preds={}; rows=[]; truth=ref.loc[gaps,'_truth'].to_numpy(float); # actual truth sidecar
    for name,sp in specs.items():
        yy=y.copy();
        if name=='sq_clip': yy=np.clip(yy,-.10,1.0)
        m=HistGradientBoostingRegressor(random_state=42,**sp); m.fit(X,yy); p=m.predict(qx)
        for clip in [(-.2,1.1),(-.05,1.0),(-.1,1.0)]:
            pp=np.clip(p,*clip); ok=np.isfinite(truth); rm=float(np.sqrt(np.mean((pp[ok]-truth[ok])**2))); mae=float(np.mean(np.abs(pp[ok]-truth[ok]))); rows.append({'seed':seed,'method':name+f'_predclip{clip[0]}_{clip[1]}','n':int(ok.sum()),'rmse':rm,'mae':mae})
        preds[name]=np.clip(p,-.2,1.1)
    # cohort metrics
    q=ref.loc[gaps,[ID,DATE,'_truth']].copy().reset_index(drop=True); q['year']=pd.to_datetime(q[DATE]).dt.year; q['cohort']=np.where(q[ID].astype(str).isin(set(tr[ID].astype(str))),'shared','new')
    for k,p in preds.items(): q[k]=p
    for gn,g in [('all',q),('history',q[q.year<2025]),('2025',q[q.year==2025]),('new2025',q[(q.cohort=='new')&(q.year==2025)]),('shared2025',q[(q.cohort=='shared')&(q.year==2025)])]:
      for k in preds:
        t=g['_truth'].to_numpy(float); p=g[k].to_numpy(float); ok=np.isfinite(t); rows.append({'seed':seed,'cohort':gn,'method':k,'n':int(ok.sum()),'rmse':float(np.sqrt(np.mean((p[ok]-t[ok])**2))),'mae':float(np.mean(np.abs(p[ok]-t[ok])))})
    q.to_csv(R/f'hgb_robust_seed{seed}_predictions.csv',index=False)
    return pd.DataFrame(rows), {'seed':seed,'n_masks':n_masks,'features':int(X.shape[1]),'train_rows':int(len(X)),'seconds':round(time.time()-t0,1)}

def main():
  allr=[]; meta=[]
  for seed,nm in [(0,2),(1,2),(70404,3)]:
    r,m=run(seed,nm); allr.append(r); meta.append(m); print(r.to_string(index=False),flush=True)
  out=pd.concat(allr,ignore_index=True); out.to_csv(R/'hgb_robust_sweep_results.csv',index=False); (R/'hgb_robust_sweep_meta.json').write_text(json.dumps(meta,indent=2))
  print(out[(out.method.str.contains('predclip|regular')) & (out.get('cohort','all')=='all')].to_string(index=False),flush=True)
if __name__=='__main__': main()

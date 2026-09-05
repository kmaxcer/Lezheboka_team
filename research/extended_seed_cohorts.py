"""Compact random-seed/cohort audit for extended HGB.

This is deliberately one model per seed (regular HGB) with disjoint OOF
pseudo-gaps.  It writes row-level predictions so shared/new AOI and 2025
cohorts can be inspected without rerunning feature construction.
"""
from __future__ import annotations
import sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); R=ROOT/'research'
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold
sys.path.insert(0,str(R)); from build_extended_hgb_private import _clear,_fit,_matrix
TARGET='primary_ndvi'

def masks(d,exclude,nfold=5,seed=1):
    known=d[TARGET].notna().to_numpy(bool)&~np.asarray(exclude,bool); out=[np.zeros(len(d),bool) for _ in range(nfold)]; rng=np.random.default_rng(seed); tab=pd.DataFrame({'id':d.anon_polygon_id.astype(str),'yr':pd.to_datetime(d.date).dt.year})
    for _,ix0 in tab.loc[known].groupby(['id','yr'],sort=False).groups.items():
        ix=np.asarray(ix0,int); rng.shuffle(ix)
        for j,ii in enumerate(ix): out[j%nfold][ii]=True
    return out

def evaluate(d,query,exclude,seed,label):
    d=d.copy().reset_index(drop=True); d.date=pd.to_datetime(d.date); d.year=d.year.fillna(d.date.dt.year).astype(int); d.doy=d.doy.fillna(d.date.dt.dayofyear).astype(int)
    if '_truth' not in d: d['_truth']=pd.to_numeric(d[TARGET],errors='coerce')
    blocks=[]; ys=[]; t=time.time()
    for j,pm in enumerate(masks(d,exclude,5,seed+1000),1):
        comb=np.asarray(exclude,bool)|pm; fr=_clear(d,comb); obs=fr[TARGET].where(~comb); x=_matrix(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True)); print(label,'block',j,int(pm.sum()),'elapsed',round(time.time()-t,1),flush=True)
    vf=_clear(d,exclude); obs=vf[TARGET].where(~exclude); qx=_matrix(d,obs,exclude).loc[query].reset_index(drop=True); xa=pd.concat(blocks,ignore_index=True); ya=pd.concat(ys,ignore_index=True).astype(float); y=d.loc[query,'_truth'].to_numpy(float)
    m=_fit('regular',xa,ya,42); p=np.clip(m.predict(qx),-.2,1.1); e=p-y; met={'protocol':label,'seed':seed,'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'train_n':len(xa),'features':xa.shape[1]}; pred=pd.DataFrame({'protocol':label,'seed':seed,'anon_polygon_id':d.loc[query,'anon_polygon_id'].to_numpy(),'date':d.loc[query,'date'].to_numpy(),'truth':y,'pred':p}); return met,pred

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False); rows=[]; preds=[]
    # Three independent random masks over known train rows.
    for seed in (0,1,2):
        d=tr.copy(); d['is_synthetic_gap']=False; d['_truth']=d[TARGET].astype(float); hold=np.zeros(len(d),bool); rng=np.random.default_rng(70000+seed)
        for _,ix0 in d.groupby(['anon_polygon_id',d.date.dt.year],sort=False).groups.items():
            ix=np.asarray(ix0,int); ix=ix[d.loc[ix,TARGET].notna().to_numpy()]
            if len(ix): hold[rng.choice(ix,size=min(len(ix),max(1,int(round(.15*len(ix))))),replace=False)]=True
        met,p=evaluate(d,hold,hold,seed,f'random{seed}'); rows.append(met); preds.append(p)
    # 2025 proxy with actual hidden rows excluded; retain cohort labels.
    tr.is_synthetic_gap=False; pr.is_synthetic_gap=pr.is_synthetic_gap.fillna(False).astype(bool); d=pd.concat([tr,pr],ignore_index=True,sort=False); d.date=pd.to_datetime(d.date); d.year=d.year.fillna(d.date.dt.year).astype(int); d.doy=d.doy.fillna(d.date.dt.dayofyear).astype(int); d['_truth']=d[TARGET].astype(float); hid=d.is_synthetic_gap.to_numpy(bool); target=(d.date.dt.year.eq(2025)&d[TARGET].notna()&~hid).to_numpy(); hold=np.zeros(len(d),bool); rng=np.random.default_rng(70303)
    for _,ix0 in d.loc[target].groupby('anon_polygon_id',sort=False).groups.items():
        ix=np.asarray(ix0,int); hold[rng.choice(ix,size=min(len(ix),max(1,int(round(.30*len(ix))))),replace=False)]=True
    met,p=evaluate(d,hold,hid|hold,3,'proxy2025'); rows.append(met); train_ids=set(tr.anon_polygon_id.astype(str)); p['cohort']=np.where(p.anon_polygon_id.astype(str).isin(train_ids),'shared','new'); preds.append(p)
    out=pd.DataFrame(rows); pred=pd.concat(preds,ignore_index=True); out.to_csv(R/'extended_seed_cohorts_results.csv',index=False); pred.to_csv(R/'extended_seed_cohorts_predictions.csv',index=False)
    # Cohort summary is only defined for the 2025 proxy rows.
    c=pred[pred.protocol=='proxy2025'].copy(); y=c.truth.to_numpy(); csum=c.groupby('cohort',as_index=False).apply(lambda g:pd.Series({'n':len(g),'rmse':float(np.sqrt(np.mean((g.pred.to_numpy()-g.truth.to_numpy())**2))),'mae':float(np.mean(abs(g.pred.to_numpy()-g.truth.to_numpy())))}),include_groups=False).reset_index(drop=True); csum.to_csv(R/'extended_seed_cohorts_aggregate.csv',index=False); print(out.to_string(index=False)); print(csum.to_string(index=False),flush=True)
if __name__=='__main__': main()

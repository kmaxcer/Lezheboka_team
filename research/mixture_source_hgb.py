"""Mixture-of-experts HGB conditioned on the latent acquisition sensor.

Research-only.  Query rows have all sensors blank, but source probabilities
can be estimated from the visible AOI/date acquisition schedule.  Separate
experts test whether pooling the three sensor populations is hurting RMSE.
"""
from __future__ import annotations
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold
sys.path.insert(0,str(ROOT/'_archive_inspect'/'agropulse_max_score'/'src'))
from agropulse.pipeline import build_features,FULL_FEATURES
sys.path.insert(0,str(ROOT/'research')); from feature_hgb_v2 import extra_features,_clear,_source

SENS=['s2_ndvi','landsat_ndvi','modis_ndvi']; TARGET='primary_ndvi'

def model(kind='regular',seed=42):
    spec={'regular':dict(learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=45,l2_regularization=12.),'default':dict(learning_rate=.035,max_iter=300,max_leaf_nodes=48,min_samples_leaf=35,l2_regularization=8.)}[kind]
    return HistGradientBoostingRegressor(loss='squared_error',random_state=seed,**spec)

def build_x(frame, mask):
    fr=_clear(frame,mask); obs=fr[TARGET].where(~mask)
    bx=build_features(fr,obs,pd.Series(mask,index=fr.index)); ex=extra_features(fr,obs,mask)
    return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1)

def run_fold(frame, qmask, truth, seed=42):
    # one OOF pseudo-mask per year; source labels are retained only as train
    # supervision and never exposed as query features.
    orig=_source(frame); known=frame[TARGET].notna().to_numpy(bool)&~qmask
    rng=np.random.default_rng(seed+17); pm=np.zeros(len(frame),bool)
    years=pd.to_datetime(frame.date).dt.year
    pool=known & ~qmask
    for _,ix0 in frame.loc[pool].groupby(['anon_polygon_id',years],sort=False).groups.items():
        ii=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ii)))); pm[rng.choice(ii,size=min(n,len(ii)),replace=False)]=True
    # mask outer+inner to prevent outer context leaking into training features
    Xtr=build_x(frame,qmask|pm); Xq=Xtr.loc[qmask]
    ytr=frame.loc[pm,TARGET].to_numpy(float); src_tr=orig[pm]
    # Fit pooled and source experts.  Experts fall back to pooled if too few.
    pooled=model('regular',seed)
    cols=list(Xtr.columns); pooled.fit(Xtr.loc[pm,cols],ytr)
    experts=[]
    for s in range(3):
        mm=pm & (orig==s)
        if mm.sum()<80: experts.append(None); continue
        m=model('regular',seed+s+1);m.fit(Xtr.loc[mm,cols],frame.loc[mm,TARGET].to_numpy(float));experts.append(m)
    pp=Xq[[c for c in Xq.columns if c.startswith('query_p_')]].to_numpy(float)
    ep=np.column_stack([(m.predict(Xq[cols]) if m is not None else pooled.predict(Xq[cols])) for m in experts])
    # clean posterior and alternatives
    pp=np.where(np.isfinite(pp),pp,1/3); pp=np.clip(pp,0,1); pp/=pp.sum(axis=1,keepdims=True)
    pred_soft=np.sum(pp*ep,axis=1); pred_mode=ep[np.arange(len(ep)),np.argmax(pp,axis=1)]
    # Blend pooled/expert conservatively; source posterior can be noisy.
    pred_mix=.5*pooled.predict(Xq[cols])+.5*pred_soft
    return {'pooled':pooled.predict(Xq[cols]),'soft':pred_soft,'mode':pred_mode,'mix':pred_mix,'src':orig[qmask]}

def main():
    warnings.filterwarnings('ignore')
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rows=[]
    for yr in [2019,2020,2021,2022,2023,2024]:
        fold,truth=make_fold(tr.copy(),pr.copy(),yr);q=fold.is_synthetic_gap.fillna(False).to_numpy(bool); y=truth.to_numpy(float)
        out=run_fold(fold,q,truth,42)
        for name in ['pooled','soft','mode','mix']:
            e=out[name]-y; rows.append({'year':yr,'method':name,'n':len(y),'rmse':np.sqrt(np.mean(e*e)),'mae':np.mean(abs(e))})
        print('done',yr,flush=True)
    o=pd.DataFrame(rows);o.to_csv(ROOT/'research/mixture_source_hgb_results.csv',index=False)
    def f(g): return pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n)),'mae':np.average(g.mae,weights=g.n)})
    a=o.groupby('method',as_index=False).apply(f,include_groups=False).reset_index(drop=True).sort_values('rmse');a.to_csv(ROOT/'research/mixture_source_hgb_aggregate.csv',index=False);print(a.to_string(index=False))

if __name__=='__main__':main()

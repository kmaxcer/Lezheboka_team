"""Cross-fitted source-posterior residual corrections for the ensemble.

Source labels are evaluation sidecars only.  The classifier sees the same
observable acquisition schedule available at inference; correction parameters
for each partition are fitted on other partitions.
"""
from __future__ import annotations
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); R=ROOT/'research'
sys.path.insert(0,str(ROOT/'src'));from validate import make_fold
sys.path.insert(0,str(R));from source_classifier_eval import source,features
sys.path.insert(0,str(R));from teammate_sweep_postcorr import _mask_private

def rf_post(frame,q):
    lab=source(frame);known=frame.primary_ndvi.notna().to_numpy(bool)&~q;X=features(frame,known,lab);tr=known&(lab>=0)
    m=RandomForestClassifier(n_estimators=300,min_samples_leaf=8,max_features=.7,n_jobs=4,random_state=42,class_weight='balanced_subsample');m.fit(X.loc[tr],lab[tr]);
    # ``q`` is a boolean mask over the complete frame; predict_proba returns
    # one row per queried item (``q.sum()``), not one row per frame row.
    p=m.predict_proba(X.loc[q]); full=np.zeros((int(np.asarray(q, dtype=bool).sum()),3));
    for j,c in enumerate(m.classes_.astype(int)):full[:,c]=p[:,j]
    return full

def load_rows(tr,pr):
    rows=[]
    # Exact hidden-DOY partitions.
    src_tr=source(tr)
    for yr in [2019,2020,2021,2022,2023,2024]:
        f,t=make_fold(tr.copy(),pr.copy(),yr);f=f.reset_index(drop=True);q=f.is_synthetic_gap.fillna(False).to_numpy(bool)
        pp=rf_post(f,q); keys=f.loc[q,['anon_polygon_id','date']].copy();keys['date']=pd.to_datetime(keys.date);keys['partition']=f'exact{yr}';keys['year']=yr;keys['doy']=keys.date.dt.dayofyear.to_numpy();keys['_truth']=t.to_numpy(float);keys['_src']=src_tr[q];keys['p0']=pp[:,0];keys['p1']=pp[:,1];keys['p2']=pp[:,2];rows.append(keys)
        print('exact',yr,flush=True)
    # Random private-like partitions, matching teammate sweep masks.
    src_pr=source(pr)
    for seed in [0,1,2]:
        f,q=_mask_private(pr.copy(),seed);f=f.reset_index(drop=True);q=np.asarray(q,bool);pp=rf_post(f,q)
        keys=f.loc[q,['anon_polygon_id','date']].copy();keys['date']=pd.to_datetime(keys.date);keys['partition']=f'random{seed}';keys['year']=keys.date.dt.year;keys['doy']=keys.date.dt.dayofyear.to_numpy();keys['_truth']=f.loc[q,'_truth'].to_numpy(float);keys['_src']=src_pr[q];keys['p0']=pp[:,0];keys['p1']=pp[:,1];keys['p2']=pp[:,2];rows.append(keys);print('random',seed,flush=True)
    z=pd.concat(rows,ignore_index=True)
    base=pd.read_csv(R/'teammate_sweep_postcorr_preds.csv',parse_dates=['date'],low_memory=False);base=base[base.method.eq('blend_lag_0.20')][['partition','anon_polygon_id','date','pred']].copy();base.date=pd.to_datetime(base.date)
    z=z.merge(base,on=['partition','anon_polygon_id','date'],how='left',validate='one_to_one');
    if z.pred.isna().any(): raise RuntimeError('missing baseline rows')
    z['resid']=z['_truth']-z.pred;z['doybin']=(z.doy//16).astype(int);z['canon']=z.doy.isin([97,113,129,145,161,177,193,209,225,241,257,273,289])
    return z

def fit_apply(cal,test,kind,scale=1.,clip=.04):
    # Fit source correction on calibration partitions only.
    if kind=='global':
        c=cal.groupby('_src').resid.mean().reindex([0,1,2]).fillna(0).to_numpy()
        corr=test[['p0','p1','p2']].to_numpy()@c
    elif kind=='median':
        c=cal.groupby('_src').resid.median().reindex([0,1,2]).fillna(0).to_numpy();corr=test[['p0','p1','p2']].to_numpy()@c
    elif kind=='doy':
        # source x DOY-bin means, shrink toward source global according to n
        glob=cal.groupby('_src').resid.mean().reindex([0,1,2]).fillna(0).to_numpy(); tab=cal.groupby(['doybin','_src']).resid.agg(['mean','count'])
        corr=np.zeros(len(test)); pp=test[['p0','p1','p2']].to_numpy()
        for i,r in test.reset_index(drop=True).iterrows():
            c=glob.copy()
            for s in range(3):
                try:
                    v,n=tab.loc[(r.doybin,s),['mean','count']]; sh=min(1.,float(n)/80.);c[s]=(1-sh)*glob[s]+sh*float(v)
                except KeyError: pass
            corr[i]=pp[i]@c
    elif kind=='canon':
        glob=cal.groupby('_src').resid.mean().reindex([0,1,2]).fillna(0).to_numpy();tab=cal.groupby(['canon','_src']).resid.mean()
        pp=test[['p0','p1','p2']].to_numpy();corr=[]
        for i,r in test.reset_index(drop=True).iterrows():
            c=glob.copy()
            for s in range(3):
                try:c[s]=float(tab.loc[(r.canon,s)])
                except KeyError:pass
            corr.append(pp[i]@c)
        corr=np.asarray(corr)
    elif kind=='mode':
        c=cal.groupby('_src').resid.mean().reindex([0,1,2]).fillna(0).to_numpy();pp=test[['p0','p1','p2']].to_numpy();corr=c[pp.argmax(1)]
    else: raise ValueError(kind)
    corr=np.clip(corr*scale,-clip,clip);return test.pred.to_numpy()+corr,corr

def main():
    warnings.filterwarnings('ignore');tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False);z=load_rows(tr,pr);z.to_csv(R/'source_correction_rows.csv',index=False)
    out=[]
    for protocol,parts in [('exact',[f'exact{y}' for y in [2019,2020,2021,2022,2023,2024]]),('random',[f'random{s}' for s in [0,1,2]])]:
      for kind in ['global','median','doy','canon','mode']:
       for scale in [.25,.5,.75,1.0]:
        for part in parts:
         test=z[z.partition.eq(part)];cal=z[(z.partition.str.startswith('exact') if protocol=='exact' else z.partition.str.startswith('random')) & ~z.partition.eq(part)]
         p,c=fit_apply(cal,test,kind,scale);e=p-test._truth.to_numpy();out.append({'protocol':protocol,'partition':part,'kind':kind,'scale':scale,'n':len(test),'rmse':np.sqrt(np.mean(e*e)),'mae':np.mean(abs(e)),'corr_abs':np.mean(abs(c))})
    o=pd.DataFrame(out);o.to_csv(R/'source_correction_results.csv',index=False)
    def agg(g):return pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n)),'mae':np.average(g.mae,weights=g.n)})
    a=o.groupby(['protocol','kind','scale'],as_index=False).apply(agg,include_groups=False).reset_index(drop=True).sort_values(['protocol','rmse']);a.to_csv(R/'source_correction_aggregate.csv',index=False);print(a.groupby('protocol').head(15).to_string(index=False))

if __name__=='__main__':main()

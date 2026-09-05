"""Measure cross-AOI same-date predictors on pseudo-private folds."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private, _prepare, _fit_source_maps  # noqa
from validate import make_fold  # noqa

def one(train,private,year):
    fold,truth=make_fold(train,private,year); d=_prepare(fold); syn=fold.is_synthetic_gap.astype(bool).to_numpy(); known=np.isfinite(d.primary_ndvi.to_numpy(float)); y=d.primary_ndvi.to_numpy(float)
    out=predict_private(fold); p=out.primary_ndvi_pred.to_numpy(float); qidx=np.flatnonzero(syn)
    z=pd.DataFrame({'date':d.date,'pid':d.anon_polygon_id,'doy':d._doy,'y':y,'known':known})
    # exact-date cross-AOI summaries, excluding hidden queries by known filter
    g=z[known].groupby('date').y.agg(['mean','median','count','std']);
    # global date means grouped by DOY, useful when exact date has no peers
    gd=z[known].groupby('doy').y.agg(['mean','median','count'])
    # AOI means / seasonal bin means
    z['db']=(z.doy//30).astype(int)
    ab=z[known].groupby(['pid','db']).y.median(); am=z[known].groupby('pid').y.median();
    rows=[]
    for n,q in enumerate(qidx):
        date=d.date.iat[q]; doy=int(d._doy.iat[q]); pid=d.anon_polygon_id.iat[q]; db=int(d._doy.iat[q]//30)
        dm=g['mean'].get(date,np.nan); dmed=g['median'].get(date,np.nan); gm=gd['mean'].get(doy,np.nan)
        base=ab.get((pid,db),am.get(pid,np.nan));
        # date shock relative to same seasonal bin across all AOIs
        gdb=z[known].groupby('db').y.median().get(db,np.nan)
        shock=dm-gdb if np.isfinite(dm) and np.isfinite(gdb) else 0.0
        rows.append({'p':p[n],'truth':truth.iloc[n],'date_mean':dm,'date_med':dmed,'doy_mean':gm,'base':base,'shock':shock,'n':g['count'].get(date,0)})
    return pd.DataFrame(rows)

def main():
    b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']); pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']); allr=[]
    for y in [2019,2020,2021,2022,2023,2024]:
        r=one(tr,pr,y);r['year']=y;allr.append(r)
    d=pd.concat(allr,ignore_index=True); y=d.truth.to_numpy();
    print('n',len(d),'baseline',np.sqrt(np.mean((d.p-y)**2)))
    for c in ['date_mean','date_med','doy_mean','base','shock']:
        x=d[c].to_numpy(float); q=np.isfinite(x); print(c,'avail',q.mean(),'rmse direct',np.sqrt(np.mean((x[q]-y[q])**2)) if q.any() else np.nan,'corr err',np.corrcoef((d.p-y)[q],(x-d.p)[q])[0,1] if q.sum()>2 else np.nan)
        # optimize p + a*(x-p)
        e=(d.p-y)[q]; h=(x-d.p)[q]; a=-np.dot(e,h)/np.dot(h,h) if np.dot(h,h)>0 else 0; pred=d.p.to_numpy(float)[q]+a*h; print('  optimal blend a',a,'rmse',np.sqrt(np.mean((pred-y[q])**2)))
    for c in ['shock']:
      for a in [0.05,.1,.15,.2,.25,.3,.4,.5, .75,1]:
       pred=d.p+a*d[c].fillna(0); print(c,a,np.sqrt(np.mean((pred-y)**2)))
    d.to_csv(ROOT/'research'/'date_features_agent2.csv',index=False)
if __name__=='__main__':main()

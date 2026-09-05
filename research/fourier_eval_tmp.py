from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
ID='anon_polygon_id'; T='primary_ndvi'; G='is_synthetic_gap'

def fourier_x(doy,k,period=366.):
    doy=np.asarray(doy,float); cols=[np.ones(len(doy))]
    # center phase around growing-season midpoint; ordinary annual basis
    for j in range(1,k+1):
        z=2*np.pi*j*doy/period; cols.extend([np.sin(z),np.cos(z)])
    return np.column_stack(cols)

def fit_group(x,y,kind='ridge',k=5,alpha=10.,clip=3.0):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<max(5,2*k+1): return None
    xx=fourier_x(x[ok],k); yy=y[ok]
    # iterative robust reweighting around a preliminary smooth fit
    if kind=='huber':
        m=HuberRegressor(alpha=alpha,epsilon=clip,max_iter=300).fit(xx,yy)
    else:
        m=Ridge(alpha=alpha).fit(xx,yy)
    return m

def predict_curve(frame,obs,query_mask, k=5,alpha=10.,kind='ridge',pool='none',clip=3.0):
    d=frame.copy().reset_index(drop=True); obs=np.asarray(obs,float).copy(); qm=np.asarray(query_mask,bool)
    dates=pd.to_datetime(d.date); doy=dates.dt.dayofyear.to_numpy(); years=dates.dt.year.to_numpy(); ids=d[ID].astype(str).to_numpy()
    pred=np.full(len(d),np.nan)
    # primary group, then fallbacks
    # Fit once per group (the previous disposable version fitted once per
    # query and was unnecessarily slow).
    qkeys=set(zip(ids[qm],years[qm]))
    for aid,yr in qkeys:
        q=(qm&(ids==aid)&(years==yr)); ix=np.flatnonzero((ids==aid)&(years==yr)&np.isfinite(obs))
        if len(ix)<max(8,2*k+2) and pool in ('aoi','aoi_crop'):
          ix=np.flatnonzero((ids==aid)&np.isfinite(obs))
        if len(ix)<max(8,2*k+2): continue
        m=fit_group(doy[ix],obs[ix],kind,k,alpha,clip)
        if m is not None: pred[q]=m.predict(fourier_x(doy[q],k))
    return pred

def make_hold(pr,seed=70404):
  pr=pr.copy(); pr[G]=pr[G].fillna(False).astype(bool); yr=pr.date.dt.year.to_numpy(); known=pr[T].notna().to_numpy()&~pr[G].to_numpy(); hold=np.zeros(len(pr),bool); rng=np.random.default_rng(seed)
  for _,ix0 in pr.assign(_yr=yr).loc[known].groupby([ID,'_yr'],sort=False).groups.items():
    ix=np.asarray(ix0,int); n=max(1,int(round(.15*len(ix)))); hold[rng.choice(ix,size=min(n,len(ix)),replace=False)]=1
  return hold

def score(y,p):
 ok=np.isfinite(y)&np.isfinite(p); return np.sqrt(np.mean((p[ok]-y[ok])**2)),np.mean(abs(p[ok]-y[ok]))

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date']); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date']); trainids=set(tr[ID])
 for seed in [70404,0,1,2]:
  if seed==70404: hold=make_hold(pr,seed)
  else: hold=make_hold(pr,seed)
  actual=pr[G].to_numpy(bool); qm=hold; mask=actual|hold; obs=pr[T].to_numpy(float).copy(); obs[mask]=np.nan; y=pr.loc[hold,T].to_numpy(float)
  ids=pr.loc[hold,ID].astype(str).to_numpy(); yrs=pr.loc[hold,'date'].dt.year.to_numpy()
  print('SEED',seed,'n',len(y))
  for kind in ['ridge','huber']:
   for k in [1,2,3,4,5,6,8,10,12]:
    for a in ([.1,1,10,50,200] if k in [3,5,8] else [10]):
     p=predict_curve(pr,obs,hold,k,a,kind,'none')
     q=p[hold]; r,ma=score(y,q); print(kind,k,a,round(r,5),round(ma,5),int(np.isfinite(q).sum()))
  # baseline climatology/profile fallback not included
  break

if __name__=='__main__': main()

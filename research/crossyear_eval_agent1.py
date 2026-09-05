"""Quick cross-year seasonal analogue experiment on private 2025 holdout."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from infer import SOURCES,_fit_source_maps,_mode_posteriors,_prepare,_query_posterior
from infer_lag import _lagged_local_poly,DEFAULT_LAGS
from private2025_holdout_agent2 import holdout

def seasonal_poly(qd, kk, doy, y, src, target, maps, qbin, k=16, degree=3):
    """Cross-year local polynomial on circular DOY, with source map."""
    if len(kk)==0:return np.nan
    # duplicate around year boundary for circular nearest-neighbour search
    xx=[]; yy=[]
    for j in kk:
        s=str(src[j]); a,b=maps.get((target,s,qbin),maps.get((target,s,'g'),(0.,1.)))
        xx.append(float(doy[j])); yy.append(float(a)+float(b)*float(y[j]))
    xx=np.asarray(xx); yy=np.asarray(yy)
    # choose nearest by circular distance and unwrap around qd
    dd=((xx-float(qd)+182.5)%365)-182.5
    sel=np.argsort(np.abs(dd))[:min(k,len(dd))]
    zxx=dd[sel]; zyy=yy[sel]; d=np.abs(zxx)
    good=np.isfinite(zxx)&np.isfinite(zyy); zxx,zyy,d=zxx[good],zyy[good],d[good]
    if len(zyy)==0:return np.nan
    if len(zyy)<=degree:return float(zyy[np.argmin(d)])
    scale=max(1.,float(np.max(d))); z=zxx/scale; w=1/(1+2*np.abs(z))
    try:v=float(np.polynomial.polynomial.polyfit(z,zyy,min(degree,len(zyy)-1),w=w)[0])
    except Exception:v=float(np.average(zyy,weights=w))
    lo,hi=np.quantile(zyy,[.05,.95]); return float(np.clip(v,lo-.04,hi+.04))

def predict_season(frame,k=16,degree=3,blend=1.0):
    df=_prepare(frame); syn=frame.is_synthetic_gap.astype(bool).to_numpy(); y=df.primary_ndvi.to_numpy(float); known=np.isfinite(y)
    src=df._src.to_numpy(object); doy=df._doy.to_numpy(int); maps=_fit_source_maps(df,known,bin_days=30)
    aoi,crop,glob,date=_mode_posteriors(df,known); pred=np.full(len(df),np.nan)
    groups=df.groupby('anon_polygon_id',sort=False).groups
    for _,idx in groups.items():
      ii=np.asarray(idx,dtype=int); kk=ii[known[ii]]
      for q in ii[syn[ii]]:
        p=_query_posterior(df,int(q),aoi,crop,glob,date)
        vals=[]
        for s,w in zip(SOURCES,p):
          v=seasonal_poly(doy[q],kk,doy,y,src,s,maps,int(doy[q]//30),k,degree)
          if np.isfinite(v):vals.append((v,float(w)))
        if vals:pred[q]=float(np.average([v for v,_ in vals],weights=[w for _,w in vals]))
    # nearest same-AOI fallback
    ids=df.anon_polygon_id.to_numpy(); x=df._ord.to_numpy(float)
    for q in np.flatnonzero(syn&~np.isfinite(pred)):
      same=np.flatnonzero(known&(ids==ids[q])); pred[q]=y[same[np.argmin(abs(x[same]-x[q]))]] if len(same) else np.nanmedian(y[known])
    return pred[syn]

def main():
  root=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); tr=pd.read_csv(root/'train_dataset.csv',parse_dates=['date']); pr=pd.read_csv(root/'private_features.csv',parse_dates=['date'])
  # evaluate one holdout seed; retain all private history (incl. 2010-24)
  for seed in [0]:
    d,mask=holdout(pr,seed); q=d[mask]; y=q._truth.to_numpy(float)
    for name,fn in [('season',lambda z:predict_season(z,16,3)),('season2',lambda z:predict_season(z,24,2))]:
      out=fn(d); keys=q[['anon_polygon_id','date']].copy(); keys.date=keys.date.dt.strftime('%Y-%m-%d'); pp=pd.DataFrame({'anon_polygon_id':q.anon_polygon_id,'date':q.date.dt.strftime('%Y-%m-%d'),'pred':out}); pp=pp.merge(keys.assign(_i=np.arange(len(q))),on=['anon_polygon_id','date'],how='right').sort_values('_i').pred.to_numpy(); e=pp-y; print(name,len(y),np.sqrt(np.mean(e*e)),np.mean(abs(e)),flush=True)
  
if __name__=='__main__':main()

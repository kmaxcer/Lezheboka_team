from __future__ import annotations
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT=Path(r'C:/Users/kmaxc/PycharmProjects/hack/_1/_lezheboka')
DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
ID='anon_polygon_id'; T='primary_ndvi'; G='is_synthetic_gap'

def make_hold(pr,seed=70404):
 pr=pr.copy(); pr[G]=pr[G].fillna(False).astype(bool); yr=pr.date.dt.year.to_numpy(); known=pr[T].notna().to_numpy()&~pr[G].to_numpy(); hold=np.zeros(len(pr),bool); rng=np.random.default_rng(seed)
 for _,ix0 in pr.assign(_yr=yr).loc[known].groupby([ID,'_yr'],sort=False).groups.items():
  ix=np.asarray(ix0,int); n=max(1,int(round(.15*len(ix)))); hold[rng.choice(ix,size=min(n,len(ix)),replace=False)]=1
 return hold

def weighted_local(xq,x,y,k=12,kind='mean'):
 ok=np.isfinite(x)&np.isfinite(y)
 if not ok.any(): return np.nan
 x=x[ok];y=y[ok]; d=np.abs(x-xq); sel=np.argsort(d)[:min(k,len(d))]; d=d[sel];y=y[sel]
 if kind=='median': return float(np.median(y))
 if kind=='linear' and len(y)>=2:
  z=(x[sel]-xq)/max(1.,d.max()); w=1/(1+2*abs(z)); sw=w.sum(); sz=(w*z).sum(); sy=(w*y).sum(); szz=(w*z*z).sum(); szy=(w*z*y).sum(); den=sw*szz-sz*sz
  if abs(den)>1e-12: return float((sy*szz-sz*szy)/den)
 return float(np.average(y,weights=1/(1+d/3)))

def predict(d,obs,qm,variant):
 d=d.copy().reset_index(drop=True); obs=np.asarray(obs,float); qm=np.asarray(qm,bool); dates=pd.to_datetime(d.date); doy=dates.dt.dayofyear.to_numpy(float); yr=dates.dt.year.to_numpy(int); ids=d[ID].astype(str).to_numpy(); clim=pd.to_numeric(d.ndvi_climatology_mean,errors='coerce').to_numpy(float)
 # query predictions
 out=np.full(len(d),np.nan); known=np.isfinite(obs); res=obs-clim
 qkeys=set(zip(ids[qm],yr[qm]))
 for aid,y0 in qkeys:
  q=(qm&(ids==aid)&(yr==y0)); cur=(ids==aid)&(yr==y0)&known
  hist=(ids==aid)&known&(yr!=y0)
  for i in np.flatnonzero(q):
   dd=doy[i]
   # climatology from nearest visible climatology points in current year, all years fallback
   ccur=cur&np.isfinite(clim); chist=hist&np.isfinite(clim)
   c=weighted_local(dd,doy[ccur],clim[ccur],12,'linear') if ccur.any() else weighted_local(dd,doy[chist],clim[chist],20,'linear')
   # local residual; variants differ in pool and weighting
   if variant.startswith('cur'):
    rr=cur&np.isfinite(res); xx=doy[rr]; yy=res[rr]
   elif variant.startswith('hist'):
    rr=hist&np.isfinite(res); xx=doy[rr]; yy=res[rr]
   else:
    rr=(cur|hist)&np.isfinite(res); xx=doy[rr]; yy=res[rr]
   if variant.endswith('median'): r=weighted_local(dd,xx,yy,12,'median')
   elif variant.endswith('linear'): r=weighted_local(dd,xx,yy,12,'linear')
   else: r=weighted_local(dd,xx,yy,12,'mean')
   # robust global/current offset alternatives
   if not np.isfinite(r): r=np.nanmedian(yy) if len(yy) else 0.
   out[i]=c+r if np.isfinite(c) else weighted_local(dd,doy[cur],obs[cur],12,'linear')
 return out

def score(y,p):
 ok=np.isfinite(y)&np.isfinite(p); return np.sqrt(np.mean((p[ok]-y[ok])**2)),np.mean(abs(p[ok]-y[ok]))

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date']); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date']); trainids=set(tr[ID])
 for seed in [70404,0,1,2]:
  h=make_hold(pr,seed); actual=pr[G].to_numpy(bool); mask=actual|h; obs=pr[T].to_numpy(float).copy();obs[mask]=np.nan;y=pr.loc[h,T].to_numpy(float); print('SEED',seed)
  for v in ['cur_mean','cur_linear','cur_median','hist_mean','hist_linear','hist_median','all_mean','all_linear','all_median']:
   p=predict(pr,obs,h,v)[h]; print(v,score(y,p),int(np.isfinite(p).sum()))
  # cohort quick
  for v in ['cur_linear','all_linear']:
   p=predict(pr,obs,h,v)[h]; z=pr.loc[h,[ID,'date']].copy();z['y']=y;z['p']=p;z['co']=np.where(z[ID].isin(trainids),'shared','new');z['yr']=z.date.dt.year
   print(v,z.groupby(['co','yr']).apply(lambda g:score(g.y,g.p),include_groups=False).to_dict())
  if seed==2: break
if __name__=='__main__':main()

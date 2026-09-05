"""Quick leakage-safe climatology/residual interpolation benchmark."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold

def pred(frame:pd.DataFrame, q:np.ndarray, mode:str)->np.ndarray:
 d=frame.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date);d['_yr']=d.date.dt.year;d['_doy']=d.date.dt.dayofyear
 y=d.primary_ndvi.to_numpy(float);known=np.isfinite(y)&~np.asarray(q,bool);cl=d.ndvi_climatology_mean.to_numpy(float); ords=d.date.map(pd.Timestamp.toordinal).to_numpy(float);out=np.full(q.sum(),np.nan);qi=np.flatnonzero(q);pos={int(i):j for j,i in enumerate(qi)}
 for (aid,yr),ix0 in d.groupby(['anon_polygon_id','_yr'],sort=False).groups.items():
  ix=np.asarray(ix0);kk=ix[known[ix]];qq=ix[q[ix]]
  if not len(qq):continue
  x=ords[kk]; yy=y[kk]; cc=cl[kk];good=np.isfinite(x)&np.isfinite(yy);x=x[good];yy=yy[good];cc=cc[good]
  if not len(x):continue
  order=np.argsort(x);x=x[order];yy=yy[order];cc=cc[order]
  # residual against visible climatology; fallback raw values if absent
  rr=yy-cc; rg=np.isfinite(rr)
  for qi0 in qq:
   j=pos[int(qi0)];xq=ords[qi0];
   n=min(8,len(x));dist=np.abs(x-xq);sel=np.argsort(dist)[:n];
   if mode=='raw_linear':
    p=np.interp(xq,x,yy)
   elif mode=='raw_local':
    z=(x[sel]-xq)/max(1.,dist[sel].max());
    try:p=np.polynomial.polynomial.polyfit(z,yy[sel],1,w=1/(1+2*np.abs(z)))[0]
    except:p=np.average(yy[sel],weights=1/(1+dist[sel]))
   elif mode.startswith('res'):
    # interpolate climatology from all row values in this AOI/year, then add
    # a local residual estimate from known target rows.
    allx=ords[ix];allc=d.ndvi_climatology_mean.to_numpy(float)[ix];ok=np.isfinite(allc);base=np.interp(xq,allx[ok],allc[ok]) if ok.any() else np.nan
    if rg.any():
     xr=x[rg];r=rr[rg];dr=np.abs(xr-xq);ss=np.argsort(dr)[:min(n,len(xr))];zz=(xr[ss]-xq)/max(1.,dr[ss].max());
     if mode=='res_local':
      try:adj=np.polynomial.polynomial.polyfit(zz,r[ss],1,w=1/(1+2*np.abs(zz)))[0]
      except:adj=np.average(r[ss],weights=1/(1+dr[ss]))
     elif mode=='res_mean':adj=np.average(r[ss],weights=1/(1+dr[ss]))
     elif mode=='res_median':adj=np.median(r[ss])
     else:adj=0.
    else:adj=0.
    p=base+adj if np.isfinite(base) else np.interp(xq,x,yy)
   else:p=np.interp(xq,x,yy)
   out[j]=p
 return out

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False);rows=[]
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr.copy(),pr.copy(),yr);q=f.is_synthetic_gap.fillna(False).to_numpy(bool);y=t.to_numpy(float)
  for m in ['raw_linear','raw_local','res_local','res_mean','res_median']:
   p=pred(f,q,m);rows.append({'year':yr,'method':m,'n':len(y),'rmse':np.sqrt(np.nanmean((p-y)**2)),'mae':np.nanmean(abs(p-y))})
 o=pd.DataFrame(rows);o.to_csv(ROOT/'research/clim_res_eval_results.csv',index=False);a=o.groupby('method',as_index=False).apply(lambda g:pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n)),'mae':np.average(g.mae,weights=g.n)}),include_groups=False).reset_index(drop=True);print(o.to_string(index=False));print(a.to_string(index=False))
if __name__=='__main__':main()

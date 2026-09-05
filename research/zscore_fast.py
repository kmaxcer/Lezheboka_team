"""Vectorised-ish z-score interpolation benchmark (kept fast for overnight runs)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private
from validate import make_fold

def local(q,x,y,k,deg):
    good=np.isfinite(x)&np.isfinite(y)
    x=x[good];y=y[good]
    if len(x)==0:return np.nan
    ix=np.argsort(np.abs(x-q))[:min(k,len(x))];xx=x[ix];yy=y[ix]
    if len(yy)<=deg:return float(yy[np.argmin(abs(xx-q))])
    sc=max(1.,float(np.max(abs(xx-q))));zz=(xx-q)/sc;w=1/(1+2*abs(zz))
    try:v=float(np.polynomial.polynomial.polyfit(zz,yy,min(deg,len(yy)-1),w=w)[0])
    except Exception:v=float(np.average(yy,weights=w))
    return v

def one(f,truth, ks=(4,8,16), degs=(0,1,2), cross=False):
    d=f.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date);d['_yr']=d.date.dt.year.to_numpy();d['_doy']=d.date.dt.dayofyear.to_numpy();d['_ord']=d.date.map(pd.Timestamp.toordinal).to_numpy(float)
    syn=d.is_synthetic_gap.to_numpy(bool); y=d.primary_ndvi.to_numpy(float);known=np.isfinite(y)&~syn
    pid=d.anon_polygon_id.to_numpy(object);yr=d._yr.to_numpy(int);doy=d._doy.to_numpy(int);ordv=d._ord.to_numpy(float);mu=d.ndvi_climatology_mean.to_numpy(float);sd=d.ndvi_climatology_std.to_numpy(float)
    z=(y-mu)/sd;goodz=known&np.isfinite(z)&np.isfinite(mu)&np.isfinite(sd)&(sd>1e-4)&(abs(z)<6)
    qs=np.flatnonzero(syn); results={(k,deg):np.full(len(qs),np.nan) for k in ks for deg in degs}
    # pre-index groups to avoid repeated boolean scans
    groups={}
    for i in np.flatnonzero(goodz):groups.setdefault((pid[i],yr[i]),[]).append(i)
    histgroups={}
    for i in np.flatnonzero(goodz):histgroups.setdefault(pid[i],[]).append(i)
    # Baseline indices include observed climatology even if z was unusable.
    bgroups={}
    for i in np.flatnonzero(known&np.isfinite(mu)&np.isfinite(sd)&(sd>1e-4)):bgroups.setdefault((pid[i],yr[i]),[]).append(i)
    bhist={}
    for i in np.flatnonzero(known&np.isfinite(mu)&np.isfinite(sd)&(sd>1e-4)):bhist.setdefault(pid[i],[]).append(i)
    for qi,q in enumerate(qs):
        key=(pid[q],yr[q]); bi=np.asarray(bgroups.get(key,[]),int)
        if len(bi)<3 and cross:bi=np.asarray(bhist.get(pid[q],[]),int)
        if len(bi):
            m=local(doy[q],doy[bi].astype(float),mu[bi],24,2);s=local(doy[q],doy[bi].astype(float),sd[bi],24,1)
        else:m=np.nan;s=np.nan
        ai=np.asarray(groups.get(key,[]),int)
        hi=np.asarray(histgroups.get(pid[q],[]),int) if cross else np.array([],int)
        for k in ks:
          for deg in degs:
            zp=np.nan
            if len(ai):zp=local(ordv[q],ordv[ai],z[ai],k,deg)
            if len(hi) and (not np.isfinite(zp) or len(ai)<3):
                z2=local(doy[q],doy[hi],z[hi],max(k,12),min(2,deg))
                if np.isfinite(z2):zp=z2 if not np.isfinite(zp) else .7*zp+.3*z2
            if np.isfinite(m)&np.isfinite(s)&np.isfinite(zp):results[(k,deg)][qi]=np.clip(m+s*np.clip(zp,-5,5),-0.5,1.2)
    base=predict_private(f,k=8,bin_days=30);keys=d.loc[qs,['anon_polygon_id','date']].copy();keys.date=keys.date.dt.strftime('%Y-%m-%d');bp=keys.merge(base,on=['anon_polygon_id','date'],how='left').primary_ndvi_pred.to_numpy(float)
    for p in results.values():p[~np.isfinite(p)]=bp[~np.isfinite(p)]
    return results,bp,qs

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False);rows=[]
 for year in range(2019,2025):
  f,t=make_fold(tr,pr,year);truth=t.to_numpy(float)
  for cross in [False,True]:
   res,bp,qs=one(f,truth,cross=cross)
   for key,p in res.items():rows.append((year,cross,key[0],key[1],np.sqrt(np.mean((p-truth)**2)),np.mean(abs(p-truth))))
 out=pd.DataFrame(rows,columns=['year','cross','k','degree','rmse','mae']);agg=out.groupby(['cross','k','degree']).apply(lambda z:np.sqrt(np.average(z.rmse**2,weights=np.ones(len(z))))).reset_index(name='rmse').sort_values('rmse');print(agg.head(30).to_string(index=False));out.to_csv(ROOT/'research'/'zscore_fast_results.csv',index=False)
if __name__=='__main__':main()

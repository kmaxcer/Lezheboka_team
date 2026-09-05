"""Scalar Kalman/RTS smoothing experiments for sparse MODIS sequences."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private,_prepare
from validate import make_fold

def smooth_grid(days, vals, qday=0.002, r=0.01):
    """Random-walk Kalman smoother on integer day coordinates."""
    if len(days)==0:return {}
    lo,hi=int(np.min(days)),int(np.max(days)); n=hi-lo+1; obs=np.full(n,np.nan)
    for d,v in zip(days,vals):
      if np.isfinite(v):obs[int(d)-lo]=v
    # forward filter
    m=np.zeros(n);P=np.zeros(n); first=np.flatnonzero(np.isfinite(obs))
    if len(first)==0:return {}
    i0=int(first[0]);m[:i0+1]=obs[i0];P[:i0+1]=1.
    for i in range(i0+1,n):
      mp=m[i-1]; pp=P[i-1]+qday
      if np.isfinite(obs[i]):
       k=pp/(pp+r);m[i]=mp+k*(obs[i]-mp);P[i]=(1-k)*pp
      else:m[i]=mp;P[i]=pp
    # RTS backward; transition is identity, process qday
    sm=m.copy();sp=P.copy()
    for i in range(n-2,i0-1,-1):
      pp=P[i]+qday;g=P[i]/pp if pp>0 else 0.;sm[i]=m[i]+g*(sm[i+1]-m[i]);sp[i]=P[i]+g*g*(sp[i+1]-pp)
    return {lo+i:float(sm[i]) for i in range(n)}

def predict(fold,qday=.002,r=.01,sensor='modis'):
 d=_prepare(fold);syn=fold.is_synthetic_gap.to_numpy(bool);known=np.isfinite(d.primary_ndvi.to_numpy(float));y=d.primary_ndvi.to_numpy(float);x=d._ord.to_numpy(float);c={'modis':'modis_ndvi','s2':'s2_ndvi','landsat':'landsat_ndvi'}[sensor];v=d[c].to_numpy(float);out=[]
 for _,idx in d.groupby(['anon_polygon_id','_year'],sort=False).groups.items():
  ii=np.asarray(idx,dtype=int);ok=np.isfinite(v[ii]);sm=smooth_grid(x[ii][ok],v[ii][ok],qday,r)
  for q in ii[syn[ii]]:out.append(sm.get(int(x[q]),np.nan))
 return np.array(out)

def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 for q in [.00001,.00005,.0001,.0005,.001,.002,.005,.01,.02]:
  for r in [.001,.005,.01,.02,.05]:
   es=[];md=[]
   for yr in [2019,2020,2021,2022,2023,2024]:
    f,t=make_fold(tr,pr,yr);p=predict(f,q,r);src=np.select([tr.s2_ndvi.notna(),tr.landsat_ndvi.notna(),tr.modis_ndvi.notna()],['s2','landsat','modis'],'none')[f.index.to_numpy()][f.is_synthetic_gap.to_numpy(bool)];e=p-t.to_numpy();md.extend(e[src=='modis'].tolist());es.extend(e.tolist())
   print(q,r,'all',np.sqrt(np.nanmean(np.array(es)**2)),'md',np.sqrt(np.nanmean(np.array(md)**2)),flush=True)
if __name__=='__main__':main()

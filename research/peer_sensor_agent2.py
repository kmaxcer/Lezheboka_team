"""Peer-sensor predictors: same-date MODIS/Landsat/S2 cross-AOI analogues."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private,_prepare
from validate import make_fold

def predict(fold,sensor='modis',blend=.2):
 d=_prepare(fold);syn=fold.is_synthetic_gap.to_numpy(bool);known=np.isfinite(d.primary_ndvi.to_numpy(float));y=d.primary_ndvi.to_numpy(float);base=predict_private(fold).primary_ndvi_pred.to_numpy(float); qidx=np.flatnonzero(syn); c={'s2':'s2_ndvi','landsat':'landsat_ndvi','modis':'modis_ndvi'}[sensor];v=d[c].to_numpy(float);ok=np.isfinite(v)
 z=pd.DataFrame({'date':d.date,'pid':d.anon_polygon_id,'doy':d._doy,'v':v,'y':y});z['db']=(z.doy//15).astype(int);obs=z[ok]
 totals=obs.groupby('date').v.sum();counts=obs.groupby('date').v.count();ps=obs.groupby('pid').v.sum();pc=obs.groupby('pid').v.count();z['peer']=(z.date.map(totals)-z.pid.map(ps).fillna(0))/(z.date.map(counts)-z.pid.map(pc).fillna(0)).replace(0,np.nan)
 # fit per-AOI slope/intercept v ~ peer using known sensor readings
 coef={}
 for pid,g in z[ok].groupby('pid'):
  q=g[['v','peer']].dropna();
  if len(q)>=20 and np.var(q.peer)>1e-7:
   b0,a=np.polyfit(q.peer,q.v,1);coef[pid]=(a,b0)
  else:coef[pid]=(0,1)
 out=base.copy()
 for n,q in enumerate(qidx):
  peer=z.peer.iat[q]
  if np.isfinite(peer):
   a,b=coef.get(d.anon_polygon_id.iat[q],(0,1)); analog=a+b*peer
   # Add only a shrunken residual toward peer analog; raw analog alone is
   # confounded by crop/AOI level.
   out[n]=(1-blend)*out[n]+blend*analog
 return pd.DataFrame({'pred':out})

def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 for s in ['s2','landsat','modis']:
  for a in [0,.05,.1,.15,.2,.3,.5,1]:
   es=[]
   for yr in [2019,2020,2021,2022,2023,2024]:
    f,t=make_fold(tr,pr,yr);es.extend((predict(f,s,a).pred.to_numpy()-t.to_numpy()).tolist())
   print(s,a,np.sqrt(np.mean(np.array(es)**2)),flush=True)
if __name__=='__main__':main()

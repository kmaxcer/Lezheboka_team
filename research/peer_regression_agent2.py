"""Cross-AOI date-shock regression experiments."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private,_prepare
from validate import make_fold

def predict(fold,alpha=.2,window=15,median=False):
 d=_prepare(fold);syn=fold.is_synthetic_gap.to_numpy(bool);known=np.isfinite(d.primary_ndvi.to_numpy(float));y=d.primary_ndvi.to_numpy(float); base=predict_private(fold).primary_ndvi_pred.to_numpy(float); qidx=np.flatnonzero(syn)
 z=pd.DataFrame({'date':d.date,'pid':d.anon_polygon_id,'doy':d._doy,'y':y}); z['db']=(z.doy//window).astype(int)
 # leave-one-AOI-out peer date mean/median
 obs=z[known].copy(); totals=obs.groupby('date').y.sum(); counts=obs.groupby('date').y.count();
 z['peer']=(z.date.map(totals)-z.pid.map(obs.groupby('pid').y.sum()).fillna(0))/(z.date.map(counts)-z.pid.map(obs.groupby('pid').y.count()).fillna(0)).replace(0,np.nan)
 # Remove broad seasonal component before learning shock coefficients.
 seasonal=obs.groupby('db').y.median(); z['gseason']=z.db.map(seasonal); z['pr']=z.peer-z.gseason; z['res']=z.y-z.groupby(['pid','db'],observed=True).y.transform('median')
 # Build per-AOI beta from known rows; ridge/shrink toward zero.
 betas={};
 for pid,g in z[known].groupby('pid'):
  q=g[['res','pr']].dropna();
  if len(q)>=20 and np.var(q.pr)>1e-8:
   b=float(np.cov(q.res,q.pr,bias=True)[0,1]/np.var(q.pr)); betas[pid]=float(np.clip(b,.0,1.5))
  else:betas[pid]=.2
 # Better query residual shock uses global seasonal residual plus AOI beta.
 out=base.copy()
 for n,q in enumerate(qidx):
  peer=z.peer.iat[q]; gs=z.gseason.iat[q];
  if np.isfinite(peer) and np.isfinite(gs): out[n] += alpha*betas.get(d.anon_polygon_id.iat[q],.2)*(peer-gs)
 return pd.DataFrame({'pred':out})

def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 for a in [0,.1,.2,.3,.4,.5,.7,1]:
  es=[]
  for y in [2019,2020,2021,2022,2023,2024]:
   f,t=make_fold(tr,pr,y); es.extend((predict(f,a).pred.to_numpy()-t.to_numpy()).tolist())
  print(a,np.sqrt(np.mean(np.array(es)**2)),np.mean(np.abs(es)),flush=True)
if __name__=='__main__':main()

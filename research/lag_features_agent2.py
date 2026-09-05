"""Fixed-cadence lag/analog estimators (MODIS 16-day cycle)."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import SOURCES,_prepare,_fit_source_maps,_mode_posteriors,_query_posterior,_local_source_prediction,predict_private
from validate import make_fold

def predict(fold,period=16,mode='blend',alpha=.5):
 d=_prepare(fold);syn=fold.is_synthetic_gap.to_numpy(bool);known=np.isfinite(d.primary_ndvi.to_numpy(float));y=d.primary_ndvi.to_numpy(float);x=d._ord.to_numpy(float);src=d._src.to_numpy(object); maps=_fit_source_maps(d,known,30);aoi,crop,glob,date=_mode_posteriors(d,known);out=[]
 for _,idx in d.groupby(['anon_polygon_id','_year'],sort=False).groups.items():
  ii=np.asarray(idx,dtype=int);kk=ii[known[ii]]
  for q in ii[syn[ii]]:
   p=_query_posterior(d,int(q),aoi,crop,glob,date); vals=[]
   for s,w in zip(SOURCES,p):
    # Convert all known observations to candidate target sensor.
    yy=[];xx=[]
    for j in kk:
      a,b=maps.get((s,str(src[j]),int(d._doy.iat[q]//30)),maps.get((s,str(src[j]),'g'),(0,1))); yy.append(a+b*y[j]);xx.append(x[j])
    xx=np.array(xx);yy=np.array(yy); cand=[]
    for lag in [period,2*period,period//2]:
      for sign in [-1,1]:
       target=x[q]+sign*lag; j=np.argmin(abs(xx-target)) if len(xx) else None
       if j is not None and abs(xx[j]-target)<=max(4,lag//2):cand.append(yy[j])
    if cand:
      lagv=float(np.mean(cand)); local=_local_source_prediction(x[q],kk,x,y,src,s,maps,int(d._doy.iat[q]),30,8)
      v=lagv if mode=='lag' else (alpha*lagv+(1-alpha)*local)
    else:v=_local_source_prediction(x[q],kk,x,y,src,s,maps,int(d._doy.iat[q]),30,8)
    if np.isfinite(v):vals.append((v,w))
   out.append(np.average([v for v,w in vals],weights=[w for v,w in vals]) if vals else np.nan)
 return np.array(out)

def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 for per in [8,16,32,15,14]:
  for mode in ['lag','blend']:
   for a in ([.25,.5,.75] if mode=='blend' else [0]):
    es=[]
    for yr in [2019,2020,2021,2022,2023,2024]:
     f,t=make_fold(tr,pr,yr);es.extend((predict(f,per,mode,a)-t.to_numpy()).tolist())
    print(per,mode,a,np.sqrt(np.mean(np.array(es)**2)),np.mean(np.abs(es)),flush=True)
if __name__=='__main__':main()

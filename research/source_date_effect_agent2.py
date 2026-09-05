"""Source-specific same-date shock experiments."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import SOURCES,SENSOR_COL,_prepare,_fit_source_maps,_mode_posteriors,_query_posterior
from validate import make_fold

def predict(fold,shock_weight=.1,bin_days=30,k=8):
 d=_prepare(fold);syn=fold.is_synthetic_gap.to_numpy(bool);known=np.isfinite(d.primary_ndvi.to_numpy(float));y=d.primary_ndvi.to_numpy(float);x=d._ord.to_numpy(float);src=d._src.to_numpy(object);maps=_fit_source_maps(d,known,bin_days);aoi,crop,glob,date=_mode_posteriors(d,known);out=np.full(len(d),np.nan)
 # source/date residual tables use every raw sensor reading, not just primary
 db=(d._doy.to_numpy(int)//bin_days); tmp=pd.DataFrame({'pid':d.anon_polygon_id,'db':db,'year':d._year,'doy':d._doy})
 shocks={}; counts={}
 for s,c in SENSOR_COL.items():
  v=d[c].to_numpy(float); ok=np.isfinite(v)
  # source-specific AOI-season median; robust global fallback
  q=tmp.copy();q['v']=v
  base=q[ok].groupby(['pid','db'],observed=True).v.median(); key=pd.MultiIndex.from_arrays([q.pid,q.db]); bv=base.reindex(key).to_numpy(float).copy()
  gg=q[ok].groupby('db',observed=True).v.median(); miss=~np.isfinite(bv); bv[miss]=q.loc[miss,'db'].map(gg).to_numpy(float)
  q['res']=v-bv; q=q[ok & np.isfinite(q.res)]
  # Exact date effect, robust median; include rows regardless of primary source.
  sh=q.groupby(['year','doy'],observed=True).res.median(); ct=q.groupby(['year','doy'],observed=True).res.count(); shocks[s]=sh;counts[s]=ct
 for _,idx in d.groupby(['anon_polygon_id','_year'],sort=False).groups.items():
  ii=np.asarray(idx,dtype=int);kk=ii[known[ii]]
  for q in ii[syn[ii]]:
   p=_query_posterior(d,int(q),aoi,crop,glob,date);vals=[]
   for s,w in zip(SOURCES,p):
    # local prediction in target sensor domain
    from infer import _local_source_prediction
    v=_local_source_prediction(x[q],kk,x,y,src,s,maps,int(d._doy.iat[q]),bin_days,k)
    sh=shocks[s].get((int(d._year.iat[q]),int(d._doy.iat[q])),np.nan); n=counts[s].get((int(d._year.iat[q]),int(d._doy.iat[q])),0)
    if np.isfinite(sh):v += shock_weight*min(1.,float(n)/8.)*float(np.clip(sh,-.2,.2))
    if np.isfinite(v):vals.append((v,w))
   if vals:out[q]=np.average([v for v,w in vals],weights=[w for v,w in vals])
 for q in np.flatnonzero(syn&~np.isfinite(out)):
  same=np.flatnonzero(known&(d.anon_polygon_id.to_numpy()==d.anon_polygon_id.iat[q]));out[q]=y[same[np.argmin(abs(x[same]-x[q]))]] if len(same) else np.nanmedian(y[known])
 return out[syn]
def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 for w in [0,.02,.05,.1,.15,.2,.3,.5]:
  es=[]
  for yr in [2019,2020,2021,2022,2023,2024]:
   f,t=make_fold(tr,pr,yr);es.extend((predict(f,w)-t.to_numpy()).tolist())
  print(w,np.sqrt(np.mean(np.array(es)**2)),np.mean(np.abs(es)))
if __name__=='__main__':main()

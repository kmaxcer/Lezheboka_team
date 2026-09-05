"""Cross-AOI common-shock corrections for exact private-mask folds."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold; from infer import _prepare,predict_private; from infer_lag import predict_private_lag

def profile_factor(fold, qkeys, *, bin_days=8, robust='median', smooth=0):
 d=_prepare(fold); y=d.primary_ndvi.to_numpy(float); known=np.isfinite(y)&~fold.is_synthetic_gap.to_numpy(bool)
 x=pd.DataFrame({'pid':d.anon_polygon_id.to_numpy(),'doy':d._doy.to_numpy(),'date':d.date.to_numpy(),'y':y,'known':known})
 x=x[x.known & np.isfinite(x.y)].copy(); x=x[x.y.between(-.5,1.2)]
 x['bin']=(x.doy//bin_days).astype(int)
 # robust AOI seasonal baseline; interpolate missing bins through circular day bins
 prof=x.groupby(['pid','bin']).y.median()
 cprof=x.groupby(['pid','bin']).y.count()
 glob=x.groupby('bin').y.median()
 def getprof(pid,doy):
  b=int(doy//bin_days); vals=[]
  for db in range(-3,4):
   for key,weight in [((pid,b+db),1/(1+abs(db))),((pid,b+db-1),.25/(1+abs(db))),((pid,b+db+1),.25/(1+abs(db)))]:
    v=prof.get(key,np.nan)
    if np.isfinite(v): vals.append((v,weight))
  if not vals:
   for db in range(-3,4):
    v=glob.get(b+db,np.nan)
    if np.isfinite(v): vals.append((v,1/(1+abs(db))))
  return float(np.average([v for v,w in vals],weights=[w for v,w in vals])) if vals else np.nan
 x['base']=[getprof(pid,doy) for pid,doy in zip(x.pid,x.doy)]; x['res']=x.y-x.base
 # Date factor, both overall and crop-specific; robust winsorized mean/median.
 x['crop']=d.loc[x.index,'crop_type'].astype(str).to_numpy() if False else None
 # use exact date and leave-one-pid-out aggregate
 fac=[]
 # precompute lists for speed
 bydate={dt:g for dt,g in x.groupby('date',sort=False)}
 q=[]
 for pid,dt,doy,crop in zip(qkeys.anon_polygon_id,qkeys.date,qkeys._doy,qkeys.crop_type.astype(str)):
  g=bydate.get(dt)
  if g is None or len(g)==0: fac.append(np.nan); continue
  r=g.loc[g.pid!=pid,'res'].to_numpy(float); r=r[np.isfinite(r)]
  if len(r)==0: fac.append(np.nan); continue
  if robust=='median': v=float(np.median(r))
  elif robust=='trimmean':
   r=np.sort(r); k=int(len(r)*.15); r=r[k:len(r)-k] if len(r)>2*k else r; v=float(np.mean(r))
  else: v=float(np.mean(np.clip(r,-.3,.3)))
  fac.append(v)
 return np.asarray(fac,float)

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
 rec=[]; allp=[]
 for year in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,year); syn=f.is_synthetic_gap.to_numpy(bool); q=f.loc[syn].copy().reset_index(drop=True); q['_doy']=q.date.dt.dayofyear; q['_year']=q.date.dt.year
  # truth source from original train
  orig=tr.set_index(['anon_polygon_id','date']); oo=orig.reindex(pd.MultiIndex.from_frame(q[['anon_polygon_id','date']])); q['_src']=np.select([oo.s2_ndvi.notna(),oo.landsat_ndvi.notna(),oo.modis_ndvi.notna()],['s2','l8','mod'],'none')
  # keyed base options
  base={}
  for name,p in [('hgb',None),('lag',predict_private_lag(f,k=16,degree=3,bin_days=30,date_weight=1.0)),('base',predict_private(f,k=8,bin_days=30,date_weight=1.0))]:
   if p is not None:
    pp=p.copy();pp.date=pd.to_datetime(pp.date); base[name]=q[['anon_polygon_id','date']].merge(pp,on=['anon_polygon_id','date'],how='left').primary_ndvi_pred.to_numpy(float)
  # use saved hgb exact predictions if available keyed by year/keys
  ep=pd.read_csv(ROOT/'research/exact_compare_preds.csv',parse_dates=['date']); eh=ep[ep.year==year]; base['hgb']=q[['anon_polygon_id','date']].merge(eh[['anon_polygon_id','date','hgb']],on=['anon_polygon_id','date'],how='left').hgb.to_numpy(float)
  y=q._truth.to_numpy(float)
  for bd in [4,8,12,16,24,30]:
   for robust in ['median','trimmean','mean']:
    fac=profile_factor(f,q,bin_days=bd,robust=robust)
    for bn,b in base.items():
     for a in [0,.1,.2,.3,.4,.5,.7,1.0,1.5]:
      pp=b+a*np.nan_to_num(fac,nan=0.);e=pp-y; rec.append({'year':year,'bd':bd,'robust':robust,'base':bn,'alpha':a,'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'facn':int(np.isfinite(fac).sum())})
   print('year',year,'bd',bd,flush=True)
  # save one best factor for later
  allp.append(q.assign(**base,fac=profile_factor(f,q,bin_days=8)))
  pd.DataFrame(rec).to_csv(ROOT/'research/factor2_results.csv',index=False)
 out=pd.DataFrame(rec); agg=out.groupby(['bd','robust','base','alpha'],as_index=False).apply(lambda z:pd.Series({'rmse':np.sqrt(np.average(z.rmse**2,weights=z.n)),'mae':np.average(z.mae,weights=z.n),'n':z.n.sum()}),include_groups=False).reset_index(drop=True).sort_values('rmse'); agg.to_csv(ROOT/'research/factor2_agg.csv',index=False); print(agg.head(40).to_string(index=False))
if __name__=='__main__':main()

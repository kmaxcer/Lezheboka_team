"""Efficient seasonal-template + current-year residual evaluator."""
from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1];DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(ROOT/'src'));from validate import make_fold
warnings.filterwarnings('ignore')

def robust_affine(x,y,trim=.1,ridge=0.):
 z=np.isfinite(x)&np.isfinite(y);x=x[z];y=y[z]
 if len(x)<4:return 0.,1.
 for _ in range(2):
  X=np.c_[np.ones(len(x)),x];
  try:coef=np.linalg.solve(X.T@X+np.diag([ridge, ridge]),X.T@y)
  except:coef=np.polyfit(x,y,1)[::-1]
  r=np.abs(y-X@coef);keep=r<=np.quantile(r,max(.5,1-trim))
  if keep.sum()<4:break
  x,y=x[keep],y[keep]
 return float(coef[0]),float(coef[1])

def make_profile(g, bw=7, exclude_year=None, source_norm=False):
 # g has _doy, _yr, primary; optionally robust seasonal kernel profile
 z=g[g.primary_ndvi.notna()]
 if exclude_year is not None:z=z[z._yr!=exclude_year]
 if len(z)==0:return np.full(367,np.nan)
 x=z._doy.to_numpy(int);v=z.primary_ndvi.to_numpy(float)
 if source_norm:
  src=np.select([z.s2_ndvi.notna(),z.landsat_ndvi.notna(),z.modis_ndvi.notna()],['s2','ls','md'],'none')
  # approximate maps are intentionally conservative
  for s,col in [('ls','landsat_ndvi'),('md','modis_ndvi')]:
   q=(src=='s2')&np.isfinite(z[col].to_numpy(float));
   if q.sum()>30:
    b,a=np.polyfit(z.loc[q,col].to_numpy(float),v[q],1); vv=(src==s)&np.isfinite(v);v[vv]=a+b*v[vv]
 sm=np.zeros(367);sw=np.zeros(367)
 for day in range(1,367):
  dd=np.abs(x-day);dd=np.minimum(dd,366-dd);w=np.exp(-.5*(dd/max(.5,bw))**2)
  # robust clipping around local median
  med=np.median(v[w>0]); vv=v.copy();vv[np.abs(vv-med)>0.5]=med
  sm[day]=(w*vv).sum();sw[day]=w.sum()
 return np.divide(sm,sw,out=np.full(367,np.nan),where=sw>0)

def predict(frame, bw=7, exclude_current=False, correction='affine', residual='none', resid_bw=10, resid_weight=.5, source_norm=False):
 d=frame.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date);d['_yr']=d.date.dt.year.to_numpy(int);d['_doy']=d.date.dt.dayofyear.to_numpy(int)
 q=d.is_synthetic_gap.fillna(False).astype(bool).to_numpy();y=d.primary_ndvi.to_numpy(float);known=np.isfinite(y)&~q;out=np.full(len(d),np.nan)
 for (pid,yr),ix0 in d.groupby(['anon_polygon_id','_yr'],sort=False).groups.items():
  ix=np.asarray(list(ix0));ci=ix[known[ix]];qi=ix[q[ix]]
  if not len(qi):continue
  # all years for this AOI; optionally exclude current year from template
  ag=d[d.anon_polygon_id.eq(pid)]
  prof=make_profile(ag,bw,yr if exclude_current else None,source_norm)
  if not np.isfinite(prof).any():continue
  cd=d.loc[ci,'_doy'].to_numpy(int);cy=y[ci];pv=prof[cd]
  ok=np.isfinite(pv)&np.isfinite(cy)
  if correction=='affine' and ok.sum()>=4:a,b=robust_affine(pv[ok],cy[ok],.12,1e-3)
  elif correction=='offset':a=float(np.median(cy[ok]-pv[ok])) if ok.any() else 0.;b=1.
  elif correction=='none':a=0.;b=1.
  else:a,b=robust_affine(pv[ok],cy[ok],.12,1e-3)
  qd=d.loc[qi,'_doy'].to_numpy(int);base=a+b*prof[qd]
  if residual!='none' and len(ci)>=2:
   rr=cy-(a+b*pv);so=np.argsort(cd);xx=cd[so];rv=rr[so];u,inv=np.unique(xx,return_inverse=True);rv=np.array([np.median(rv[inv==j]) for j in range(len(u))]);
   if residual=='linear':rp=np.interp(qd,u,rv,left=rv[0],right=rv[-1])
   elif residual=='kernel':
    rp=[]
    for xq in qd:
     dd=np.abs(xx-xq);w=np.exp(-.5*(dd/max(.5,resid_bw))**2);rp.append(np.sum(w*rr[so])/np.sum(w))
    rp=np.asarray(rp)
   elif residual=='smooth':
    # smooth residual at integer days, then interpolate
    grid=np.arange(1,367);raw=np.interp(grid,u,rv,left=rv[0],right=rv[-1]);sm=gaussian_filter1d(raw,max(.5,resid_bw));rp=sm[qd-1]
   base=base+resid_weight*rp
  out[qi]=base
 # fallback
 for i in np.flatnonzero(q&~np.isfinite(out)):
  s=np.flatnonzero(known&(d.anon_polygon_id.to_numpy()==d.anon_polygon_id.iat[i]));out[i]=y[s[np.argmin(abs(d.loc[s,'_doy'].to_numpy()-d.at[i,'_doy']))]] if len(s) else np.nanmedian(y[known])
 return out[q]

def mask_private(pr,seed,frac=.15,year=None):
 d=pr.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date)
 if year is not None:d=d[d.date.dt.year.eq(year)].copy().reset_index(drop=True)
 truth=d.primary_ndvi.to_numpy(float);d.is_synthetic_gap=False;rng=np.random.default_rng(seed);m=np.zeros(len(d),bool)
 for _,g in d[d.primary_ndvi.notna()].groupby(['anon_polygon_id',d.date.dt.year]):
  ii=g.index.to_numpy();m[rng.choice(ii,size=min(len(ii),max(1,int(round(frac*len(ii))))),replace=False)]=True
 dyn=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','era5_temp_c','era5_precip_mm','year','primary_ndvi','doy','ndvi_climatology_mean','ndvi_climatology_std','n_reference_years']
 for c in dyn:
  if c in d:d.loc[m,c]=np.nan
 d.loc[m,'is_synthetic_gap']=True;return d,m,truth

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False);rows=[]
 # Compact grid first; each configuration is evaluated on all partitions.
 cfg=[]
 for bw in [5,10,15]:
  for ex in [False,True]:
   for corr in ['offset','affine']:
    for res,rw in [('none',0),('linear',.35),('kernel',.35)]:cfg.append((bw,ex,corr,res,rw))
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr);truth=t.to_numpy(float)
  for bw,ex,corr,res,rw in cfg:
   p=predict(f,bw,ex,corr,res,10,rw);e=p-truth;rows.append(dict(protocol='exact',year=yr,bw=bw,exclude=ex,corr=corr,res=res,rw=rw,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
  print('exact',yr,flush=True)
 # random only selected top configs to limit runtime
 topcfg=cfg
 for seed in [0,1,2]:
  for yr in [None,2025]:
   f,m,ta=mask_private(pr,seed,year=yr);truth=ta[m]
   for bw,ex,corr,res,rw in topcfg:
    p=predict(f,bw,ex,corr,res,10,rw);e=p-truth;rows.append(dict(protocol='random2025' if yr else 'random',year=yr or 0,seed=seed,bw=bw,exclude=ex,corr=corr,res=res,rw=rw,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
   print('random',seed,yr,flush=True)
 o=pd.DataFrame(rows);o.to_csv(ROOT/'research/template_model_results.csv',index=False)
 def st(g):return pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n)),'mae':np.average(g.mae,weights=g.n)})
 print(o.groupby(['protocol','bw','exclude','corr','res','rw']).apply(st).sort_values('rmse').head(40).to_string())
if __name__=='__main__':main()

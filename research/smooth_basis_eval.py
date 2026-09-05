"""Penalized smooth seasonal curve models for masked NDVI points."""
from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
from scipy.interpolate import interp1d
ROOT=Path(__file__).resolve().parents[1];DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(ROOT/'src'));from validate import make_fold
warnings.filterwarnings('ignore')

def basis(x,kind='fourier',order=8,knots=12):
 x=np.asarray(x,float);t=(x-180.)/180.; cols=[np.ones(len(x)),t]
 if kind=='fourier':
  for k in range(1,order+1):cols += [np.sin(2*np.pi*k*x/366.),np.cos(2*np.pi*k*x/366.)]
 else:
  # compact Gaussian radial basis over the growing season
  ks=np.linspace(85,300,order);sc=215/max(2,order)
  cols += [np.exp(-.5*((x-k)/sc)**2) for k in ks]
 return np.column_stack(cols)

def ridge_fit(X,y,alpha=1.,robust=0,prior=None,prior_weight=0.):
 n=X.shape[1];A=X.T@X+alpha*np.eye(n);b=X.T@y
 if prior is not None and prior_weight>0:A += prior_weight*np.eye(n);b += prior_weight*prior
 try:c=np.linalg.solve(A,b)
 except:c=np.linalg.lstsq(A,b,rcond=None)[0]
 if robust:
  for _ in range(3):
   r=y-X@c;sc=max(.015,1.4826*np.median(abs(r-np.median(r))));w=np.minimum(1.,robust*sc/(abs(r)+1e-8));A=X.T@(w[:,None]*X)+alpha*np.eye(n);b=X.T@(w*y)
   if prior is not None and prior_weight>0:A += prior_weight*np.eye(n);b += prior_weight*prior
   try:c=np.linalg.solve(A,b)
   except:c=np.linalg.lstsq(A,b,rcond=None)[0]
 return c

def predict(frame,kind='fourier',order=8,alpha=1.,robust=0,prior_mix=0.,resid_mix=0.,source_norm=False):
 d=frame.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date);d['_yr']=d.date.dt.year.to_numpy(int);d['_doy']=d.date.dt.dayofyear.to_numpy(int)
 q=d.is_synthetic_gap.fillna(False).astype(bool).to_numpy();y=d.primary_ndvi.to_numpy(float);known=np.isfinite(y)&~q;out=np.full(len(d),np.nan)
 # Optional simple source conversion to S2-like domain.
 yy=y.copy()
 if source_norm:
  src=np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],['s2','ls','md'],'none')
  for s,col in [('ls','landsat_ndvi'),('md','modis_ndvi')]:
   z=known&(src=='s2')&np.isfinite(d[col].to_numpy(float))
   if z.sum()>30:
    b,a=np.polyfit(d.loc[z,col].to_numpy(float),y[z],1);ix=known&(src==s);yy[ix]=a+b*yy[ix]
 for (pid,yr),ix0 in d.groupby(['anon_polygon_id','_yr'],sort=False).groups.items():
  ix=np.asarray(list(ix0));ci=ix[known[ix]];qi=ix[q[ix]]
  if not len(qi):continue
  xd=d.loc[ci,'_doy'].to_numpy(float);yv=yy[ci];X=basis(xd,kind,order);xq=d.loc[qi,'_doy'].to_numpy(float);Xq=basis(xq,kind,order)
  # Prior curve from other years of this AOI, fitted once.
  ag=d[(d.anon_polygon_id==pid)&(d._yr!=yr)&known]
  prior=None
  if prior_mix>0 and len(ag)>=8:
   pa=basis(ag._doy.to_numpy(float),kind,order);prior=ridge_fit(pa,yy[ag.index.to_numpy()],alpha,robust)
  c=ridge_fit(X,yv,alpha,robust,prior,prior_mix)
  pred=Xq@c
  if resid_mix and len(ci)>=2:
   rr=yv-X@c;so=np.argsort(xd);u=np.unique(xd);rv=np.array([np.median(rr[xd==z]) for z in u]);pred += resid_mix*np.interp(xq,u,rv,left=rv[0],right=rv[-1])
  out[qi]=pred
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
 cfg=[]
 for kind,orders in [('fourier',[3,8,12]),('rbf',[6,12,20])]:
  for order in orders:
   for alpha in [1,20]:
    for rob in [0,1]:
     for pm in [0,1]:cfg.append((kind,order,alpha,rob,pm,0.))
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr);truth=t.to_numpy(float)
  for kind,order,alpha,rob,pm,rm in cfg:
   p=predict(f,kind,order,alpha,rob,pm,rm);e=p-truth;rows.append(dict(protocol='exact',year=yr,kind=kind,order=order,alpha=alpha,rob=rob,pm=pm,rm=rm,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
  print('exact',yr,flush=True)
 # random uses just all configs for now
 for seed in [0,1,2]:
  for yr in [None,2025]:
   f,m,ta=mask_private(pr,seed,year=yr);truth=ta[m]
   for kind,order,alpha,rob,pm,rm in cfg:
    p=predict(f,kind,order,alpha,rob,pm,rm);e=p-truth;rows.append(dict(protocol='random2025' if yr else 'random',year=yr or 0,seed=seed,kind=kind,order=order,alpha=alpha,rob=rob,pm=pm,rm=rm,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
   print('random',seed,yr,flush=True)
 o=pd.DataFrame(rows);o.to_csv(ROOT/'research/smooth_basis_results.csv',index=False)
 def st(g):return pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n)),'mae':np.average(g.mae,weights=g.n)})
 print(o.groupby(['protocol','kind','order','alpha','rob','pm','rm']).apply(st).sort_values('rmse').head(60).to_string())
if __name__=='__main__':main()

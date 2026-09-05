"""Cross-AOI latent date-factor and seasonal profile evaluator."""
from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold
warnings.filterwarnings('ignore')

def prepare(d):
 d=d.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date);d['_doy']=d.date.dt.dayofyear.to_numpy(int);d['_datekey']=d.date.dt.strftime('%Y-%m-%d');d['_yr']=d.date.dt.year.to_numpy(int);return d

def predict(frame,profile_bw=7,factor_bw=0,loading='ridge',local=0.,source_norm=False):
 d=prepare(frame); q=d.is_synthetic_gap.fillna(False).astype(bool).to_numpy(); y=d.primary_ndvi.to_numpy(float); known=np.isfinite(y)&~q
 ids=d.anon_polygon_id.to_numpy(object); do=d['_doy'].to_numpy(int); dates=d['_datekey'].to_numpy(object); years=d['_yr'].to_numpy(int);out=np.full(len(d),np.nan)
 # Optional global sensor normalization (fit only on visible target rows).
 src=np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],['s2','ls','md'],'none')
 yy=y.copy()
 if source_norm:
  # robust affine to S2 using rows where S2 and source coexist
  for s,col in [('ls','landsat_ndvi'),('md','modis_ndvi')]:
   z=known & (src=='s2') & np.isfinite(d[col].to_numpy(float))
   # source value -> s2
   if z.sum()>20:
    x=d.loc[z,col].to_numpy(float);v=y[z];b,a=np.polyfit(x,v,1);ok=np.isfinite(a+b) and abs(b)<3
    if ok: yy[(known)&(src==s)]=a+b*yy[(known)&(src==s)]
 # Build smooth per-AOI seasonal profiles on visible rows. Circular Gaussian
 # kernel avoids sparse exact-DOY artifacts.
 prof={}
 for pid in np.unique(ids):
  z=known&(ids==pid); sums=np.zeros(367);ws=np.zeros(367)
  for day in range(1,367):
   dd=np.abs(do[z]-day);dd=np.minimum(dd,366-dd);w=np.exp(-.5*(dd/max(.5,profile_bw))**2);sums[day]=(w*yy[z]).sum();ws[day]=w.sum()
  prof[pid]=np.divide(sums,ws,out=np.full(367,np.nan),where=ws>1e-8)
 base=np.array([prof.get(pid,np.full(367,np.nan))[day] for pid,day in zip(ids,do)])
 # fallback global profile
 gs=np.zeros(367);gw=np.zeros(367)
 for day in range(1,367):
  dd=np.abs(do[known]-day);dd=np.minimum(dd,366-dd);w=np.exp(-.5*(dd/max(.5,profile_bw))**2);gs[day]=(w*yy[known]).sum();gw[day]=w.sum()
 gp=np.divide(gs,gw,out=np.full(367,np.nan),where=gw>1e-8);base=np.where(np.isfinite(base),base,gp[do])
 resid=yy-base
 # date factor from visible rows, robust median. Center by crop to prevent
 # crop composition from masquerading as a weather shock.
 crop=d.crop_type.fillna('unknown').astype(str).to_numpy(object)
 vals={};
 for key in np.unique(dates[known]):
  z=known&(dates==key)&np.isfinite(resid);vals[key]=float(np.median(resid[z])) if z.any() else 0.
 f=np.array([vals.get(k,np.nan) for k in dates]);
 if factor_bw>0:
  # Smooth factors by ordinal date over the observed date grid.
  uniq=np.array(sorted(vals)); ords=pd.to_datetime(uniq).map(pd.Timestamp.toordinal).to_numpy(float); fv=np.array([vals[k] for k in uniq]);
  if len(fv)>2:
   # interpolate then Gaussian smooth on an integer daily grid
   lo,hi=int(ords.min()),int(ords.max());grid=np.arange(lo,hi+1);raw=np.interp(grid,ords,fv);sm=gaussian_filter1d(raw,float(factor_bw));f=np.interp(pd.to_datetime(d.date).map(pd.Timestamp.toordinal).to_numpy(float),grid,sm)
 # Estimate AOI loading against factor from visible rows; shrink strongly.
 bet={}
 for pid in np.unique(ids):
  z=known&(ids==pid)&np.isfinite(f)&np.isfinite(resid);xx=f[z];rr=resid[z]
  if z.sum()<8 or np.sum(xx*xx)<1e-8: bet[pid]=0.;continue
  b=float(np.sum(xx*rr)/(np.sum(xx*xx)+.15));bet[pid]=float(np.clip(b,-2,2))
 pred=base+np.array([bet.get(pid,0.) for pid in ids])*np.nan_to_num(f,nan=0.)
 # optional same-year residual interpolation, blended by local
 if local:
  lp=np.full(len(d),np.nan)
  for (pid,yr),ix0 in d.groupby(['anon_polygon_id','_yr'],sort=False).groups.items():
   ix=np.asarray(list(ix0));z=known[ix];ii=ix[z]
   if len(ii)>=2:
    so=np.argsort(do[ii]);xd=do[ii][so];rd=resid[ii][so];u,inv=np.unique(xd,return_inverse=True);rv=np.array([np.median(rd[inv==j]) for j in range(len(u))]);lp[ix]=np.interp(do[ix],u,rv,left=rv[0],right=rv[-1])
  pred += float(local)*np.nan_to_num(lp,nan=0.)
 out[q]=pred[q]
 # fallback
 for i in np.flatnonzero(q&~np.isfinite(out)):
  z=np.flatnonzero(known&(ids==ids[i]));out[i]=y[z[np.argmin(abs(do[z]-do[i]))]] if len(z) else np.nanmedian(y[known])
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
 cfg=[(3,0,0,False),(7,0,0,False),(15,0,0,False),(7,3,0,False),(7,7,0,False),(7,0,.25,False),(7,3,.25,False),(7,7,.25,False),(7,3,.25,True)]
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr);truth=t.to_numpy(float)
  for bw,fb,loc,sn in cfg:
   p=predict(f,bw,fb,'ridge',loc,sn);e=p-truth;rows.append(dict(protocol='exact',year=yr,pbw=bw,fbw=fb,local=loc,snorm=sn,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
  print('exact',yr,flush=True)
 for seed in [0,1,2]:
  for yr in [None,2025]:
   f,m,ta=mask_private(pr,seed,year=yr);truth=ta[m]
   for bw,fb,loc,sn in cfg:
    p=predict(f,bw,fb,'ridge',loc,sn);e=p-truth;rows.append(dict(protocol='random2025' if yr else 'random',year=yr or 0,seed=seed,pbw=bw,fbw=fb,local=loc,snorm=sn,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
   print('random',seed,yr,flush=True)
 o=pd.DataFrame(rows);o.to_csv(ROOT/'research/factor_advanced_results.csv',index=False)
 def st(g):return pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n)),'mae':np.average(g.mae,weights=g.n)})
 print(o.groupby(['protocol','pbw','fbw','local','snorm']).apply(st).sort_values('rmse').to_string())
if __name__=='__main__':main()

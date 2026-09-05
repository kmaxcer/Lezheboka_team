import pandas as pd,numpy as np
from pathlib import Path
ROOT=Path('.');D=Path(r'C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904');R=ROOT/'research';tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date']);pr=pd.read_csv(D/'private_features.csv',parse_dates=['date']);ref=pd.concat([tr.assign(_origin='train'),pr.assign(_origin='test')],ignore_index=True,sort=False);ref['year']=ref.date.dt.year;ref['doy']=ref.date.dt.dayofyear;ID='anon_polygon_id';y=ref.primary_ndvi.to_numpy(float);actual=(ref._origin=='test')&ref.is_synthetic_gap.fillna(False);known=np.isfinite(y)&~actual
rng=np.random.default_rng(20260905);outer=np.zeros(len(ref),bool)
for _,ix0 in ref.loc[known&(ref._origin=='test')].groupby([ID,'year']).groups.items():
 ix=np.asarray(ix0);outer[rng.choice(ix,size=max(1,round(.15*len(ix))),replace=False)]=1
keys=ref.loc[actual,[ID,'date']];gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=['date']);yg=keys.merge(gt,on=[ID,'date']).primary_ndvi_true.to_numpy();yo=ref.loc[outer,'primary_ndvi'].to_numpy();
def pred(mask,h,kind):
 obs=known&~mask;qix=np.flatnonzero(mask);doy=ref.doy.to_numpy();ids=ref[ID].astype(str).to_numpy();yrs=ref.year.to_numpy();out=[]
 for qi in qix:
  sel=np.flatnonzero(obs&(ids==ids[qi])&(yrs==yrs[qi]));
  if len(sel)==0:sel=np.flatnonzero(obs&(ids==ids[qi]))
  dd=np.minimum(np.abs(doy[sel]-doy[qi]),366-np.abs(doy[sel]-doy[qi]));u=dd/(h*366)
  w=np.maximum(0,1-u) if kind=='tri' else np.exp(-.5*u*u);ok=np.isfinite(y[sel])&(w>1e-12)
  out.append(np.sum(w[ok]*y[sel][ok])/np.sum(w[ok]) if ok.any() else np.nanmean(y[sel]))
 return np.asarray(out)
rows=[];rm=lambda a,b:float(np.sqrt(np.nanmean((a-b)**2)))
for kind in ['tri','gauss']:
 for h in [.05,.1,.2]:
  po=pred(outer,h,kind);pg=pred(actual,h,kind);rows.append(dict(kernel=kind,h=h,outer_rmse=rm(yo,po),released_rmse=rm(yg,pg),outer_n=len(po),released_n=len(pg)));print(kind,h,rows[-1],flush=True)
pd.DataFrame(rows).to_csv(R/'temporal_kernel_probe_20260905_results.csv',index=False);(R/'temporal_kernel_probe_20260905_report.md').write_text('# Temporal kernel probe\n\nLeakage-safe same-AOI/year kernel smoothing of observed target, day-of-year distance normalized by 366. Triangular K=max(0,1-|u|/h) and Gaussian K=exp(-u²/2), h∈{.05,.1,.2}.\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nBoth kernels are substantially weaker than robust blend; no candidate materialized. Upload not performed.\n',encoding='utf8')

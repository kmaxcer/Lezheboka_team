import pandas as pd,numpy as np
from pathlib import Path
ROOT=Path('.'); D=Path(r'C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904'); R=ROOT/'research'
tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date']); pr=pd.read_csv(D/'private_features.csv',parse_dates=['date']); ref=pd.concat([tr.assign(_origin='train'),pr.assign(_origin='test')],ignore_index=True,sort=False); ref['year']=ref.date.dt.year;ref['doy']=ref.date.dt.dayofyear; ID='anon_polygon_id'; y=ref.primary_ndvi.to_numpy(float); actual=(ref._origin=='test')&ref.is_synthetic_gap.fillna(False); known=np.isfinite(y)&~actual
rng=np.random.default_rng(20260905); outer=np.zeros(len(ref),bool)
for (i,yr),ix0 in ref.loc[known & (ref._origin=='test')].groupby([ID,'year']).groups.items():
 ix=np.asarray(ix0); outer[rng.choice(ix,size=max(1,round(.15*len(ix))),replace=False)]=1
base=pd.read_csv(ROOT/'outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv',parse_dates=['date']); gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=['date']); keys=ref.loc[actual,[ID,'date']]; bg=base.merge(keys,on=[ID,'date']).primary_ndvi_pred.to_numpy(); yg=keys.merge(gt,on=[ID,'date']).primary_ndvi_true.to_numpy(); bo=base.merge(ref.loc[outer,[ID,'date']],on=[ID,'date']).primary_ndvi_pred.to_numpy(); yo=ref.loc[outer,'primary_ndvi'].to_numpy()
def pred(mask,h):
 obs=known & ~mask; out=np.full(mask.sum(),np.nan); qix=np.flatnonzero(mask); doy=ref.doy.to_numpy(); ids=ref[ID].astype(str).to_numpy(); yrs=ref.year.to_numpy()
 for j,qi in enumerate(qix):
  sel=np.flatnonzero(obs & (ids==ids[qi]) & (yrs==yrs[qi]))
  if len(sel)==0: sel=np.flatnonzero(obs & (ids==ids[qi]))
  if len(sel)==0: out[j]=np.nanmean(y[obs]); continue
  dd=np.abs(doy[sel]-doy[qi]); dd=np.minimum(dd,366-dd); u=dd/(h*366); w=np.exp(-0.5*u*u)
  out[j]=np.sum(w*y[sel])/np.sum(w)
 return out
rows=[]
for h in [.03,.05,.1,.2]:
 po=pred(outer,h); pg=pred(actual,h); rm=lambda a,b:float(np.sqrt(np.nanmean((a-b)**2))); rows.append(dict(h=h,outer_rmse=rm(yo,po),released_rmse=rm(yg,pg),delta_outer=rm(yo,po)-0.0661))
pd.DataFrame(rows).to_csv(R/'gaussian_kernel_probe_20260905.csv',index=False); (R/'gaussian_kernel_probe_20260905_report.md').write_text('# Gaussian kernel probe\n\nSame AOI/year temporal Nadaraya-Watson, K=exp(-0.5(u/h)^2), h as year fraction. Leakage-safe outer mask and released old GT diagnostic.\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nNo candidate materialized; upload not performed.\n',encoding='utf8'); print(pd.DataFrame(rows).to_string(index=False))

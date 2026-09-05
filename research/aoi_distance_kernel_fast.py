import pandas as pd,numpy as np
from pathlib import Path
R=Path('research');D=Path(r'C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904');tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date']);pr=pd.read_csv(D/'private_features.csv',parse_dates=['date']);ref=pd.concat([tr.assign(_origin='train'),pr.assign(_origin='test')],ignore_index=True,sort=False);ref['year']=ref.date.dt.year;ref['doy']=ref.date.dt.dayofyear;ref['aoi_num']=ref.anon_polygon_id.str.extract(r'(\d+)$')[0].astype(float);ref['ds']=ref.date.dt.strftime('%Y-%m-%d');ref['crop']=ref.crop_type.fillna('NA').astype(str);y=ref.primary_ndvi.to_numpy(float);act=(ref._origin=='test')&ref.is_synthetic_gap.fillna(False);known=np.isfinite(y)&~act
import sys;sys.path.insert(0,str(R));import hgb_54_safe_probe_20260905_1810 as H
outer=H.stratified_mask(ref,known&(ref._origin=='test').to_numpy(),20260905,.15);keys=ref.loc[act,['anon_polygon_id','date']];gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=['date']);yg=keys.merge(gt,on=['anon_polygon_id','date']).primary_ndvi_true.to_numpy();yo=ref.loc[outer,'primary_ndvi'].to_numpy()
def pred(mask,r,k):
 obs=known&~mask; groups={}
 for key,ix in ref.loc[obs].groupby(['ds','crop']).groups.items(): groups[key]=np.asarray(ix)
 out=[]
 for qi in np.flatnonzero(mask):
  sel=groups.get((ref.ds.iloc[qi],ref.crop.iloc[qi]),np.array([],int));
  if len(sel)==0: out.append(np.nanmean(y[obs]));continue
  dd=np.abs(ref.aoi_num.iloc[sel].to_numpy()-ref.aoi_num.iloc[qi]);u=dd/r;w=np.maximum(0,1-u) if k=='tri' else np.exp(-.5*(dd/(r*.5))**2);out.append(np.sum(w*y[sel])/np.sum(w))
 return np.array(out)
rows=[];rm=lambda a,b:float(np.sqrt(np.nanmean((a-b)**2)))
for k in ['tri','gauss']:
 for r in [2,4,8,12]:
  po=pred(outer,r,k);pg=pred(act,r,k);rows.append(dict(kernel=k,radius=r,outer_rmse=rm(yo,po),released_rmse=rm(yg,pg)));print(rows[-1])
pd.DataFrame(rows).to_csv(R/'aoi_distance_kernel_probe_20260905.csv',index=False);(R/'aoi_distance_kernel_probe_20260905_report.md').write_text('# AOI distance kernel probe\n\nLeakage-safe same-date/crop smoothing by numeric AOI-ID radius; triangular/Gaussian.\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nStandalone results only; no candidate/upload.\n',encoding='utf8')

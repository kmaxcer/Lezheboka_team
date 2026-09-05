import pandas as pd,numpy as np
from pathlib import Path
R=Path('research');D=Path(r'C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904');tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date']);pr=pd.read_csv(D/'private_features.csv',parse_dates=['date']);ref=pd.concat([tr.assign(_origin='train'),pr.assign(_origin='test')],ignore_index=True,sort=False);ref['year']=ref.date.dt.year;ref['doy']=ref.date.dt.dayofyear;ref['aoi_num']=ref.anon_polygon_id.str.extract(r'(\d+)$')[0].astype(float);y=ref.primary_ndvi.to_numpy(float);act=(ref._origin=='test')&ref.is_synthetic_gap.fillna(False);known=np.isfinite(y)&~act
# deterministic outer same as hgb script
import sys;sys.path.insert(0,str(R));import hgb_54_safe_probe_20260905_1810 as H
outer=H.stratified_mask(ref,known&(ref._origin=='test').to_numpy(),20260905,.15);basep=pd.read_csv('outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_cal0148_20260905_submission.csv',parse_dates=['date']);gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=['date']);keys=ref.loc[act,['anon_polygon_id','date']];yg=keys.merge(gt,on=['anon_polygon_id','date']).primary_ndvi_true.to_numpy();bg=basep.merge(keys,on=['anon_polygon_id','date']).primary_ndvi_pred.to_numpy();yo=ref.loc[outer,'primary_ndvi'].to_numpy()
def pred(mask,r,h,window):
 obs=known&~mask; qix=np.flatnonzero(mask); ids=ref.aoi_num.to_numpy();ds=ref.date.dt.strftime('%Y-%m-%d').to_numpy();crop=ref.crop_type.fillna('NA').astype(str).to_numpy();out=[]
 for qi in qix:
  d=np.abs(ids-ids[qi]);sel=np.flatnonzero(obs&(d<=r)&(ds==ds[qi])&(crop==crop[qi]));
  if len(sel)==0: sel=np.flatnonzero(obs&(d<=r)&(ds==ds[qi]))
  if len(sel)==0: sel=np.flatnonzero(obs&(d<=r))
  dd=np.abs(ids[sel]-ids[qi]);u=dd/max(r,1);w=np.maximum(0,1-u) if h=='tri' else np.exp(-.5*(dd/max(r*.5,1))**2);out.append(np.sum(w*y[sel])/np.sum(w))
 return np.array(out)
rows=[];rm=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
for typ in ['tri','gauss']:
 for r in [2,4,8,12]:
  po=pred(outer,r,typ,None);pg=pred(act,r,typ,None);rows.append(dict(kernel=typ,radius=r,outer_rmse=rm(yo,po),released_rmse=rm(yg,pg)));print(rows[-1])
pd.DataFrame(rows).to_csv(R/'aoi_distance_kernel_probe_20260905.csv',index=False);(R/'aoi_distance_kernel_probe_20260905_report.md').write_text('# AOI distance kernel probe\n\nSame-date/crop local smoothing by numeric anonymized AOI-ID radius; triangular/Gaussian weights.\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nStandalone kernels only; no overlay candidate because robust base outer predictions are unavailable and standalone is diagnostic. Upload not performed.\n',encoding='utf8')

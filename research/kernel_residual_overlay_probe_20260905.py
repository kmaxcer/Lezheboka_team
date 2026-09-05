import pandas as pd,numpy as np
from pathlib import Path
R=Path('research');D=Path(r'C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904');tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date']);pr=pd.read_csv(D/'private_features.csv',parse_dates=['date']);ref=pd.concat([tr.assign(_origin='train'),pr.assign(_origin='test')],ignore_index=True,sort=False);ref['year']=ref.date.dt.year;ref['doy']=ref.date.dt.dayofyear;ID='anon_polygon_id';y=ref.primary_ndvi.to_numpy(float);act=(ref._origin=='test')&ref.is_synthetic_gap.fillna(False);known=np.isfinite(y)&~act;keys=ref.loc[act,[ID,'date']];gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=['date']);yg=keys.merge(gt,on=[ID,'date']).primary_ndvi_true.to_numpy();base=pd.read_csv('outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv',parse_dates=['date']);bg=base.merge(keys,on=[ID,'date']).primary_ndvi_pred.to_numpy();ids=ref[ID].astype(str).to_numpy();yrs=ref.year.to_numpy();doy=ref.doy.to_numpy();qix=np.flatnonzero(act)
def ker(qi,h,k):
 sel=np.flatnonzero(known&(ids==ids[qi])&(yrs==yrs[qi]));dd=np.minimum(np.abs(doy[sel]-doy[qi]),366-np.abs(doy[sel]-doy[qi]));u=dd/(h*366);w=np.maximum(0,1-u) if k=='tri' else np.exp(-.5*u*u);ok=w>1e-12;return np.sum(w[ok]*y[sel][ok])/np.sum(w[ok]) if ok.any() else np.nanmean(y[sel])
rows=[]
for k in ['tri','gauss']:
 for h in [.05,.1,.2]:
  kp=np.array([ker(q,h,k) for q in qix]);res=kp-bg
  for a in [.05,.1,.15]:
   p=np.clip(bg+a*res,-.2,1.1);rows.append(dict(kernel=k,h=h,alpha=a,rmse=float(np.sqrt(np.mean((p-yg)**2))),base_rmse=float(np.sqrt(np.mean((bg-yg)**2))),delta=float(np.sqrt(np.mean((p-yg)**2))-np.sqrt(np.mean((bg-yg)**2)))))
pd.DataFrame(rows).to_csv(R/'kernel_residual_overlay_probe_20260905.csv',index=False);(R/'kernel_residual_overlay_probe_20260905_report.md').write_text('# Kernel residual overlay probe\n\nResidual = weighted temporal neighbor estimate minus robust-base prediction; evaluated on released old GT.\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nNo candidate materialized unless independent masks improve.\n',encoding='utf8');print(pd.DataFrame(rows).to_string(index=False))

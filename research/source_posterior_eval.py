from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(ROOT/'src'));from validate import make_fold;from infer import _prepare,_mode_posteriors,_query_posterior
def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date']);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date']); rows=[]
 for year in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,year); d=_prepare(f); hidden=f.is_synthetic_gap.to_numpy(bool); known=np.isfinite(d.primary_ndvi.to_numpy(float))&~hidden; a,c,g,dt=_mode_posteriors(d,known)
  orig=tr.set_index(['anon_polygon_id','date']);q=f.loc[hidden].copy().reset_index(drop=True);oo=orig.reindex(pd.MultiIndex.from_frame(q[['anon_polygon_id','date']])); true=np.select([oo.s2_ndvi.notna(),oo.landsat_ndvi.notna(),oo.modis_ndvi.notna()],['s2','landsat','mod'],'none')
  for w in [0,.25,.5,.75,1,1.5,2]:
   pred=[]; ent=[]
   for i in np.flatnonzero(hidden):
    p=_query_posterior(d,int(i),a,c,g,dt,date_weight=w);pred.append(('s2','landsat','modis')[int(np.argmax(p))]);ent.append(-np.sum(p*np.log(p)))
   pred=np.array(pred); rows.append({'year':year,'w':w,'acc':float(np.mean(pred==true)),'acc_s2':float(np.mean(pred[true=='s2']=='s2')),'acc_l8':float(np.mean(pred[true=='l8']=='l8')),'acc_mod':float(np.mean(pred[true=='mod']=='mod')),'entropy':np.mean(ent)})
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()

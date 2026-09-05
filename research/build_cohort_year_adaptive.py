"""Emit an additional year-smoothed cohort route from validated components."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
KEY=['anon_polygon_id','date']

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def read(p):
 z=pd.read_csv(p,parse_dates=['date']);
 if list(z.columns)!=KEY+['primary_ndvi_pred'] or z.duplicated(KEY).any() or not np.isfinite(z.primary_ndvi_pred).all(): raise ValueError(str(p))
 return z

def main():
 private=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
 train=pd.read_csv(DATA/'train_dataset.csv',usecols=['anon_polygon_id'])
 hidden=private.is_synthetic_gap.fillna(False).astype(bool)
 keys=private.loc[hidden,KEY].reset_index(drop=True)
 shared=set(train.anon_polygon_id.unique())
 cohort=np.where(keys.anon_polygon_id.isin(shared),'shared','new')
 year=keys.date.dt.year.to_numpy()
 base=keys.merge(read(OUT/'model_dani_lag40_peer10_a350_b200_submission.csv'),on=KEY,validate='one_to_one').primary_ndvi_pred.to_numpy()
 ext=keys.merge(read(OUT/'model_dani_extended_hgb_wide.csv'),on=KEY,validate='one_to_one').primary_ndvi_pred.to_numpy()
 # Per-year optimum from the saved holdout, shrunk halfway toward the
 # stable 0.40 historical weight to reduce selection noise.
 q=pd.read_csv(ROOT/'research/private_cohort_blend_holdout_predictions.csv')
 wh={}
 for y in range(2010,2025):
  g=q[(q.cohort=='new')&(q.year==y)]
  grid=np.linspace(0,1,101)
  rm=[np.mean(((1-w)*g.joint40+w*g.extended-g.truth)**2)**.5 for w in grid]
  wh[y]=float(.5*.4+.5*grid[int(np.argmin(rm))])
 rw=np.where((cohort=='new')&(year<2025),np.array([wh.get(int(y),.4) for y in year]),np.where((cohort=='new')&(year>=2025),.4,.3))
 pred=np.clip((1-rw)*base+rw*ext,-.2,1.1)
 out=keys.copy(); out['primary_ndvi_pred']=pred
 path=OUT/'model_dani_lag40_peer10_cohort_year_adaptive_wide_submission.csv'; out.to_csv(path,index=False,float_format='%.8f')
 chk=read(path)
 meta={'candidate':path.name,'sha256':sha(path),'base_sha256':sha(OUT/'model_dani_lag40_peer10_a350_b200_submission.csv'),'ext_sha256':sha(OUT/'model_dani_extended_hgb_wide.csv'),'rows':len(chk),'weights_history':wh,'weights_new_2025':.4,'weights_shared_2025':.3,'production_baseline_overwritten':False}
 (OUT/'model_dani_cohort_year_adaptive_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

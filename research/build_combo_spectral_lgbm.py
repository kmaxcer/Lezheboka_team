import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; RES=ROOT/'research'; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
ID,DATE='anon_polygon_id','date'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE]); hidden=pr.is_synthetic_gap.fillna(False).astype(bool); keys=pr.loc[hidden,[ID,DATE]].reset_index(drop=True)
def pred(file,col='primary_ndvi_pred'):
 d=pd.read_csv(OUT/file,parse_dates=[DATE]); d[DATE]=pd.to_datetime(d[DATE]); return keys.merge(d,on=[ID,DATE],how='left',validate='one_to_one')[col].to_numpy(float)
base=pred('model_dani_lag40_peer10_extwide40_v3_30_submission.csv'); lgb=pred('model_dani_lgbm_extended.csv'); sp=pd.read_csv(RES/'spectral_full_predictions_checkpoint.csv',parse_dates=[DATE]); sp[DATE]=pd.to_datetime(sp[DATE]); spec=keys.merge(sp,on=[ID,DATE],how='left',validate='one_to_one').spectral_pred.to_numpy(float)
assert np.isfinite(base).all() and np.isfinite(lgb).all() and np.isfinite(spec).all()
years=keys[DATE].dt.year.to_numpy(int); sw=np.where(years<2025,.4,0.0); lw=.1; p=np.clip((1-sw-lw)*base+sw*spec+lw*lgb,-.2,1.1)
name='model_dani_extwide40_v3_30_spectral40_history_lgbm10_submission.csv'; out=keys.copy(); out['primary_ndvi_pred']=p; path=OUT/name; out.to_csv(path,index=False,float_format='%.8f')
meta={'candidate':name,'rows':len(out),'spectral_weight_history':.4,'lgbm_weight_all':.1,'routing':'spectral history<2025; LGBM all years','base_sha256':sha(OUT/'model_dani_lag40_peer10_extwide40_v3_30_submission.csv'),'lgbm_sha256':sha(OUT/'model_dani_lgbm_extended.csv'),'spectral_checkpoint_sha256':sha(RES/'spectral_full_predictions_checkpoint.csv'),'output_sha256':sha(path),'min':float(p.min()),'max':float(p.max()),'mean':float(p.mean()),'production_baseline_overwritten':False}
(OUT/'model_dani_combo_spectral_lgbm_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2))

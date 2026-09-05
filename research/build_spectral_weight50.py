import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; RES=ROOT/'research'; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
ID,DATE='anon_polygon_id','date'; base_name='model_dani_lag40_peer10_extwide40_v3_30_submission.csv'; sp_name='spectral_full_predictions_checkpoint.csv'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE]); hidden=pr.is_synthetic_gap.fillna(False).astype(bool); keys=pr.loc[hidden,[ID,DATE]].reset_index(drop=True)
b=pd.read_csv(OUT/base_name,parse_dates=[DATE]); b[DATE]=pd.to_datetime(b[DATE]); bp=keys.merge(b,on=[ID,DATE],how='left',validate='one_to_one').primary_ndvi_pred.to_numpy(float)
s=pd.read_csv(RES/sp_name,parse_dates=[DATE]); s[DATE]=pd.to_datetime(s[DATE]); sp=keys.merge(s,on=[ID,DATE],how='left',validate='one_to_one').spectral_pred.to_numpy(float)
assert np.isfinite(bp).all() and np.isfinite(sp).all()
years=keys[DATE].dt.year.to_numpy(int); w=np.where(years<2025,.5,0.0); p=np.clip((1-w)*bp+w*sp,-.2,1.1)
out=keys.copy(); out['primary_ndvi_pred']=p; name='model_dani_lag40_peer10_extwide40_v3_30_spectral50_historyonly_submission.csv'; path=OUT/name; out.to_csv(path,index=False,float_format='%.8f')
meta={'candidate':name,'rows':len(out),'spectral_weight':.5,'routing':'history_only;zero_all2025','base_component':base_name,'base_sha256':sha(OUT/base_name),'spectral_checkpoint_sha256':sha(RES/sp_name),'output_sha256':sha(path),'min':float(p.min()),'max':float(p.max()),'mean':float(p.mean()),'history_rows':int((years<2025).sum()),'all2025_rows':int((years>=2025).sum()),'production_baseline_overwritten':False}
(OUT/'model_dani_spectral_weight50_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2))

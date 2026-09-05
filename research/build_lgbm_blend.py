"""Create optional low-weight LightGBM blends; baseline files untouched."""
from pathlib import Path
import hashlib, json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 l=pd.read_csv(OUT/'model_dani_lgbm_extended.csv',parse_dates=['date'])
 b=pd.read_csv(OUT/'model_dani_lag40_peer10_extwide40_v3_30_submission.csv',parse_dates=['date'])
 x=b.merge(l,on=['anon_polygon_id','date'],how='left',validate='one_to_one',suffixes=('','_lgbm'))
 for w in (.05,.10,.15,.20):
  y=b[['anon_polygon_id','date']].copy(); y['primary_ndvi_pred']=(1-w)*x.primary_ndvi_pred+w*x.primary_ndvi_pred_lgbm
  p=OUT/f'model_dani_lag40_peer10_extwide40_v3_30_lgbm{int(w*100):02d}_submission.csv'; y.to_csv(p,index=False,float_format='%.8f'); print(p.name,sha(p))
if __name__=='__main__': main()

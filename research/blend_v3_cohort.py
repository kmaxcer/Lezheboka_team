"""Create cohort-aware blends with the v3 private component, research only."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); OUT=ROOT/"outputs"
def sha(p):
 h=hashlib.sha256();
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def main():
 pr=pd.read_csv(DATA/"private_features.csv",parse_dates=["date"],low_memory=False); base=pd.read_csv(OUT/"model_dani_lag40_peer10_a350_b200_submission.csv"); v3=pd.read_csv(OUT/"model_dani_extended_hgb_v3_wide.csv")
 key=["anon_polygon_id","date"]; base.date=pd.to_datetime(base.date); v3.date=pd.to_datetime(v3.date); pr.date=pd.to_datetime(pr.date)
 q=base.merge(v3,on=key,how="inner",validate="one_to_one",suffixes=("_base","_v3")); q=q.merge(pr[key],on=key,how="left",validate="one_to_one"); q["year"]=q.date.dt.year
 train_ids=set(pd.read_csv(DATA/"train_dataset.csv",usecols=["anon_polygon_id"],low_memory=False).anon_polygon_id.astype(str)); q["shared"]=q.anon_polygon_id.astype(str).isin(train_ids); q["new2025"]=(q.year==2025)&~q.shared; q["w"]=np.where(q.new2025,0.0,np.where(q.shared,.30,.35))
 # Safe variants around the intended routing; all remain separate from baseline.
 for w_hist,w_shared in [(0.20,0.20),(0.30,0.30),(0.40,0.30),(0.40,0.40)]:
  ww=np.where(q.new2025,0.0,np.where(q.shared,w_shared,w_hist)); p=(1-ww)*q.primary_ndvi_pred_base+ww*q.primary_ndvi_pred_v3
  o=q[key].copy(); o["primary_ndvi_pred"]=np.clip(p,-.2,1.1); tag=f"v3_cohort_h{int(w_hist*100):02d}_s{int(w_shared*100):02d}_new2025_00"; fn=OUT/f"{tag}_submission.csv"; o.to_csv(fn,index=False,float_format="%.8f")
  meta={"candidate":fn.name,"history_v3_weight":w_hist,"shared_2025_v3_weight":w_shared,"new_2025_v3_weight":0.0,"rows":len(o),"shared_rows":int(q.shared.sum()),"new2025_rows":int(q.new2025.sum()),"sha256":sha(fn),"baseline_sha256":sha(OUT/"model_dani_lag40_peer10_a350_b200_submission.csv"),"v3_sha256":sha(OUT/"model_dani_extended_hgb_v3_wide.csv"),"production_baseline_overwritten":False}; (OUT/f"{tag}_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"); print(meta)
if __name__=="__main__": main()

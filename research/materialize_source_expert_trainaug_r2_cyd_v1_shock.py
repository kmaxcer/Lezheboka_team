"""Overlay observable train-augmented 24-day crop shock on r2 candidate.

This produces separate exploratory `.15` and `.175` outputs with a unique
suffix.  The shock profile is built from train + visible private rows only;
actual hidden gaps are masked before feature construction.  Existing files
are never overwritten.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; O=ROOT/"outputs"; D=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); ARCH=ROOT/"_archive_inspect"/"agropulse_max_score"/"data"
sys.path.insert(0,str(R)); from shock_bin_sweep_v1 import _features  # noqa: E402
ID,DATE,GAP="anon_polygon_id","date","is_synthetic_gap"

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def main():
    base=O/"model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_submission.csv"; tr=pd.read_csv(D/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/"private_features.csv",parse_dates=[DATE],low_memory=False); actual=pr[GAP].fillna(False).astype(bool).to_numpy(); combo=pd.concat([tr,pr],ignore_index=True,sort=False); combo["_truth"]=pd.to_numeric(combo.primary_ndvi,errors="coerce"); mask=np.r_[np.zeros(len(tr),bool),actual]; ft=_features(combo,mask,24); sm=ft.set_index([ID,DATE])["crop_shock"]; keys=pr.loc[actual,[ID,DATE]].copy(); keys[DATE]=pd.to_datetime(keys[DATE]); bdf=pd.read_csv(base,parse_dates=[DATE],low_memory=False).set_index([ID,DATE]); bi=pd.MultiIndex.from_frame(keys); b=bdf.loc[bi,"primary_ndvi_pred"].to_numpy(float); shock=np.asarray([sm.get(k,np.nan) for k in bi],float); finite=np.isfinite(shock); rec=[]; stem="model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_shock"
    for label,a in [("015",.15),("0175",.175)]:
        p=np.clip(b+a*np.nan_to_num(shock,nan=0.),-.2,1.1); out=keys.copy(); out["primary_ndvi_pred"]=p; out[DATE]=pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d"); path=O/f"{stem}{label}_submission.csv";
        if path.exists(): raise RuntimeError(f"refuse overwrite {path.name}")
        out.to_csv(path,index=False,float_format="%.9f"); chk=pd.read_csv(path); ok=(len(chk)==3112 and list(chk.columns)==[ID,DATE,"primary_ndvi_pred"] and chk[[ID,DATE]].drop_duplicates().shape[0]==3112 and np.isfinite(chk.primary_ndvi_pred).all()); m={"candidate":path.name,"formula":f"base=model_dani_source_expert_route_v2_trainaug_r2_cyd_v1; add {a}*observable train+visible-private 24-day date+crop shock","rows":len(out),"finite":bool(ok),"unique_keys":int(chk[[ID,DATE]].drop_duplicates().shape[0]),"shock_finite":int(finite.sum()),"shock_mean":float(np.nanmean(shock)),"shock_std":float(np.nanstd(shock)),"base_sha256":sha(base),"candidate_sha256":sha(path),"production_baseline_overwritten":False,"no_upload":True}; (path.with_name(path.stem+"_metadata.json")).write_text(json.dumps(m,indent=2),encoding="utf-8"); rec.append(m)
    (R/"source_expert_trainaug_r2_cyd_v1_shock_report.md").write_text("# Train-augmented r2 source-route shock overlays\n\n"+json.dumps(rec,indent=2)+"\n",encoding="utf-8"); print(json.dumps(rec,indent=2))

if __name__=="__main__": main()

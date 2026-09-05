"""Materialize train-augmented fixed-r2 source expert candidates.

All output names carry the ``trainaug_r2_cyd_v1`` suffix to avoid touching
existing candidates.  The source route is built from train + visible private
same-date/same-crop peers; alpha is the four-mask validated cohort/year/
distance policy.  Optional shock overlays are diagnostics, not submissions.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; O=ROOT/"outputs"; D=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); ARCH=ROOT/"_archive_inspect"/"agropulse_max_score"/"data"
sys.path.insert(0,str(R))
import source_expert_q1 as q1  # noqa: E402
import source_expert_route_v2 as rv  # noqa: E402
from overnight_source_eval import _predict_matrix  # noqa: E402
from source_schedule_route_probe import classify  # noqa: E402

ID,DATE,TARGET,GAP="anon_polygon_id","date","primary_ndvi","is_synthetic_gap"

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def main():
    t0=time.time(); tr=pd.read_csv(D/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/"private_features.csv",parse_dates=[DATE],low_memory=False); tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool); actual=pr[GAP].to_numpy(bool)
    print("fit source experts actual",flush=True); ref,gref,sref,pm,gaps=q1._make_masked_ref(tr,pr,np.zeros(len(pr),bool)); _,ep,_=q1._fit_experts(ref,gref,sref); qref=ref.loc[gref,[ID,DATE]].copy().reset_index(drop=True); qref[["e_s2","e_landsat","e_modis"]]=ep; keys=pr.loc[actual,[ID,DATE,"crop_type"]].copy().reset_index(drop=True); keys[DATE]=pd.to_datetime(keys[DATE]); q=keys.merge(qref,on=[ID,DATE],how="left",validate="one_to_one"); E=q[["e_s2","e_landsat","e_modis"]].to_numpy(float); 
    if not np.isfinite(E).all(): raise RuntimeError("expert alignment")
    # Schedule posterior fallback from masked private; same-date/crop modes
    # are recomputed with train + visible private by the classifier.
    pmatrix,_=_predict_matrix(pm,train=tr,family="base",k=8,degree=1,bin_days=30,date_weight=1.0); pmap=pmatrix.set_index("row_index"); qi=np.flatnonzero(actual); post=np.column_stack([[pmap.loc[i,c] if i in pmap.index else 1/3 for i in qi] for c in ("p_s2","p_landsat","p_modis")]); post=np.where(np.isfinite(post),post,1/3); post/=post.sum(1,keepdims=True); postmode=np.argmax(post,axis=1).astype(int)
    sched=classify(tr,pr,actual); raw=sched.sp_crop_2.to_numpy(int).copy(); n2=sched.sp_crop_2_n.to_numpy(int); n8=sched.sp_crop_8_n.to_numpy(int); raw[raw<0]=postmode[raw<0]; route=raw; psrc=E[np.arange(len(E)),route]
    yr=keys[DATE].dt.year.to_numpy(int); shared=keys[ID].astype(str).isin(set(tr[ID].astype(str))).to_numpy(bool); new=~shared; near=n2>0; mid=(~near)&(n8>0); alpha=np.where(near,.50,np.where(mid,.40,.30)); alpha=np.where(new&(yr==2025),.60,alpha); alpha=np.where(shared&(yr==2025),.35,alpha)
    base= pd.read_csv(O/"model_dani_extwide40_v3_30_peerblend12_history_submission.csv",parse_dates=[DATE],low_memory=False).set_index([ID,DATE])["primary_ndvi_pred"]; b=np.asarray([base.get((i,d),np.nan) for i,d in keys[[ID,DATE]].itertuples(index=False,name=None)],float); 
    if not np.isfinite(b).all(): raise RuntimeError("baseline alignment")
    pred=np.clip((1-alpha)*b+alpha*psrc,-.2,1.1); stem="model_dani_source_expert_route_v2_trainaug_r2_cyd_v1"; path=O/(stem+"_submission.csv");
    if path.exists(): raise RuntimeError(f"refuse overwrite {path.name}")
    out=keys[[ID,DATE]].copy(); out["primary_ndvi_pred"]=pred; out[DATE]=pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d"); out.to_csv(path,index=False,float_format="%.9f")
    ok=(len(out)==3112 and out[[ID,DATE]].drop_duplicates().shape[0]==len(out) and np.isfinite(pred).all() and list(out.columns)==[ID,DATE,"primary_ndvi_pred"])
    side=keys[[ID,DATE,"crop_type"]].copy(); side["route_source_index"]=route; side["route_peer_n_r2"]=n2; side["route_peer_n_r8"]=n8; side["alpha"]=alpha; side["fallback_schedule"]=(sched.sp_crop_2.to_numpy(int)<0); side.to_csv(R/(stem+"_sidecar.csv"),index=False,float_format="%.9f")
    meta={"candidate":path.name,"formula":"baseline=history_peer12; source=three OOF HGB experts routed by train+visible-private same-date/same-crop fixed radius2 mode, fallback observable schedule posterior; alpha=.50 if r2 peer, .40 if r3-8 peer, .30 otherwise, override new2025=.60/shared2025=.35","rows":len(out),"finite":bool(ok),"unique_keys":int(out[[ID,DATE]].drop_duplicates().shape[0]),"sha256":sha(path),"actual_gap_rows":int(actual.sum()),"route_r2_peer":int(n2.astype(bool).sum()),"route_r8_peer":int(n8.astype(bool).sum()),"alpha_counts":{str(x):int((alpha==x).sum()) for x in sorted(np.unique(alpha))},"production_baseline_overwritten":False,"no_upload":True,"seconds":round(time.time()-t0,1)}
    (O/(stem+"_metadata.json")).write_text(json.dumps(meta,indent=2),encoding="utf-8")
    report=["# Train-augmented fixed-r2 source expert candidate v1","", "Route uses train + visible private rows only; hidden actual gaps are excluded. Four-mask source-route audit selected fixed r2 with cohort/year/distance alpha policy.","",json.dumps(meta,indent=2),"",f"Candidate: `outputs/{path.name}`", "", "No submission was uploaded; existing outputs were not overwritten."]
    (R/(stem+"_report.md")).write_text("\n".join(report)+"\n",encoding="utf-8"); print(json.dumps(meta,indent=2))

if __name__=="__main__": main()

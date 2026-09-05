"""Materialize a conservative source-route candidate with adaptive alpha.

Policy selected by four independent private-like masks (0, 1, 2, 70404):
alpha=.50 for an observable same-date/same-crop peer within numeric AOI
distance <=2, .40 for distance 3--8, .30 for farther/no peer; override to
.60 on new-AOI 2025 and .35 on shared-AOI 2025.  The route itself remains
crop-aware first-peer mode (n>=1, purity>=.67), then observable schedule mode.
No source labels are read in this inference path.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); R = ROOT/"research"; OUT = ROOT/"outputs"
sys.path.insert(0, str(R))
import source_expert_q1 as q1  # noqa: E402
import source_expert_route_v2 as rv  # noqa: E402
from overnight_source_eval import _predict_matrix  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"

def sha(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    t0=time.time(); tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False); tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool)
    actual=pr[GAP].to_numpy(bool)
    ref,gaps_ref,sref,pm,gaps_pr=q1._make_masked_ref(tr,pr,np.zeros(len(pr),bool))
    # Three source experts use OOF pseudo-gaps; no target/sensor values remain
    # on actual query rows in `ref`.
    _,ep_ref,_=q1._fit_experts(ref,gaps_ref,sref)
    qref=ref.loc[gaps_ref,[ID,DATE]].copy().reset_index(drop=True); qref[["e_s2","e_landsat","e_modis"]]=ep_ref
    qkeys=pr.loc[actual,[ID,DATE,"crop_type"]].copy().reset_index(drop=True); qkeys[DATE]=pd.to_datetime(qkeys[DATE])
    q=qkeys.merge(qref,on=[ID,DATE],how="left",validate="one_to_one")
    if q[["e_s2","e_landsat","e_modis"]].isna().any().any(): raise RuntimeError("expert key alignment failed")
    # Observable schedule posterior and crop-aware neighboring source counts.
    pmatrix,_=_predict_matrix(pm,train=tr,family="base",k=8,degree=1,bin_days=30,date_weight=1.0); pmap=pmatrix.set_index("row_index"); qi=np.flatnonzero(actual)
    post=np.column_stack([[pmap.loc[i,c] if i in pmap.index else 1/3 for i in qi] for c in ("p_s2","p_landsat","p_modis")]); post=np.where(np.isfinite(post),post,1/3); post/=post.sum(1,keepdims=True)
    cc,ac,near=rv._neighbor_counts(pm,gaps_pr,qkeys); routes=rv._route_variants(cc,ac,post); route=routes["crop_hier_n1_p67"].astype(int)
    E=q[["e_s2","e_landsat","e_modis"]].to_numpy(float); psrc=E[np.arange(len(E)),route]
    # Existing strong baseline is read but never overwritten.
    basepath=OUT/"model_dani_extwide40_v3_30_peerblend12_history_submission.csv"; base=pd.read_csv(basepath,parse_dates=[DATE],low_memory=False); bm=base.set_index([ID,DATE])["primary_ndvi_pred"]
    b=np.asarray([bm.get((i,d),np.nan) for i,d in q[[ID,DATE]].itertuples(index=False,name=None)],float)
    if not np.isfinite(b).all(): raise RuntimeError("baseline key alignment failed")
    yr=q[DATE].dt.year.to_numpy(int); shared=q[ID].astype(str).isin(set(tr[ID].astype(str))).to_numpy(bool); new=~shared; near2=np.isfinite(near)&(near<=2); mid=np.isfinite(near)&(near>2)&(near<=8)
    alpha=np.where(near2,.50,np.where(mid,.40,.30)); alpha=np.where(new&(yr==2025),.60,alpha); alpha=np.where(shared&(yr==2025),.35,alpha)
    pred=np.clip((1-alpha)*b+alpha*psrc,-.2,1.1)
    out=q[[ID,DATE]].copy(); out["primary_ndvi_pred"]=pred; out[DATE]=pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d")
    path=OUT/"model_dani_source_expert_route_v2_cohort_year_dist_submission.csv"; out.to_csv(path,index=False,float_format="%.9f")
    if len(out)!=3112 or out[[ID,DATE]].drop_duplicates().shape[0]!=len(out) or not np.isfinite(pred).all(): raise RuntimeError("candidate integrity failure")
    # Observable route sidecar (source labels deliberately absent).
    side=q[[ID,DATE,"crop_type"]].copy(); side["near_dist"]=near; side["route_source_index"]=route; side["alpha"]=alpha; side["route_used_peer"]=(cc[:,0].sum(1)>0); side["route_peer_purity"]=cc[:,0].max(1)/np.maximum(1.,cc[:,0].sum(1)); side.to_csv(R/"source_expert_route_v2_cohort_year_dist_sidecar.csv",index=False,float_format="%.9f")
    counts={"rows":int(len(out)),"new2025":int((new&(yr==2025)).sum()),"shared2025":int((shared&(yr==2025)).sum()),"near_le2":int(near2.sum()),"mid_3_8":int(mid.sum()),"far_or_none":int((~near2&~mid).sum()),"peer_any_r1":int((cc[:,0].sum(1)>0).sum()),"peer_any_r2":int((cc[:,1].sum(1)>0).sum())}
    meta={"candidate":path.name,"formula":"baseline=history_peer12; source=HGB(S2/Landsat/MODIS) routed by observable same-date same-crop peer mode (n>=1,purity>=0.67), fallback schedule posterior mode; alpha=.50 if near<=2, .40 if 3..8, .30 otherwise; new2025 override=.60, shared2025 override=.35","rows":len(out),"finite":bool(np.isfinite(pred).all()),"unique_keys":int(out[[ID,DATE]].drop_duplicates().shape[0]),"sha256":sha(path),"observable_coverage":counts,"production_baseline_overwritten":False,"seconds":round(time.time()-t0,1)}
    (path.with_name(path.stem+"_metadata.json")).write_text(json.dumps(meta,indent=2),encoding="utf-8")
    report=["# Source-expert route v2 cohort/year/distance candidate","", "Four-mask audit (seeds 0,1,2,70404) selected the fixed observable policy:", "- crop-aware same-date route (`n>=1`, purity `>=0.67`), fallback schedule posterior mode;", "- alpha `.50/.40/.30` for near (`<=2`), mid (`3--8`), far/no-peer;", "- new-AOI 2025 override `.60`, shared-AOI 2025 override `.35`.", "", "## Actual-gap observable coverage", "", json.dumps(counts,indent=2), "", f"Candidate: `{path.relative_to(ROOT).as_posix()}`", f"SHA256: `{sha(path)}`", f"Rows: {len(out)}; finite: {bool(np.isfinite(pred).all())}; unique keys: {out[[ID,DATE]].drop_duplicates().shape[0]}", "", "The inference path never reads true source labels or hidden targets; sidecar stores only observable route diagnostics.", "Existing outputs were not overwritten."]
    (R/"source_expert_route_v2_cohort_year_dist_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))

if __name__ == "__main__": main()

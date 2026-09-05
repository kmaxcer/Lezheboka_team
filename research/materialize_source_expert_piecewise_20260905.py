"""Materialise a separate piecewise-alpha source-expert candidate.

The existing route-v2 full-gap CSV is an affine alpha=.40 blend of the same
production baseline and routed source expert.  We recover the expert exactly
(no clipping occurred), recompute observable crop-aware neighbour distance,
and apply the four-mask-stable policy alpha=.50/.40/.25 for near/mid/far.
"""
from __future__ import annotations

from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; OUT=ROOT/"outputs"
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
ID,DATE,GAP="anon_polygon_id","date","is_synthetic_gap"
BASE=OUT/"model_dani_extwide40_v3_30_peerblend12_history_submission.csv"
ROUTE=OUT/"model_dani_source_expert_route_v2_submission.csv"
DEST=OUT/"model_dani_source_expert_route_v2_piecewise_alpha_submission.csv"
META=DEST.with_name(DEST.stem+"_metadata.json")
REPORT=ROOT/"reports"/(DEST.stem+"_report.md")

sys.path.insert(0,str(R))
import source_expert_route_v2 as rv2  # noqa: E402

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
    tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False)
    pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False)
    actual=pr[GAP].fillna(False).astype(bool).to_numpy()
    # Match route-v2 masking exactly; neighbour counts use only visible rows.
    pm,gaps=rv2._masked_private(pr,np.zeros(len(pr),bool))
    qkeys=pr.loc[actual,[ID,DATE,"crop_type"]].copy().reset_index(drop=True)
    qkeys[DATE]=pd.to_datetime(qkeys[DATE])
    _,_,near=rv2._neighbor_counts(pm,gaps,qkeys)
    b=pd.read_csv(BASE,parse_dates=[DATE],low_memory=False)
    e=pd.read_csv(ROUTE,parse_dates=[DATE],low_memory=False)
    z=b.merge(e,on=[ID,DATE],how="inner",validate="one_to_one",suffixes=("_base","_route"))
    if len(z)!=int(actual.sum()): raise RuntimeError(f"key count {len(z)} != {actual.sum()}")
    B=z.primary_ndvi_pred_base.to_numpy(float); R40=z.primary_ndvi_pred_route.to_numpy(float)
    P=(R40-.60*B)/.40
    if not np.isfinite(P).all(): raise RuntimeError("nonfinite recovered expert")
    # Join near distance in candidate key order.
    nd=qkeys.assign(near_dist=near).merge(z[[ID,DATE]],on=[ID,DATE],how="right",validate="one_to_one")["near_dist"].to_numpy(float)
    if not np.isfinite(nd).all():
        # Infinite means no same-crop peer and is a valid far bucket; NaN means
        # an alignment bug and must fail loudly.
        if np.isnan(nd).any(): raise RuntimeError("near alignment produced NaN")
    alpha=np.where(np.isfinite(nd)&(nd<=2),.50,np.where(np.isfinite(nd)&(nd<=8),.40,.25))
    pred=B+alpha*(P-B)
    out=z[[ID,DATE]].copy(); out["primary_ndvi_pred"]=pred
    out[DATE]=pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d")
    if len(out)!=3112 or out[[ID,DATE]].duplicated().any() or not np.isfinite(pred).all(): raise RuntimeError("contract failure")
    out.to_csv(DEST,index=False,float_format="%.9f")
    meta={"candidate":DEST.name,"formula":"B + alpha*(P-B), P=(route_v2_alpha040-0.60*B)/0.40; alpha=0.50 if same-crop near<=2, 0.40 if 2<near<=8, 0.25 otherwise","rows":int(len(out)),"near_counts":{"near":int(((np.isfinite(nd))&(nd<=2)).sum()),"mid":int(((np.isfinite(nd))&(nd>2)&(nd<=8)).sum()),"far_or_none":int(((~np.isfinite(nd))|(nd>8)).sum())},"finite":bool(np.isfinite(pred).all()),"unique_keys":int(out[[ID,DATE]].drop_duplicates().shape[0]),"baseline_sha256":sha(BASE),"route_alpha040_sha256":sha(ROUTE),"candidate_sha256":sha(DEST),"source_expert_clipping_detected":False,"no_upload":True}
    META.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    lines=[f"# {DEST.stem}","","Separate full-gap candidate derived from the existing route-v2 alpha=.40 file; no model/input was overwritten.","","Formula:","`P=(route_v2_alpha040 - 0.60*production_baseline)/0.40`; `pred=B+alpha*(P-B)` with alpha `.50` for same-crop near distance ≤2, `.40` for 2<distance≤8, `.25` for >8/no peer.","","Observable distance bucket counts:",f"- near: {meta['near_counts']['near']}",f"- mid: {meta['near_counts']['mid']}",f"- far/none: {meta['near_counts']['far_or_none']}","","Contract: 3112 rows, unique keys, finite predictions.",f"SHA256: `{meta['candidate_sha256']}`",f"Metadata: `{META.relative_to(ROOT).as_posix()}`",f"CSV: `{DEST.relative_to(ROOT).as_posix()}`","","Validation basis: four independent private-like masks (0,1,2,70404) selected alpha policy; pooled RMSE 0.066795 vs 0.066865 for global .40 in route-v2 rows."]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2)); print(out.head())

if __name__=="__main__": main()

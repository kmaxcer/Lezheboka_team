"""Materialise the best seasonal local-peer residual sweep variant.

Variant: 16-day AOI seasonal profile, same-date/same-crop peers within numeric
AOI-ID radius 4, uniform peer residual mean.  Four-mask LOO selects alpha near
0.25; a conservative fixed .20 companion is also written.  Existing files
are never overwritten.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, time
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; O = ROOT / "outputs"
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(R))
from teammate_sweep_postcorr import _mask_private  # noqa: E402
from local_peer_residual_sweep_v1 import feature  # noqa: E402

ID, DATE, GAP = "anon_polygon_id", "date", "is_synthetic_gap"


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def build_actual_feature() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    actual = pr[GAP].fillna(False).astype(bool).to_numpy()
    tr0 = tr.copy(); tr0["_truth"] = pd.to_numeric(tr0.primary_ndvi, errors="coerce"); tr0["_hidden"] = False
    f = pr.copy(); f["_truth"] = pd.to_numeric(f.primary_ndvi, errors="coerce"); f["_hidden"] = actual
    combo = pd.concat([tr0, f], ignore_index=True, sort=False)
    known = combo.primary_ndvi.notna().to_numpy(bool) & ~combo._hidden.to_numpy(bool)
    qidx = np.flatnonzero(np.r_[np.zeros(len(tr), bool), actual])
    # The sweep's feature helper is deterministic and uses only `known` rows.
    x = feature(combo, known, qidx, width=16, radius=4, source_level=False, agg="mean")
    keys = f.loc[actual, [ID, DATE]].copy().reset_index(drop=True); keys[DATE] = pd.to_datetime(keys[DATE]); keys["local_feature"] = x
    return keys, tr, actual


def main():
    t0 = time.time(); keys, tr, actual = build_actual_feature()
    basepath = O / "model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_submission.csv"
    base = pd.read_csv(basepath, parse_dates=[DATE], low_memory=False)
    q = keys.merge(base, on=[ID, DATE], how="left", validate="one_to_one")
    if q.primary_ndvi_pred.isna().any(): raise RuntimeError("base alignment failed")
    b = q.primary_ndvi_pred.to_numpy(float); x = q.local_feature.to_numpy(float); x0 = np.nan_to_num(x, nan=0.)
    # LOO alphas from the independent sweep: .255454,.249845,.258840,.236051;
    # rounded robust global choice .25.  Keep .20 as a conservative companion.
    configs = [
        ("model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_submission.csv", .25,
         "base=trainaug_r2_cyd_v1; pred=clip(base+0.25*visible 16-day AOI seasonal residual mean from same-date/same-crop ID-radius4 peers)"),
        ("model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a020_submission.csv", .20,
         "base=trainaug_r2_cyd_v1; pred=clip(base+0.20*visible 16-day AOI seasonal residual mean from same-date/same-crop ID-radius4 peers)"),
    ]
    metas=[]
    for name,a,formula in configs:
        path=O/name
        if path.exists(): raise RuntimeError(f"refusing overwrite {path}")
        pred=np.clip(b+a*x0,-.2,1.1); out=keys[[ID,DATE]].copy();out["primary_ndvi_pred"]=pred;out[DATE]=pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d");out=out[[ID,DATE,"primary_ndvi_pred"]];out.to_csv(path,index=False,float_format="%.9f")
        chk=pd.read_csv(path); ok=len(chk)==int(actual.sum()) and list(chk.columns)==[ID,DATE,"primary_ndvi_pred"] and chk[[ID,DATE]].drop_duplicates().shape[0]==len(chk) and np.isfinite(chk.primary_ndvi_pred).all()
        meta={"candidate":path.name,"formula":formula,"rows":int(len(out)),"finite":bool(ok),"unique_keys":int(chk[[ID,DATE]].drop_duplicates().shape[0]),"alpha":a,"local_feature_finite":int(np.isfinite(x).sum()),"local_feature_coverage":float(np.isfinite(x).mean()),"local_feature_mean":float(np.nanmean(x)),"local_feature_std":float(np.nanstd(x)),"base_sha256":sha(basepath),"candidate_sha256":sha(path),"production_baseline_overwritten":False,"no_upload":True}
        path.with_name(path.stem+"_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8");metas.append(meta)
    # Full slice audit from the sweep for this exact variant, plus a compact
    # alpha summary.  The sweep was run on all four masks and remains labels-
    # confined to research artifacts.
    sr=R/"local_peer_residual_sweep_v1_results.csv"; sa=R/"local_peer_residual_sweep_v1_aggregate.csv"
    rr=pd.read_csv(sr); sel=rr[(rr.width==16)&(rr.radius==4)&(~rr.source_profile)&(rr["agg"]=="mean")].copy(); sel.to_csv(R/"local_peer_residual_w16_r4_mean_results.csv",index=False,float_format="%.10f")
    # Include detailed slices from prior r2 audit as context (same base family)
    slpath=R/"source_expert_trainaug_r2_localpeer_v1_slices.csv"; sl=pd.read_csv(slpath) if slpath.exists() else pd.DataFrame()
    report=["# Local peer residual width16/radius4 mean candidate", "", "Leakage-safe feature: visible train + unmasked private rows only; 16-day AOI seasonal median profile; same-date/same-crop peers with numeric AOI-ID distance <=4; uniform residual mean.", "", "## Four-mask sweep rows", "", sel.to_string(index=False), "", "## Prior r2 slice context (alpha=.20; width24/r8)", "", sl.to_string(index=False), "", "## Candidate metadata", "", json.dumps(metas,indent=2), "", f"Elapsed seconds: {time.time()-t0:.1f}", "No existing candidate overwritten; no upload performed."]
    (R/"local_peer_residual_w16_r4_mean_report.md").write_text("\n".join(report)+"\n",encoding="utf8")
    print(sel.to_string(index=False));print(json.dumps(metas,indent=2))


if __name__=='__main__': main()

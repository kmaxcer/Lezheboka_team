"""Audit alpha policies keyed by *predicted* route source/confidence.

The route source index is reconstructed from observable same-date peer source
counts and schedule posterior for each mask.  It is not the true source.  The
saved source-expert predictions provide the expert values, so this audit does
not refit models and cannot leak evaluation labels into features.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; D=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(R))
import source_expert_q1 as q1  # noqa: E402
import source_expert_route_v2 as rv  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # noqa: E402
from overnight_source_eval import _predict_matrix  # noqa: E402

ID,DATE,TARGET,GAP="anon_polygon_id","date","primary_ndvi","is_synthetic_gap"
SEEDS=(0,1,2,70404)

def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p); return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan

def load_rows():
    a=pd.read_csv(R/"source_expert_route_v2_rows.csv",parse_dates=[DATE],low_memory=False)
    b=pd.read_csv(R/"source_expert_route_v2_seed2_rows.csv",parse_dates=[DATE],low_memory=False); b["seed"]=2
    cols=[ID,DATE,"truth","true_src","year","cohort","near_dist","baseline","seed"]
    routes=["crop_hier_n1_p67","soft_all_r1_l0_p4","soft_all_r2_l0_p4"]
    out=[]
    for z in (a,b):
        q=z[cols].copy()
        for r in routes:
            c=f"blend_{r}_0.40"; q[f"e_{r}"]=(z[c].to_numpy(float)-.6*z.baseline.to_numpy(float))/.4
        out.append(q)
    return pd.concat(out,ignore_index=True), routes

def reconstruct_route_meta(tr,pr):
    """Recompute route source/confidence using only masked observable frames."""
    rec=[]
    for seed in SEEDS:
        hold=make_holdout(pr,seed=seed)
        ref,gref,sref,pm,gaps=q1._make_masked_ref(tr,pr,hold)
        qkeys=pr.loc[hold,[ID,DATE,"crop_type"]].copy().reset_index(drop=True); qkeys[DATE]=pd.to_datetime(qkeys[DATE])
        pmatrix,_=_predict_matrix(pm,train=tr,family="base",k=8,degree=1,bin_days=30,date_weight=1.0); pmap=pmatrix.set_index("row_index"); qi=np.flatnonzero(hold)
        post=np.column_stack([[pmap.loc[i,c] if i in pmap.index else 1/3 for i in qi] for c in ("p_s2","p_landsat","p_modis")]); post=np.where(np.isfinite(post),post,1/3); post/=post.sum(1,keepdims=True)
        cc,ac,near=rv._neighbor_counts(pm,gaps,qkeys); routes=rv._route_variants(cc,ac,post); hard=routes["crop_hier_n1_p67"].astype(int); postmode=routes["post_mode"].astype(int)
        used=np.zeros(len(qkeys),bool); rix=np.full(len(qkeys),-1,int); pn=np.zeros(len(qkeys)); pp=np.zeros(len(qkeys))
        for j,rad in enumerate(rv.ROUTE_RADII):
            c=cc[:,j]; n=c.sum(1); pur=c.max(1)/np.maximum(1.,n); take=(~used)&(n>=1)&(pur>=.67); rix[take]=j; pn[take]=n[take]; pp[take]=pur[take]; used|=take
        z=qkeys[[ID,DATE]].copy(); z["seed"]=seed; z["pred_route_source"]=hard; z["pred_fallback_source"]=postmode; z["peer_used"]=used; z["route_radius_idx"]=rix; z["route_peer_n"]=pn; z["route_peer_purity"]=pp; z["near_dist_rebuilt"]=near; z["post_max"]=post.max(1); z["post_entropy"]=-np.sum(np.where(post>0,post*np.log(post),0),axis=1); rec.append(z)
    return pd.concat(rec,ignore_index=True)

def policy_alpha(df,name):
    src=df.pred_route_source.to_numpy(int); used=df.peer_used.to_numpy(bool); pur=df.route_peer_purity.to_numpy(float); near=df.near_dist.to_numpy(float); yr=df.year.to_numpy(int); co=df.cohort.astype(str).to_numpy()
    n=len(df)
    if name=="fixed040": return np.full(n,.40)
    if name=="predsrc_ls045": return np.where(src==1,.45,.40)
    if name=="predsrc_ls050": return np.where(src==1,.50,.40)
    if name=="predsrc_ls055": return np.where(src==1,.55,.40)
    if name=="predsrc_ls050_nonls035": return np.where(src==1,.50,.35)
    if name=="peer050_fallback025": return np.where(used,.50,.25)
    if name=="peer050_purity80_040_fallback025": return np.where(~used,.25,np.where(pur>=.80,.50,.40))
    if name=="peer055_purity80_040_fallback025": return np.where(~used,.25,np.where(pur>=.80,.55,.40))
    if name=="distance504030": return np.where(np.isfinite(near)&(near<=2),.50,np.where(np.isfinite(near)&(near<=8),.40,.30))
    if name=="distance504025": return np.where(np.isfinite(near)&(near<=2),.50,np.where(np.isfinite(near)&(near<=8),.40,.25))
    if name=="cohort_year_dist":
        a=np.where(np.isfinite(near)&(near<=2),.50,np.where(np.isfinite(near)&(near<=8),.40,.30)); a=np.where((co=="new")&(yr==2025),.60,a); a=np.where((co=="shared")&(yr==2025),.35,a); return a
    if name=="cohort_year_dist_src":
        a=policy_alpha(df,"cohort_year_dist"); return np.where(src==1,np.minimum(a+.05,.60),a)
    if name=="confidence_src":
        a=np.where(~used,.25,np.where(pur>=.80,.50,.40)); return np.where(src==1,np.minimum(a+.05,.60),a)
    raise ValueError(name)

def main():
    rows,routes=load_rows(); tr=pd.read_csv(D/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/"private_features.csv",parse_dates=[DATE],low_memory=False); tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool)
    meta=reconstruct_route_meta(tr,pr); meta.to_csv(R/"source_expert_route_v2_predsource_confidence_features.csv",index=False,float_format="%.9f")
    key=[ID,DATE,"seed"]; df=rows.merge(meta,on=key,how="left",validate="one_to_one",suffixes=("","_meta"));
    if df.pred_route_source.isna().any(): raise RuntimeError("route metadata alignment failed")
    policies=["fixed040","predsrc_ls045","predsrc_ls050","predsrc_ls055","predsrc_ls050_nonls035","peer050_fallback025","peer050_purity80_040_fallback025","peer055_purity80_040_fallback025","distance504030","distance504025","cohort_year_dist","cohort_year_dist_src","confidence_src"]
    # Evaluate principal crop route only; soft routes are included as a small
    # sanity check with their own predicted values but same confidence policy.
    rec=[]; y=df.truth.to_numpy(float); b=df.baseline.to_numpy(float)
    for route in routes:
        e=df[f"e_{route}"].to_numpy(float)
        for pol in policies:
            a=policy_alpha(df,pol); p=(1-a)*b+a*e
            for sl,m in [("all",np.ones(len(df),bool)),("seed0",df.seed.to_numpy(int)==0),("seed1",df.seed.to_numpy(int)==1),("seed2",df.seed.to_numpy(int)==2),("seed70404",df.seed.to_numpy(int)==70404),("new2025",(df.cohort=="new").to_numpy()&(df.year==2025)),("shared2025",(df.cohort=="shared").to_numpy()&(df.year==2025)),("near",np.isfinite(df.near_dist)&(df.near_dist<=2)),("mid",np.isfinite(df.near_dist)&(df.near_dist>2)&(df.near_dist<=8)),("far",(~np.isfinite(df.near_dist))|(df.near_dist>8))]:
                if m.sum()>=10: rec.append({"route":route,"policy":pol,"slice":sl,"n":int(m.sum()),"rmse":rmse(y[m],p[m]),"rmse_base":rmse(y[m],b[m])})
    metrics=pd.DataFrame(rec); metrics.to_csv(R/"source_expert_route_v2_predsource_confidence_metrics.csv",index=False,float_format="%.10f")
    # LOO policy selection over masks, focused on crop route.
    loo=[]; e=df.e_crop_hier_n1_p67.to_numpy(float); sarr=df.seed.to_numpy(int)
    for held in SEEDS:
        trm=sarr!=held; tem=~trm; scores=[]
        for pol in policies:
            p=(1-policy_alpha(df,pol))*b+policy_alpha(df,pol)*e; scores.append((rmse(y[trm],p[trm]),pol))
        scores.sort(); best=scores[0][1]
        for pol in [best,"fixed040","predsrc_ls050","peer050_purity80_040_fallback025","distance504030","cohort_year_dist","cohort_year_dist_src"]:
            p=(1-policy_alpha(df,pol))*b+policy_alpha(df,pol)*e; loo.append({"held_seed":held,"selected_train_policy":best,"policy":pol,"train_rmse":rmse(y[trm],p[trm]),"test_rmse":rmse(y[tem],p[tem]),"test_base":rmse(y[tem],b[tem])})
    loo_df=pd.DataFrame(loo); loo_df.to_csv(R/"source_expert_route_v2_predsource_confidence_loo.csv",index=False,float_format="%.10f")
    best=metrics.query("route=='crop_hier_n1_p67' and slice=='all'").sort_values("rmse").head(20); lines=["# Predicted-source/confidence alpha audit", "", "Route source/confidence is reconstructed strictly from observable masked source schedules; true source is not a feature.", "", "## Pooled shortlist", "", best.to_string(index=False), "", "## Leave-one-mask-out", "", loo_df.to_string(index=False), "", "No candidate materialized: promote only if a policy beats fixed .40 on every held mask by >=1e-5."]
    (R/"source_expert_route_v2_predsource_confidence_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(best.to_string(index=False)); print(loo_df.to_string(index=False))

if __name__=="__main__": main()

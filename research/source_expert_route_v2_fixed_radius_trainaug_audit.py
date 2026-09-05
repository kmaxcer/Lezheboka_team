"""Evaluate fixed-radius routes using train + visible-private neighbors.

The prior ``*_srcfix`` audit intentionally used only visible private rows.  A
separate schedule probe established that adding fully visible train rows gives
much stronger same-date/crop source modes.  This script reuses the already fit
expert matrices from that audit and the observable route indices from
``source_schedule_route_probe_rows.csv``; no HGB refit or label-derived
feature is needed.  It writes only uniquely named research artifacts.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"
BASE=R/"source_expert_route_v2_fixed_radius_srcfix_rows.csv"; SCHED=R/"source_schedule_route_probe_rows.csv"
ROUTES=[1,2,4,8,16,32]

def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p); return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan

def main():
    b=pd.read_csv(BASE,parse_dates=["date"],low_memory=False); s=pd.read_csv(SCHED,parse_dates=["date"],low_memory=False)
    key=["anon_polygon_id","date","seed"]
    take=key+sum(([f"sp_crop_{r}",f"sp_crop_{r}_n"] for r in ROUTES),[])
    z=b.merge(s[take],on=key,how="left",validate="one_to_one")
    if z[f"sp_crop_4"].isna().any(): raise RuntimeError("schedule alignment failed")
    y=z.truth.to_numpy(float); B=z.baseline.to_numpy(float); srcmap={"s2":0,"landsat":1,"modis":2}; truthsrc=z.true_src.map(srcmap).to_numpy(int); seeds=sorted(z.seed.unique().astype(int)); rec=[]; acc=[]
    # Train-augmented fixed mode; fallback is the already computed observable
    # posterior mode when no same-date/crop peer exists.
    for r in ROUTES:
        raw=z[f"sp_crop_{r}"].to_numpy(int); npeer=z[f"sp_crop_{r}_n"].to_numpy(int); idx=raw.copy(); fb=idx<0; idx[fb]=z.route_post_mode.to_numpy(int)[fb]; e=z[["e_s2","e_landsat","e_modis"]].to_numpy(float); psrc=e[np.arange(len(z)),idx];
        for a in (.2,.3,.4,.5,.6):
            p=(1-a)*B+a*psrc; rec.append({"route":f"trainaug_r{r}","policy":f"a{a:.2f}","alpha":a,"n":len(z),"rmse":rmse(y,p),"base_rmse":rmse(y,B),"per_seed":";".join(f"{ss}:{rmse(y[z.seed.to_numpy(int)==ss],p[z.seed.to_numpy(int)==ss]):.6f}" for ss in seeds)})
        # Conservative distance/cohort policy using train-aug peer counts:
        # <=r2 is near, r4--r8 mid, no r8 peer far.
        n2=z.sp_crop_2_n.to_numpy(int); n8=z.sp_crop_8_n.to_numpy(int); near=n2>0; mid=(~near)&(n8>0); a=np.where(near,.50,np.where(mid,.40,.30)); yr=z.year.to_numpy(int); co=z.cohort.astype(str).to_numpy(); a=np.where((co=="new")&(yr==2025),.60,a); a=np.where((co=="shared")&(yr==2025),.35,a); p=(1-a)*B+a*psrc; rec.append({"route":f"trainaug_r{r}","policy":"cohort_year_trainaug","alpha":np.nan,"n":len(z),"rmse":rmse(y,p),"base_rmse":rmse(y,B),"per_seed":";".join(f"{ss}:{rmse(y[z.seed.to_numpy(int)==ss],p[z.seed.to_numpy(int)==ss]):.6f}" for ss in seeds)})
        for ss in seeds:
            g=z.seed.to_numpy(int)==ss; cov=npeer>0; cg=cov&g; acc.append({"route":f"trainaug_r{r}","seed":ss,"n":int(g.sum()),"coverage":float(cov[g].mean()),"covered_n":int(cg.sum()),"source_accuracy_covered":float(np.mean(idx[cg]==truthsrc[cg])) if cg.any() else np.nan,"source_accuracy_all":float(np.mean(idx[g]==truthsrc[g]))})
    # LOO alpha selection for each augmented route (fixed alpha grid + policy).
    loo=[]; sarr=z.seed.to_numpy(int); eall=z[["e_s2","e_landsat","e_modis"]].to_numpy(float)
    for r in ROUTES:
        idx=z[f"sp_crop_{r}"].to_numpy(int).copy(); fb=idx<0; idx[fb]=z.route_post_mode.to_numpy(int)[fb]; e=eall[np.arange(len(z)),idx]; n2=z.sp_crop_2_n.to_numpy(int); n8=z.sp_crop_8_n.to_numpy(int); near=n2>0; mid=(~near)&(n8>0); aa=np.where(near,.50,np.where(mid,.40,.30)); yr=z.year.to_numpy(int); co=z.cohort.astype(str).to_numpy(); aa=np.where((co=="new")&(yr==2025),.60,aa); aa=np.where((co=="shared")&(yr==2025),.35,aa)
        for held in seeds:
            trm=sarr!=held; tem=~trm; vals=[(rmse(y[trm],(1-a)*B[trm]+a*e[trm]),a) for a in np.arange(.2,.61,.05)]; vals.append((rmse(y[trm],(1-aa[trm])*B[trm]+aa[trm]*e[trm]),"cohort_year_trainaug")); vals.sort(key=lambda x:x[0]); sel=vals[0][1]; p=(1-(aa if isinstance(sel,str) else sel))*B+(aa if isinstance(sel,str) else sel)*e; loo.append({"route":f"trainaug_r{r}","held_seed":held,"selected":sel,"train_rmse":rmse(y[trm],p[trm]),"test_rmse":rmse(y[tem],p[tem]),"test_base":rmse(y[tem],B[tem])})
    metrics=pd.DataFrame(rec).sort_values("rmse"); accdf=pd.DataFrame(acc); loodf=pd.DataFrame(loo); stem="source_expert_route_v2_fixed_radius_trainaug"; metrics.to_csv(R/(stem+"_metrics.csv"),index=False,float_format="%.10f"); accdf.to_csv(R/(stem+"_source_accuracy.csv"),index=False,float_format="%.10f"); loodf.to_csv(R/(stem+"_loo.csv"),index=False,float_format="%.10f")
    # Keep compact per-row predictions for any later candidate materialization.
    out=z[["anon_polygon_id","date","seed","truth","true_src","year","cohort","baseline"]].copy()
    for r in ROUTES:
        idx=z[f"sp_crop_{r}"].to_numpy(int).copy(); fb=idx<0; idx[fb]=z.route_post_mode.to_numpy(int)[fb]; out[f"route_trainaug_r{r}"]=idx; out[f"expert_trainaug_r{r}"]=eall[np.arange(len(z)),idx]
    out.to_csv(R/(stem+"_rows.csv"),index=False,float_format="%.9f")
    lines=["# Fixed-radius train-augmented source-route audit","", "Routes use same-date/same-crop numeric-AOI modes from train + visible private rows; hidden/holdout rows are excluded by the schedule probe. True source is scoring-only.","","## Pooled metrics","",metrics.head(40).to_string(index=False),"","## Source accuracy/coverage","",accdf.to_string(index=False),"","## LOO","",loodf.to_string(index=False),"", "No output candidate was materialized or overwritten."]
    (R/(stem+"_report.md")).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(metrics.head(40).to_string(index=False)); print(accdf.to_string(index=False)); print(loodf.to_string(index=False))

if __name__=="__main__": main()

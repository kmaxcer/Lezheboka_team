"""All-mask fixed-radius source-route probe (source-fix branch).

Fits the existing leakage-safe S2/Landsat/MODIS HGB experts once per
private-like mask, then compares cumulative same-date/same-crop numeric-AOI
mode routes at radii 1,2,3,4,5,6,8,16,32.  A route falls back to the
observable schedule posterior mode when no peer is present.  A small r2+r4
weighted-vote route is included.  No true source/target enters route features;
they are retained only in the scored sidecar.  This script writes research
artifacts only (candidate materialization is a separate explicit mode).
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; OUT=ROOT/"outputs"; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(R))
import source_expert_q1 as q1  # noqa: E402
import source_expert_route_v2 as rv  # noqa: E402
import meta_residual_v2_independent as indep  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # noqa: E402
from overnight_source_eval import _predict_matrix, _source_labels  # noqa: E402

ID,DATE,TARGET,GAP="anon_polygon_id","date","primary_ndvi","is_synthetic_gap"
SEEDS=(0,1,2,70404); RADII=(1,2,3,4,5,6,8,16,32)

def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p); return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan

def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def baseline_for(seed,tr,pr,hold):
    if seed==2:
        q=indep._make_q(tr,pr,seed); m=q.set_index([ID,DATE])["ext40"]
    elif seed==70404:
        z=pd.read_csv(R/"private_cohort_blend_holdout_predictions.csv",parse_dates=[DATE],low_memory=False); z[DATE]=pd.to_datetime(z[DATE]); yrs=z[DATE].dt.year.to_numpy(int); z["baseline"]=z.ext40.to_numpy(float)+np.where(yrs<2025,.12,0.)*np.nan_to_num(z.n16_c60_r125_k2.to_numpy(float)-z.ext40.to_numpy(float),nan=0.); m=z.set_index([ID,DATE])["baseline"]
    else:
        z=pd.read_csv(R/"meta_residual_v2_independent_predictions.csv",parse_dates=[DATE],low_memory=False); z=z[z.mask_seed.eq(seed)].copy(); z[DATE]=pd.to_datetime(z[DATE]); yrs=z[DATE].dt.year.to_numpy(int); z["baseline"]=z.ext40.to_numpy(float)+np.where(yrs<2025,.12,0.)*np.nan_to_num(z.n16_c60_r125_k2.to_numpy(float)-z.ext40.to_numpy(float),nan=0.); m=z.set_index([ID,DATE])["baseline"]
    qkeys=pr.loc[hold,[ID,DATE]]; return np.asarray([m.get((i,d),np.nan) for i,d in qkeys.itertuples(index=False,name=None)],float)

def counts_arbitrary(pm,gaps,qkeys,radii=RADII):
    d=pm.reset_index(drop=True).copy(); d[DATE]=pd.to_datetime(d[DATE]); src=np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],[0,1,2],-1); ids=pd.to_numeric(d[ID].astype(str).str.extract(r"(\d+)",expand=False),errors="coerce").fillna(-1).to_numpy(int); dates=d[DATE].to_numpy(); crops=d.crop_type.fillna("unknown").astype(str).to_numpy(); vis=np.flatnonzero((~np.asarray(gaps,bool))&(src>=0)); bydate={dt:np.asarray(ix,int) for dt,ix in pd.Series(vis,index=vis).groupby(dates[vis])}; q=qkeys.reset_index(drop=True); qdates=pd.to_datetime(q[DATE]).to_numpy(); qids=pd.to_numeric(q[ID].astype(str).str.extract(r"(\d+)",expand=False),errors="coerce").fillna(-1).to_numpy(int); qcrops=q.crop_type.fillna("unknown").astype(str).to_numpy(); cc=np.zeros((len(q),len(radii),3),float); near=np.full(len(q),np.inf)
    for n,(dt,aid,crop) in enumerate(zip(qdates,qids,qcrops)):
        z0=np.asarray(bydate.get(dt,np.empty(0,int)),dtype=int)
        for j,rad in enumerate(radii):
            z=z0[np.abs(ids[z0]-aid)<=rad]; zz=z[crops[z]==crop] if len(z) else z
            if len(zz): cc[n,j]=np.bincount(src[zz],minlength=3); near[n]=min(near[n],float(np.min(np.abs(ids[zz]-aid))))
    return cc,near

def route_indices(cc,post):
    out={"post_mode":np.argmax(post,axis=1).astype(int)}
    for j,rad in enumerate(RADII):
        c=cc[:,j]; n=c.sum(1); r=np.argmax(c,axis=1).astype(int); r[n<=0]=out["post_mode"][n<=0]; out[f"fixed_r{rad}"]=r
    # Weighted/majority vote of cumulative r2 and r4 modes.  On ties the
    # larger-radius count mode wins; no labels are consulted.
    c2,c4=cc[:,1],cc[:,3]; n2,n4=c2.sum(1),c4.sum(1); m2=np.argmax(c2,axis=1); m4=np.argmax(c4,axis=1); vote=np.argmax(c2+c4,axis=1).astype(int); vote[(n2<=0)&(n4<=0)]=out["post_mode"][(n2<=0)&(n4<=0)]; out["vote_r2_r4"]=vote
    return out

def fit_one(seed,tr,pr):
    hold=make_holdout(pr,seed=seed); ref,gref,sref,pm,gaps=q1._make_masked_ref(tr,pr,hold); _,ep_ref,_=q1._fit_experts(ref,gref,sref); qref=ref.loc[gref,[ID,DATE]].copy().reset_index(drop=True); qref[["e_s2","e_landsat","e_modis"]]=ep_ref; qkeys=pr.loc[hold,[ID,DATE,"crop_type"]].copy().reset_index(drop=True); qkeys[DATE]=pd.to_datetime(qkeys[DATE]); q=qkeys.merge(qref,on=[ID,DATE],how="left",validate="one_to_one");
    if q[["e_s2","e_landsat","e_modis"]].isna().any().any(): raise RuntimeError(f"expert alignment seed {seed}")
    pmatrix,_=_predict_matrix(pm,train=tr,family="base",k=8,degree=1,bin_days=30,date_weight=1.0); pmap=pmatrix.set_index("row_index"); qi=np.flatnonzero(hold); post=np.column_stack([[pmap.loc[i,c] if i in pmap.index else 1/3 for i in qi] for c in ("p_s2","p_landsat","p_modis")]); post=np.where(np.isfinite(post),post,1/3); post/=post.sum(1,keepdims=True); cc,near=counts_arbitrary(pm,gaps,qkeys); routes=route_indices(cc,post); B=baseline_for(seed,tr,pr,hold); E=q[["e_s2","e_landsat","e_modis"]].to_numpy(float); y=pr.loc[hold,TARGET].to_numpy(float); true=_source_labels(pr)[hold]; yr=q[DATE].dt.year.to_numpy(int); cohort=np.where(q[ID].astype(str).isin(set(tr[ID].astype(str))),"shared","new")
    rows=q[[ID,DATE]].copy(); rows["seed"]=seed; rows["truth"]=y; rows["true_src"]=true; rows["year"]=yr; rows["cohort"]=cohort; rows["baseline"]=B; rows["near_dist"]=near; rows[["e_s2","e_landsat","e_modis"]]=E; rows["post_mode"]=routes["post_mode"]
    metrics=[]; acc=[]
    for name,idx in routes.items():
        psrc=E[np.arange(len(E)),idx]; rows[f"route_{name}"]=idx; rows[f"expert_{name}"]=psrc
        covered=(cc[:,0].sum(1)>0) if name.startswith("fixed_r1") else None
        # Source accuracy and coverage by route radius; diagnostics only.
        if name.startswith("fixed_r"):
            rj=RADII.index(int(name[7:])); cov=cc[:,rj].sum(1)>0; acc.append({"seed":seed,"route":name,"n":len(y),"coverage":float(cov.mean()),"covered_n":int(cov.sum()),"source_accuracy_covered":float(np.mean(idx[cov]==true[cov])) if cov.any() else np.nan,"source_accuracy_all":float(np.mean(idx==true))})
        for alpha in (0.2,.3,.4,.5,.6): metrics.append({"seed":seed,"route":name,"policy":"fixed","alpha":alpha,"n":len(y),"rmse":rmse(y,(1-alpha)*B+alpha*psrc),"rmse_base":rmse(y,B)})
        # Adaptive cohort/year policy established by the separate four-mask audit.
        near2=np.isfinite(near)&(near<=2); mid=np.isfinite(near)&(near>2)&(near<=8); a=np.where(near2,.50,np.where(mid,.40,.30)); a=np.where((cohort=="new")&(yr==2025),.60,a); a=np.where((cohort=="shared")&(yr==2025),.35,a); metrics.append({"seed":seed,"route":name,"policy":"cohort_year_dist","alpha":np.nan,"n":len(y),"rmse":rmse(y,(1-a)*B+a*psrc),"rmse_base":rmse(y,B)})
    return rows,pd.DataFrame(metrics),pd.DataFrame(acc)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",default=','.join(map(str,SEEDS))); args=ap.parse_args(); seeds=tuple(int(x) for x in args.seeds.split(',')); t0=time.time(); tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False); tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool); allr=[]; allm=[]; alla=[]
    for s in seeds:
        print(f"fixed-radius source fit seed={s}",flush=True); r,m,a=fit_one(s,tr,pr); allr.append(r); allm.append(m); alla.append(a); print(a.to_string(index=False),flush=True)
    rows=pd.concat(allr,ignore_index=True); mets=pd.concat(allm,ignore_index=True); acc=pd.concat(alla,ignore_index=True); stem="source_expert_route_v2_fixed_radius_srcfix"; rows.to_csv(R/(stem+"_rows.csv"),index=False,float_format="%.9f"); mets.to_csv(R/(stem+"_metrics.csv"),index=False,float_format="%.10f"); acc.to_csv(R/(stem+"_source_accuracy.csv"),index=False,float_format="%.10f")
    pooled=mets.groupby(["route","policy","alpha"],dropna=False).apply(lambda g: pd.Series({"n":int(g.n.sum()),"rmse":float(np.sqrt(np.average(g.rmse**2,weights=g.n))),"base_rmse":float(np.sqrt(np.average(g.rmse_base**2,weights=g.n))),"per_seed":";".join(f"{int(s)}:{rmse(h.truth.to_numpy(),((1-(h.alpha.iloc[0] if h.policy.iloc[0]=='fixed' else .4))*h.baseline.to_numpy()+(h.rmse*0+h.baseline).to_numpy())):.6f}" for s,h in [])}),include_groups=False).reset_index() if False else None
    # Simpler pooled score from row sidecar and route columns.
    pool=[]
    for route in [c[len("expert_"):] for c in rows.columns if c.startswith("expert_")]:
        e=rows[f"expert_{route}"].to_numpy(float); b=rows.baseline.to_numpy(float); y=rows.truth.to_numpy(float); sarr=rows.seed.to_numpy(int); near=rows.near_dist.to_numpy(float); yr=rows.year.to_numpy(int); co=rows.cohort.to_numpy(str); near2=np.isfinite(near)&(near<=2); mid=np.isfinite(near)&(near>2)&(near<=8); aa=np.where(near2,.50,np.where(mid,.40,.30)); aa=np.where((co=="new")&(yr==2025),.60,aa); aa=np.where((co=="shared")&(yr==2025),.35,aa)
        for pol,a in [("a020",np.full(len(rows),.2)),("a030",np.full(len(rows),.3)),("a040",np.full(len(rows),.4)),("a050",np.full(len(rows),.5)),("a060",np.full(len(rows),.6)),("cohort_year_dist",aa)]:
            p=(1-a)*b+a*e; pool.append({"route":route,"policy":pol,"n":len(y),"rmse":rmse(y,p),"per_seed":";".join(f"{s}:{rmse(y[sarr==s],p[sarr==s]):.6f}" for s in sorted(np.unique(sarr)))})
    pool=pd.DataFrame(pool).sort_values("rmse"); pool.to_csv(R/(stem+"_pooled.csv"),index=False,float_format="%.10f")
    # LOO fixed alpha and adaptive policy for each route.
    loo=[]
    for route in [c[len("expert_"):] for c in rows.columns if c.startswith("expert_")]:
        e=rows[f"expert_{route}"].to_numpy(float); b=rows.baseline.to_numpy(float); y=rows.truth.to_numpy(float); sarr=rows.seed.to_numpy(int); near=rows.near_dist.to_numpy(float); yr=rows.year.to_numpy(int); co=rows.cohort.to_numpy(str); near2=np.isfinite(near)&(near<=2); mid=np.isfinite(near)&(near>2)&(near<=8); aa=np.where(near2,.50,np.where(mid,.40,.30)); aa=np.where((co=="new")&(yr==2025),.60,aa); aa=np.where((co=="shared")&(yr==2025),.35,aa)
        for held in sorted(np.unique(sarr)):
            trm=sarr!=held; tem=~trm; cand=[]
            for a in np.arange(.2,.61,.05): cand.append((rmse(y[trm],(1-a)*b[trm]+a*e[trm]),f"a{a:.2f}"))
            cand.append((rmse(y[trm],(1-aa[trm])*b[trm]+aa[trm]*e[trm]),"cohort_year_dist")); best=min(cand)[1]; av=float(best[1:]) if best.startswith('a') else np.nan; p=(1-(av if np.isfinite(av) else aa))*b+(av if np.isfinite(av) else aa)*e; loo.append({"route":route,"held_seed":held,"selected":best,"train_rmse":rmse(y[trm],p[trm]),"test_rmse":rmse(y[tem],p[tem]),"test_base":rmse(y[tem],b[tem])})
    loo=pd.DataFrame(loo); loo.to_csv(R/(stem+"_loo.csv"),index=False,float_format="%.10f")
    report=["# Fixed-radius source-route probe (source-fix)","",f"Seeds: {seeds}; routes use only masked observable same-date/same-crop source counts; true source is scoring-only.","","## Pooled scores","",pool.head(40).to_string(index=False),"","## Source-route accuracy","",acc.to_string(index=False),"","## LOO","",loo.to_string(index=False),"",f"Elapsed seconds: {time.time()-t0:.1f}","No outputs/ candidate was overwritten or materialized."]
    (R/(stem+"_report.md")).write_text("\n".join(report)+"\n",encoding="utf-8"); (R/(stem+"_metadata.json")).write_text(json.dumps({"seeds":seeds,"rows":len(rows),"private_hidden":int(pr[GAP].sum()),"artifacts":[stem+"_rows.csv",stem+"_metrics.csv",stem+"_source_accuracy.csv",stem+"_pooled.csv",stem+"_loo.csv"],"production_baseline_overwritten":False},indent=2),encoding="utf-8")
    print(pool.head(40).to_string(index=False)); print(loo.to_string(index=False))

if __name__=="__main__": main()

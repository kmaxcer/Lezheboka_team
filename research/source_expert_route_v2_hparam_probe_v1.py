"""Bounded source-HGB hyperparameter and clipping probe.

OOF feature matrices are built once per independent mask, then a small set of
regularized source-expert variants is fit.  Predictions are routed with the
observable train-augmented fixed-r2 schedule, and evaluated across masks
0/1/2/70404.  This is research-only; no output candidate is written.
"""
from __future__ import annotations
from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; D=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); sys.path.insert(0,str(R))
import source_expert_q1 as q1  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # noqa: E402
from agropulse.pipeline import FULL_FEATURES, build_features  # type: ignore # noqa: E402
from source_expert_route_v2_fixed_radius_trainaug_audit import rmse  # noqa: E402

ID,DATE,TARGET,GAP="anon_polygon_id","date","primary_ndvi","is_synthetic_gap"; SOURCES=("s2","landsat","modis"); SEEDS=(0,1,2,70404)

VARIANTS={
    "current_refit": {"max_leaf_nodes":40,"min_samples_leaf":40,"l2_regularization":10.,"learning_rate":.035,"max_iter":260},
    "alt_reg32": {"max_leaf_nodes":32,"min_samples_leaf":60,"l2_regularization":16.,"learning_rate":.035,"max_iter":260},
    "alt_smooth48": {"max_leaf_nodes":48,"min_samples_leaf":80,"l2_regularization":25.,"learning_rate":.025,"max_iter":300},
}

def build_oof(seed,tr,pr):
    hold=make_holdout(pr,seed=seed); ref,gaps,sref,pm,gaps_pr=q1._make_masked_ref(tr,pr,hold); known=ref[TARGET].notna().to_numpy(bool)&~gaps; folds=np.full(len(ref),-1,int); rng=np.random.default_rng(42042)
    for _,ix0 in ref.loc[known].groupby(ID,sort=False).groups.items(): ix=np.asarray(ix0,int).copy(); rng.shuffle(ix); folds[ix]=np.arange(len(ix))%5
    xb=[];yb=[];sb=[]
    for f in range(5):
        pseudo=folds==f
        if not pseudo.any(): continue
        hidden=gaps|pseudo; obs=ref[TARGET].mask(hidden); xx=build_features(ref,obs,pd.Series(hidden,index=ref.index)); keep=pseudo&np.isin(sref,SOURCES); xb.append(xx.loc[keep,FULL_FEATURES]); yb.append(ref.loc[keep,"_truth"].astype(float)); sb.append(sref[keep]); print(f"seed {seed} fold {f}: {int(keep.sum())}",flush=True)
    X=pd.concat(xb,ignore_index=True); y=pd.concat(yb,ignore_index=True).to_numpy(float); s=np.concatenate(sb); obsq=ref[TARGET].mask(gaps); xq=build_features(ref,obsq,pd.Series(gaps,index=ref.index)).loc[gaps,FULL_FEATURES]; qref=ref.loc[gaps,[ID,DATE]].copy().reset_index(drop=True); qkeys=pr.loc[hold,[ID,DATE]].copy().reset_index(drop=True); qkeys[DATE]=pd.to_datetime(qkeys[DATE]); return hold,pm,gaps_pr,X,y,s,xq,qref,qkeys

def model_params(name,src):
    p=dict(VARIANTS[name]);
    if src=="modis":
        if name=="current_refit": p.update(max_leaf_nodes=28,min_samples_leaf=55,l2_regularization=14.)
        elif name=="alt_reg32": p.update(max_leaf_nodes=24,min_samples_leaf=70,l2_regularization=20.)
        else: p.update(max_leaf_nodes=32,min_samples_leaf=90,l2_regularization=28.)
    p["loss"]="squared_error"; p["random_state"]=42+SOURCES.index(src); return p

def main():
    t0=time.time(); tr=pd.read_csv(D/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/"private_features.csv",parse_dates=[DATE],low_memory=False); tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool)
    # Existing q1 matrices give the exact current-refit reference and route
    # sidecars; use only keys/observable schedule fields here.
    cur=pd.read_csv(R/"source_expert_route_v2_fixed_radius_srcfix_rows.csv",parse_dates=[DATE],low_memory=False); sched=pd.read_csv(R/"source_schedule_route_probe_rows.csv",parse_dates=[DATE],low_memory=False)
    rec=[]; predparts=[]
    for seed in SEEDS:
        print(f"hparam seed={seed}",flush=True); hold,pm,gaps,X,y,s,xq,qref,qkeys=build_oof(seed,tr,pr); qref_keys=qref.copy(); qref_keys[DATE]=pd.to_datetime(qref_keys[DATE]); qpreds={}
        for name in VARIANTS:
            ep=np.full((len(xq),3),np.nan)
            for j,src in enumerate(SOURCES):
                take=s==src; print(f"fit {name} {src}: {int(take.sum())}",flush=True); m=HistGradientBoostingRegressor(**model_params(name,src)); m.fit(X.loc[take],y[take]); ep[:,j]=m.predict(xq)
            # Map sorted-ref gap predictions to this mask's holdout keys.
            z=qref_keys.copy(); z[["e_s2","e_landsat","e_modis"]]=ep; z=z.merge(qkeys,on=[ID,DATE],how="inner",validate="one_to_one")
            # Join saved current baseline and train-augmented route indices.
            cc=cur[cur.seed.astype(int)==seed][[ID,DATE,"baseline"]+["e_s2","e_landsat","e_modis"]].copy(); cc[DATE]=pd.to_datetime(cc[DATE]); z=z.merge(cc,on=[ID,DATE],how="left",suffixes=("","_cur"),validate="one_to_one"); ss=sched[sched.seed.astype(int)==seed][[ID,DATE,"sp_crop_2","sp_crop_2_n","sp_crop_8_n"]].copy(); ss[DATE]=pd.to_datetime(ss[DATE]); z=z.merge(ss,on=[ID,DATE],how="left",validate="one_to_one");
            if z.baseline.isna().any(): raise RuntimeError(f"baseline alignment {seed}")
            raw=z.sp_crop_2.to_numpy(int).copy(); fb=raw<0; fallback=cur[cur.seed.astype(int)==seed].set_index([ID,DATE])["route_post_mode"]; raw[fb]=[int(fallback.get((i,d),0)) for i,d in z.loc[fb,[ID,DATE]].itertuples(index=False,name=None)]; idx=raw; E=z[["e_s2","e_landsat","e_modis"]].to_numpy(float); B=z.baseline.to_numpy(float); Y=pr.set_index([ID,DATE]).loc[pd.MultiIndex.from_frame(z[[ID,DATE]]),TARGET].to_numpy(float); psrc=E[np.arange(len(E)),idx]; near=z.sp_crop_2_n.to_numpy(int)>0; mid=(~near)&(z.sp_crop_8_n.to_numpy(int)>0); yr=z[DATE].dt.year.to_numpy(int); co=np.where(z[ID].astype(str).isin(set(tr[ID].astype(str))),"shared","new"); A=np.where(near,.50,np.where(mid,.40,.30)); A=np.where((co=="new")&(yr==2025),.60,A); A=np.where((co=="shared")&(yr==2025),.35,A)
            # Clipping variants: absolute sensor range and source-specific
            # residual caps are observable safeguards, not label-derived.
            clip_modes={"none":psrc,"abs01":np.clip(psrc,0.,1.),"delta_src":B+np.clip(psrc-B,-np.where(idx==0,.10,np.where(idx==1,.08,.06)),np.where(idx==0,.10,np.where(idx==1,.08,.06)))}
            for clipname,srcp in clip_modes.items():
                for pol,a in [("a040",np.full(len(z),.40)),("a050",np.full(len(z),.50)),("cohort_year_trainaug",A)]:
                    p=(1-a)*B+a*srcp; rec.append({"seed":seed,"variant":name,"clip":clipname,"policy":pol,"n":len(Y),"rmse":rmse(Y,p),"base_rmse":rmse(Y,B)})
            # Save only the raw predictions needed for post-hoc LOO.
            predparts.append(pd.DataFrame({ID:z[ID],DATE:z[DATE],"seed":seed,"truth":Y,"baseline":B,"variant":name,"psrc":psrc,"alpha_policy":A}))
    mdf=pd.DataFrame(rec); pdf=pd.concat(predparts,ignore_index=True); stem="source_expert_route_v2_hparam_probe_v1"; mdf.to_csv(R/(stem+"_metrics.csv"),index=False,float_format="%.10f"); pdf.to_csv(R/(stem+"_predictions.csv"),index=False,float_format="%.9f")
    # Pooled and LOO summaries for raw/no-clip policy; alternatives are still
    # listed in metrics for audit.
    po=[]
    for (v,c,pol),g in mdf.groupby(["variant","clip","policy"],dropna=False): po.append({"variant":v,"clip":c,"policy":pol,"n":int(g.n.sum()),"rmse":float(np.sqrt(np.average(g.rmse**2,weights=g.n))),"base_rmse":float(np.sqrt(np.average(g.base_rmse**2,weights=g.n))),"per_seed":";".join(f"{int(s)}:{h.rmse.iloc[0]:.6f}" for s,h in g.groupby("seed"))})
    podf=pd.DataFrame(po).sort_values("rmse"); podf.to_csv(R/(stem+"_pooled.csv"),index=False,float_format="%.10f")
    loo=[]
    for v in pdf.variant.unique():
        for held in SEEDS:
            trm=pdf.seed!=held; te=~trm; choices=[]
            for clip in ["none","abs01","delta_src"]:
                for pol in ["a040","a050","cohort_year_trainaug"]:
                    g=pdf[(pdf.variant==v)&trm]; a=np.where(pol=="a040",.4,np.where(pol=="a050",.5,g.alpha_policy.to_numpy(float))); # alpha policy saved only for cohort
                    psrc=g.psrc.to_numpy(float); b=g.baseline.to_numpy(float); yy=g.truth.to_numpy(float); 
                    if clip=="abs01": psrc=np.clip(psrc,0,1)
                    elif clip=="delta_src": psrc=b+np.clip(psrc-b,-.08,.08)
                    choices.append((rmse(yy,(1-a)*b+a*psrc),clip,pol))
            choices.sort(); sel=choices[0]; clip,pol=sel[1],sel[2]; g=pdf[(pdf.variant==v)&te]; a=np.where(pol=="a040",.4,np.where(pol=="a050",.5,g.alpha_policy.to_numpy(float))); psrc=g.psrc.to_numpy(float); b=g.baseline.to_numpy(float); psrc=np.clip(psrc,0,1) if clip=="abs01" else (b+np.clip(psrc-b,-.08,.08) if clip=="delta_src" else psrc); loo.append({"variant":v,"held_seed":held,"selected":f"{clip}_{pol}","train_rmse":sel[0],"test_rmse":rmse(g.truth.to_numpy(float),(1-a)*b+a*psrc),"test_base":rmse(g.truth.to_numpy(float),b)})
    loodf=pd.DataFrame(loo); loodf.to_csv(R/(stem+"_loo.csv"),index=False,float_format="%.10f")
    report=["# Source-expert HGB hyperparameter probe v1","", "All four masks; q1 OOF features rebuilt once per mask. Routes use train-augmented fixed-r2 observable schedule; no true source/hidden target enters features.","","## Pooled", "",podf.head(40).to_string(index=False),"","## LOO", "",loodf.to_string(index=False),"",f"Elapsed seconds: {time.time()-t0:.1f}","No output candidate was materialized."]
    (R/(stem+"_report.md")).write_text("\n".join(report)+"\n",encoding="utf-8"); (R/(stem+"_metadata.json")).write_text(json.dumps({"seeds":SEEDS,"variants":list(VARIANTS),"production_baseline_overwritten":False},indent=2),encoding="utf-8"); print(podf.head(40).to_string(index=False)); print(loodf.to_string(index=False))

if __name__=="__main__": main()

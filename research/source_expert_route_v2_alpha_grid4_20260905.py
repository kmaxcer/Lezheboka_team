"""Fine-grid and nested LOO search for route-v2 alpha policies (4 masks).

This is a bounded audit after the stable distance policy was found.  It uses
the persisted route-v2 rows (plus fresh seed=2 rows when present), reconstructs
the raw routed expert from alpha=.40 blends, and never uses ``true_src`` as an
input.  Outputs are diagnostics/candidate selection only; no prior file is
overwritten.
"""
from __future__ import annotations
from pathlib import Path
import itertools, json
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; REPORT=ROOT/"reports"/"source_expert_route_v2_alpha_grid4_report_20260905.md"
MAIN=R/"source_expert_route_v2_rows.csv"; S2=R/"source_expert_route_v2_seed2_rows.csv"
OUT_GRID=R/"source_expert_route_v2_alpha_grid4_20260905.csv"; OUT_LOO=R/"source_expert_route_v2_alpha_grid4_loo_20260905.csv"

def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); return float(np.sqrt(np.mean((p-y)**2)))

def load():
    a=pd.read_csv(MAIN,parse_dates=["date"],low_memory=False)
    a["seed"]=a.seed.astype(int); a["expert"]=(a["blend_crop_hier_n1_p67_0.40"]-.6*a.baseline)/.4
    cols=["anon_polygon_id","date","truth","year","cohort","near_dist","seed","baseline","expert"]
    a=a[cols].copy()
    if S2.exists():
        b=pd.read_csv(S2,parse_dates=["date"],low_memory=False)
        b["seed"]=2
        b=b[["anon_polygon_id","date","truth","year","cohort","near_dist","seed","baseline","expert_crop_hier_n1_p67"]].rename(columns={"expert_crop_hier_n1_p67":"expert"})
        a=pd.concat([a,b],ignore_index=True)
    nd=a.near_dist.to_numpy(float)
    a["bin3"]=np.where(np.isfinite(nd)&(nd<=2),0,np.where(np.isfinite(nd)&(nd<=8),1,2))
    # Finer bins probe whether the 2/8-day breakpoints are too coarse.
    a["bin5"]=np.select([np.isfinite(nd)&(nd<=1),np.isfinite(nd)&(nd<=2),np.isfinite(nd)&(nd<=4),np.isfinite(nd)&(nd<=8)],[0,1,2,3],default=4)
    a["bin4"]=np.where(np.isfinite(nd)&(nd<=2),0,np.where(np.isfinite(nd)&(nd<=5),1,np.where(np.isfinite(nd)&(nd<=10),2,3)))
    a["year_bin"]=np.where(a.year.astype(int).to_numpy()==2025,"2025","history")
    a["cohort_bin"]=a.cohort.astype(str).to_numpy()
    a["delta_abs"]=np.abs(a.expert-a.baseline)
    q=np.nanquantile(a.delta_abs,[0,.25,.5,.75,1]); q=np.maximum.accumulate(q); q[-1]+=1e-12
    a["delta_bin"]=np.clip(np.searchsorted(q[1:-1],a.delta_abs,side="right"),0,3)
    return a

def fixed_scores(a, key, alpha, held=None):
    if held is not None: a=a[a.seed!=held]
    b=a.baseline.to_numpy(float); d=a.expert.to_numpy(float)-b; y=a.truth.to_numpy(float); k=a[key].to_numpy(int)
    p=b+np.asarray(alpha)[k]*d
    return rmse(y,p)

def eval_grid(a,key,vals):
    # SSE is quadratic in each bin's alpha; pre-aggregate sufficient
    # statistics so even the 5-bin Cartesian grid is instantaneous.
    vals=np.asarray(vals,float); n_bins=int(a[key].max()+1); seeds=sorted(a.seed.unique())
    b=a.baseline.to_numpy(float); d=a.expert.to_numpy(float)-b; r=a.truth.to_numpy(float)-b; k=a[key].to_numpy(int)
    stats={}
    for s in ["all"]+seeds:
        m=np.ones(len(a),bool) if s=="all" else (a.seed.to_numpy()==s)
        srr=np.zeros(n_bins); sdr=np.zeros(n_bins); sdd=np.zeros(n_bins)
        for j in range(n_bins):
            z=m&(k==j); srr[j]=np.dot(r[z],r[z]); sdr[j]=np.dot(d[z],r[z]); sdd[j]=np.dot(d[z],d[z])
        stats[s]=(srr,sdr,sdd)
    rec=[]
    for al in itertools.product(vals,repeat=n_bins):
        aa=np.asarray(al,float); row={"key":key,"alpha":"/".join(f"{x:.3f}" for x in aa)}
        for s in ["all"]+seeds:
            srr,sdr,sdd=stats[s]; row["pooled_rmse" if s=="all" else f"rmse_{int(s)}"]=float(np.sqrt(np.sum(srr-2*aa*sdr+aa*aa*sdd)/(len(a) if s=="all" else int((a.seed==s).sum()))))
        rec.append(row)
    return pd.DataFrame(rec)

def fit_piece(cal,key,n_bins,grid_step=.025,shrink=0.,min_n=20):
    """Fit independent group slopes with optional global shrinkage."""
    vals=np.zeros(n_bins); d=cal.expert.to_numpy(float)-cal.baseline.to_numpy(float); r=cal.truth.to_numpy(float)-cal.baseline.to_numpy(float); k=cal[key].to_numpy(int); ok=np.isfinite(d)&np.isfinite(r)
    glob=float(np.clip(np.dot(d[ok],r[ok])/max(np.dot(d[ok],d[ok]),1e-12),0,1))
    vals[:]=glob
    for j in range(n_bins):
        z=ok&(k==j)
        if z.sum()<min_n: continue
        den=float(np.dot(d[z],d[z])); num=float(np.dot(d[z],r[z])); scale=max(den/z.sum(),1e-10)
        vals[j]=float(np.clip((num+shrink*glob*scale)/(den+shrink*scale),0,1))
    return vals,glob

def nested(a,key,n_bins,shrink,round_step=.025):
    rec=[]; pred=[]
    for held in sorted(a.seed.unique()):
        cal=a[a.seed!=held]; te=a[a.seed==held]; vals,g=fit_piece(cal,key,n_bins,shrink=shrink)
        vr=np.clip(np.round(vals/round_step)*round_step,0,1)
        d=te.expert.to_numpy(float)-te.baseline.to_numpy(float); p=te.baseline.to_numpy(float)+vr[te[key].to_numpy(int)]*d
        rec.append({"key":key,"shrink":shrink,"held_seed":int(held),"alpha_fit":"/".join(f"{x:.4f}" for x in vals),"alpha_round":"/".join(f"{x:.3f}" for x in vr),"rmse":rmse(te.truth,p),"rmse_base040":rmse(te.truth,.6*te.baseline+.4*te.expert)})
        pred.append((te.truth.to_numpy(float),p))
    return rec

def main():
    a=load(); grid=[]
    # Coarse/fine 3-bin grid.  Values around the empirically useful region are
    # denser while retaining the full [0,.8] range.
    vals=np.unique(np.r_[np.arange(0,.801,.025),[.825,.85, .9, .95,1.0]])
    g3=eval_grid(a,"bin3",vals); g3["n_bins"]=3; grid.append(g3)
    # 4/5-bin grids restricted to a smaller but still broad range to keep the
    # Cartesian search bounded.
    vsmall=np.arange(.10,.701,.05)
    g4=eval_grid(a,"bin4",vsmall); g4["n_bins"]=4; grid.append(g4)
    g5=eval_grid(a,"bin5",vsmall); g5["n_bins"]=5; grid.append(g5)
    gd=pd.concat(grid,ignore_index=True)
    seeds=sorted(a.seed.unique()); base=a.baseline.to_numpy(float)+.4*(a.expert.to_numpy(float)-a.baseline.to_numpy(float)); y=a.truth.to_numpy(float)
    # Keep top pooled and all-fold-improving policies.
    mask=np.ones(len(gd),bool)
    for s in seeds: mask &= gd[f"rmse_{int(s)}"].to_numpy() < rmse(y[a.seed.to_numpy()==s],base[a.seed.to_numpy()==s])
    gd["all_fold_improve"]=mask
    gd.sort_values(["pooled_rmse"],inplace=True); gd.to_csv(OUT_GRID,index=False,float_format="%.10f")
    loo=[]
    for key,n in [("bin3",3),("bin4",4),("bin5",5),("delta_bin",4)]:
        for sh in (0,20,50,100,200,400,800): loo.extend(nested(a,key,n,sh))
    lo=pd.DataFrame(loo); lo.to_csv(OUT_LOO,index=False,float_format="%.10f")
    # Aggregate nested policies.
    agg=[]
    for (key,sh),z in lo.groupby(["key","shrink"]): agg.append({"key":key,"shrink":sh,"pooled_rmse":float(np.sqrt(np.mean(z.rmse.to_numpy()**2))),"mean_fold_rmse":z.rmse.mean(),"worst":z.rmse.max(),"all_improve":bool((z.rmse<z.rmse_base040).all()),"per_seed":";".join(f"{int(r.held_seed)}:{r.rmse:.6f}" for _,r in z.sort_values("held_seed").iterrows())})
    ag=pd.DataFrame(agg).sort_values("pooled_rmse")
    top3=gd.head(20); topall=gd[gd.all_fold_improve].head(20)
    report=["# Route-v2 alpha grid and nested LOO audit (2026-09-05)","",f"Masks: {seeds}; rows={len(a)}. Raw routed expert recovered from persisted alpha=.40 blend. `true_src` is not used for any policy.","", "## Best fixed 3-bin policies (pooled across masks)","",top3[top3.n_bins==3].to_string(index=False),"","## Best fixed policies improving every mask", "",topall.head(30).to_string(index=False),"","## Nested LOO group-slope policies", "",ag.to_string(index=False),"","Artifacts:",f"- `{OUT_GRID.relative_to(ROOT).as_posix()}`",f"- `{OUT_LOO.relative_to(ROOT).as_posix()}`","No existing candidate overwritten; no submission emitted."]
    REPORT.write_text("\n".join(report)+"\n",encoding="utf-8")
    print(top3[top3.n_bins==3].to_string(index=False)); print("\nall improve:\n",topall.head(20).to_string(index=False)); print("\nLOO:\n",ag.to_string(index=False))

if __name__=="__main__": main()

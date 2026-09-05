"""Independent LOO audit of piecewise source-expert blend weights.

The saved route-v2 rows contain alpha=.40 blends, so the underlying routed
expert is recovered algebraically.  This audit tests fixed and calibration-
selected alpha policies by near-distance (and optional year/cohort) bins.  A
held mask's coefficients are fitted on the other masks only; no ``true_src``
or hidden target is used as a feature.  If seed=2 artifacts exist they are
included as an additional independent stress test.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
BASEFILE = R / "source_expert_route_v2_rows.csv"
SEED2FILE = R / "source_expert_route_v2_seed2_rows.csv"
OUT_MET = R / "source_expert_route_v2_piecewise_metrics_20260905.csv"
OUT_GRID = R / "source_expert_route_v2_piecewise_grid_20260905.csv"
OUT_REPORT = ROOT / "reports" / "source_expert_route_v2_piecewise_report_20260905.md"


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def load_rows():
    d = pd.read_csv(BASEFILE, parse_dates=["date"], low_memory=False)
    d["route"] = "crop_hier_n1_p67"
    d["expert"] = (d["blend_crop_hier_n1_p67_0.40"] - .60*d["baseline"])/.40
    # Add the other saved route experts as independent route choices.
    extras = {
        "soft_all_r1_l0_p4": "blend_soft_all_r1_l0_p4_0.40",
        "soft_crop_r1_l0_p4": "blend_soft_crop_r1_l0_p4_0.40",
        "soft_all_r2_l0_p4": "blend_soft_all_r2_l0_p4_0.40",
        "soft_all_r32_l0_p4": "blend_soft_all_r32_l0_p4_0.40",
        "post_mode": "blend_post_mode_0.40",
    }
    parts=[]
    keep=["anon_polygon_id","date","truth","year","cohort","near_dist","seed","baseline"]
    for route,col in {"crop_hier_n1_p67":"blend_crop_hier_n1_p67_0.40",**extras}.items():
        z=d[keep].copy(); z["route"]=route; z["expert"]=(d[col]-.60*d.baseline)/.40; parts.append(z)
    if SEED2FILE.exists():
        s=pd.read_csv(SEED2FILE,parse_dates=["date"],low_memory=False)
        if "seed" not in s:
            s["seed"] = 2
        # Seed-2 rows are saved with one column per route expert.
        for route in ["crop_hier_n1_p67","soft_all_r1_l0_p4","soft_crop_r1_l0_p4","soft_all_r2_l0_p4","soft_all_r32_l0_p4","post_mode"]:
            ec=f"expert_{route}"; bc=f"blend_{route}_0.40"
            if ec not in s: continue
            z=s[["anon_polygon_id","date","truth","year","cohort","near_dist","seed","baseline"]].copy()
            z["route"]=route; z["expert"]=s[ec].to_numpy(float); parts.append(z)
    out=pd.concat(parts,ignore_index=True)
    # Keep each route/seed as a separate panel.  For the main route, seed2 is
    # optional and can be absent while its long-running fit is in progress.
    out["dist_bin"] = pd.cut(out.near_dist,[-np.inf,2,8,np.inf],labels=["near","mid","far_or_none"]).astype(str)
    out["year_bin"] = np.where(out.year.astype(int)==2025,"2025","history")
    out["delta_abs"] = np.abs(out.expert-out.baseline)
    # Fixed bins from all rows ensure no fold-specific quantile leakage.
    q=np.nanquantile(out.delta_abs,[0,.25,.5,.75,1]); q=np.maximum.accumulate(q); q[-1]+=1e-12
    out["delta_bin"] = np.clip(np.searchsorted(q[1:-1],out.delta_abs,side="right"),0,3)
    out["year_dist"] = out.year_bin+"_"+out.dist_bin
    out["cohort_dist"] = out.cohort.astype(str)+"_"+out.dist_bin
    out["delta_dist"] = out.delta_bin.astype(str)+"_"+out.dist_bin
    return out


def choose_alpha(cal, key=None, shrink=0.0, bounds=(0.,1.)):
    """Choose group slopes on calibration rows, optionally shrink to global."""
    d=cal.expert.to_numpy(float)-cal.baseline.to_numpy(float); r=cal.truth.to_numpy(float)-cal.baseline.to_numpy(float)
    ok=np.isfinite(d)&np.isfinite(r)
    den=float(np.dot(d[ok],d[ok])); glob=float(np.clip(np.dot(d[ok],r[ok])/den if den>1e-12 else .4,*bounds))
    if key is None: return glob,glob,{}
    mp={}
    for g,ix in cal.groupby(key,dropna=False).groups.items():
        z=cal.loc[np.asarray(ix)]; dz=z.expert.to_numpy(float)-z.baseline.to_numpy(float); rz=z.truth.to_numpy(float)-z.baseline.to_numpy(float); good=np.isfinite(dz)&np.isfinite(rz)
        if good.sum()<30: continue
        dd=float(np.dot(dz[good],dz[good])); nn=float(np.dot(dz[good],rz[good]));
        # ``shrink`` is equivalent to shrink observations at global slope with
        # the group's observed d^2/n scale.
        scale=max(dd/good.sum(),1e-10); a=(nn+shrink*glob*scale)/(dd+shrink*scale)
        mp[g]=float(np.clip(a,*bounds))
    return glob,glob,mp


def apply_policy(cal,te,policy):
    d=te.expert.to_numpy(float)-te.baseline.to_numpy(float); b=te.baseline.to_numpy(float)
    if policy[0]=="fixed":
        a=np.asarray([policy[1].get(v,policy[2]) for v in te.dist_bin],float)
    elif policy[0]=="fixed_year_dist":
        a=np.asarray([policy[1].get(v,policy[2]) for v in te.year_dist],float)
    elif policy[0]=="fixed_cohort_dist":
        a=np.asarray([policy[1].get(v,policy[2]) for v in te.cohort_dist],float)
    elif policy[0]=="cal_dist":
        glob,_,mp=choose_alpha(cal,"dist_bin",shrink=policy[1]); a=np.asarray([mp.get(v,glob) for v in te.dist_bin],float)
    elif policy[0]=="cal_year_dist":
        glob,_,mp=choose_alpha(cal,"year_dist",shrink=policy[1]); a=np.asarray([mp.get(v,glob) for v in te.year_dist],float)
    elif policy[0]=="cal_cohort_dist":
        glob,_,mp=choose_alpha(cal,"cohort_dist",shrink=policy[1]); a=np.asarray([mp.get(v,glob) for v in te.cohort_dist],float)
    elif policy[0]=="cal_delta":
        glob,_,mp=choose_alpha(cal,"delta_bin",shrink=policy[1]); a=np.asarray([mp.get(v,glob) for v in te.delta_bin],float)
    elif policy[0]=="cal_delta_dist":
        glob,_,mp=choose_alpha(cal,"delta_dist",shrink=policy[1]); a=np.asarray([mp.get(v,glob) for v in te.delta_dist],float)
    else: raise ValueError(policy)
    return b+a*d


def main():
    d=load_rows(); records=[]; pooled=[]
    fixed_policies={
        "fixed_050_040_030":("fixed",{"near":.50,"mid":.40,"far_or_none":.30},.40),
        "fixed_050_040_025":("fixed",{"near":.50,"mid":.40,"far_or_none":.25},.40),
        "fixed_055_040_025":("fixed",{"near":.55,"mid":.40,"far_or_none":.25},.40),
        "fixed_050_035_025":("fixed",{"near":.50,"mid":.35,"far_or_none":.25},.40),
        "fixed_045_040_025":("fixed",{"near":.45,"mid":.40,"far_or_none":.25},.40),
    }
    cal_policies={f"cal_dist_s{q}":("cal_dist",q) for q in (20,50,100,200,400)}
    cal_policies.update({f"cal_year_dist_s{q}":("cal_year_dist",q) for q in (50,100,200,400)})
    cal_policies.update({f"cal_cohort_dist_s{q}":("cal_cohort_dist",q) for q in (50,100,200,400)})
    cal_policies.update({f"cal_delta_s{q}":("cal_delta",q) for q in (50,100,200,400)})
    cal_policies.update({f"cal_delta_dist_s{q}":("cal_delta_dist",q) for q in (100,250,500)})
    for route,dr in d.groupby("route",sort=False):
        for held in sorted(dr.seed.unique()):
            te=dr[dr.seed==held].copy(); cal=dr[dr.seed!=held].copy()
            # Fixed policies are route-specific but coefficient-free.
            policies={"base_a040":None}; policies.update(fixed_policies); policies.update(cal_policies)
            for name,pol in policies.items():
                if pol is None: pred=.6*te.baseline.to_numpy(float)+.4*te.expert.to_numpy(float)
                else: pred=apply_policy(cal,te,pol)
                records.append({"route":route,"held_seed":int(held),"policy":name,"n":len(te),"rmse":rmse(te.truth,pred),"bias":float(np.mean(pred-te.truth.to_numpy(float))),"mae":float(np.mean(np.abs(pred-te.truth.to_numpy(float))))})
    met=pd.DataFrame(records); met.to_csv(OUT_MET,index=False,float_format="%.10f")
    pool=[]
    for (route,pol),g in met.groupby(["route","policy"]):
        pool.append({"route":route,"policy":pol,"n":int(g.n.sum()),"mean_fold_rmse":float(g.rmse.mean()),"worst_fold_rmse":float(g.rmse.max()),"folds_improved_vs_a040":int((g.rmse < met[(met.route==route)&(met.policy=="base_a040")].rmse.to_numpy()).sum()) if len(g)==len(met[(met.route==route)&(met.policy=="base_a040")]) else -1})
    # Recompute true pooled RMSE directly from rows by rerunning predictions for
    # each policy, because fold RMSE averaging is not a pooled metric.
    pool=[]
    for route,dr in d.groupby("route",sort=False):
        for name,pol in {"base_a040":None,**fixed_policies,**cal_policies}.items():
            ys=[]; ps=[]
            for held in sorted(dr.seed.unique()):
                te=dr[dr.seed==held].copy(); cal=dr[dr.seed!=held].copy();
                pred=.6*te.baseline.to_numpy(float)+.4*te.expert.to_numpy(float) if pol is None else apply_policy(cal,te,pol)
                ys.extend(te.truth.to_numpy(float)); ps.extend(pred)
            pool.append({"route":route,"policy":name,"n":len(ys),"pooled_rmse":rmse(ys,ps),"per_seed":";".join(f"{int(s)}:{met[(met.route==route)&(met.held_seed==s)&(met.policy==name)].rmse.iloc[0]:.6f}" for s in sorted(dr.seed.unique()))})
    pg=pd.DataFrame(pool).sort_values("pooled_rmse"); pg.to_csv(OUT_GRID,index=False,float_format="%.10f")
    focus=pg[pg.route=="crop_hier_n1_p67"].head(30)
    lines=["# Source-expert route v2 piecewise alpha audit (2026-09-05)","","Underlying routed experts recovered from saved alpha=.40 blends. For each held seed, coefficients are fit on the other masks only; no true source is used.","Seed=2 is included automatically if `source_expert_route_v2_seed2_rows.csv` exists.","","## Main route pooled LOO", "",focus.to_string(index=False),"","## All route pooled results", "",pg.head(60).to_string(index=False),"","## Per-fold metrics (main route)","",met[met.route=="crop_hier_n1_p67"].to_string(index=False),"","Artifacts:",f"- `{OUT_MET.relative_to(ROOT).as_posix()}`",f"- `{OUT_GRID.relative_to(ROOT).as_posix()}`","No existing candidate overwritten; no submission emitted."]
    OUT_REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(focus.to_string(index=False)); print("Seeds:",sorted(d.seed.unique()))


if __name__=="__main__": main()

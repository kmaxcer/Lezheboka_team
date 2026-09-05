"""Combine source-route v2 masks (including fresh seed=2) and audit policies.

The first route-v2 run saved seeds 0/1/70404.  This script normalizes those
rows with the fresh seed=2 table, reconstructs route expert predictions from
the alpha=.40 blends, and evaluates fixed/conditional alpha policies under
leave-one-mask-out checks.  Labels are used only for this offline score.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"
ROUTES = ["crop_hier_n1_p67", "soft_all_r1_l0_p4", "soft_all_r2_l0_p4", "soft_all_r32_l0_p4", "post_mode"]


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan


def load() -> pd.DataFrame:
    a = pd.read_csv(R/"source_expert_route_v2_rows.csv", parse_dates=["date"], low_memory=False)
    b = pd.read_csv(R/"source_expert_route_v2_seed2_rows.csv", parse_dates=["date"], low_memory=False)
    # Both files carry alpha=.40 blends with identical route names.  Keep the
    # common observable/evaluation columns and enforce numeric types.
    cols = ["anon_polygon_id", "date", "truth", "true_src", "year", "cohort", "near_dist", "baseline", "seed"]
    for c in cols[:-1]:
        if c not in b: raise RuntimeError(f"seed2 missing {c}")
    out = []
    for z in (a, b):
        take_cols = [c for c in cols if c in z.columns]
        q = z[take_cols].copy()
        # The seed2 compact sidecar predates the explicit seed column; its
        # provenance is unambiguous from the filename.
        if "seed" not in z.columns:
            q["seed"] = 2
        for route in ROUTES:
            c = f"blend_{route}_0.40"
            if c not in z: raise RuntimeError(f"missing {c}")
            q[f"e_{route}"] = (z[c].to_numpy(float) - .60*z["baseline"].to_numpy(float))/.40
        out.append(q)
    return pd.concat(out, ignore_index=True)


def masks(df):
    yr = df.year.to_numpy(int); co = df.cohort.astype(str).to_numpy(); nd = df.near_dist.to_numpy(float); src = df.true_src.astype(str).to_numpy()
    near = np.isfinite(nd) & (nd <= 2); mid = np.isfinite(nd) & (nd > 2) & (nd <= 8); far = (~np.isfinite(nd)) | (nd > 8)
    return {"all": np.ones(len(df), bool), "near": near, "mid": mid, "far": far,
            "history": yr < 2025, "2025": yr == 2025, "new": co == "new", "shared": co == "shared",
            "new2025": (co == "new") & (yr == 2025), "shared2025": (co == "shared") & (yr == 2025),
            "new2025_near": (co == "new") & (yr == 2025) & near, "new2025_far": (co == "new") & (yr == 2025) & far,
            "shared2025_near": (co == "shared") & (yr == 2025) & near, "shared2025_far": (co == "shared") & (yr == 2025) & far,
            "source_s2": src == "s2", "source_landsat": src == "landsat", "source_modis": src == "modis"}


def alpha_opt(y, b, e):
    d = e-b; den = float(np.dot(d,d)); return float(np.clip(np.dot(d,y-b)/den, 0., 1.)) if den > 1e-12 else 0.


def policy_pred(df, route, policy):
    b = df.baseline.to_numpy(float); e = df[f"e_{route}"].to_numpy(float); yr = df.year.to_numpy(int); co = df.cohort.astype(str).to_numpy(); nd = df.near_dist.to_numpy(float)
    near = np.isfinite(nd) & (nd <= 2); mid = np.isfinite(nd) & (nd > 2) & (nd <= 8); far = (~np.isfinite(nd)) | (nd > 8)
    if policy == "fixed040": a = np.full(len(df), .40)
    elif policy == "fixed045": a = np.full(len(df), .45)
    elif policy == "distance_50_40_30": a = np.where(near, .50, np.where(mid, .40, .30))
    elif policy == "distance_50_45_25": a = np.where(near, .50, np.where(mid, .45, .25))
    elif policy == "distance_45_40_30": a = np.where(near, .45, np.where(mid, .40, .30))
    elif policy.startswith("new2025_"):
        # Vary only new-2025; retain fixed .40 elsewhere.
        v = float(policy.split("_")[-1]); a = np.where((co == "new") & (yr == 2025), v, .40)
    elif policy.startswith("dist_new2025_"):
        # Distance policy for all rows, with an explicit new-2025 override.
        v = float(policy.split("_")[-1]); a = np.where(near, .50, np.where(mid, .40, .30)); a = np.where((co == "new") & (yr == 2025), v, a)
    elif policy == "cohort_year":
        # Conservative shared-2025/history/far values; new 2025 gets more
        # expert weight only if the independent masks support it.
        a = np.where((co == "new") & (yr == 2025), .60, np.where((co == "shared") & (yr == 2025), .35, .40))
    elif policy == "cohort_year_dist":
        a = np.where(near, .50, np.where(mid, .40, .30)); a = np.where((co == "new") & (yr == 2025), .60, a); a = np.where((co == "shared") & (yr == 2025), .35, a)
    else: raise ValueError(policy)
    return (1-a)*b + a*e


def main():
    df = load(); y = df.truth.to_numpy(float); seeds = sorted(df.seed.unique().astype(int)); ms = masks(df)
    policies = ["fixed040", "fixed045", "distance_50_40_30", "distance_50_45_25", "distance_45_40_30", "cohort_year", "cohort_year_dist"] + [f"new2025_{v:.2f}" for v in (.45,.50,.55,.60,.65,.70)] + [f"dist_new2025_{v:.2f}" for v in (.50,.55,.60,.65,.70)]
    rec=[]
    for route in ROUTES:
        for pol in policies:
            p = policy_pred(df, route, pol)
            rec.append({"route": route, "policy": pol, "slice": "all", "n": len(df), "rmse": rmse(y,p), "rmse_base": rmse(y,df.baseline)})
            for sl,m in ms.items():
                if sl == "all" or m.sum() < 10: continue
                rec.append({"route": route, "policy": pol, "slice": sl, "n": int(m.sum()), "rmse": rmse(y[m],p[m]), "rmse_base": rmse(y[m],df.baseline.to_numpy(float)[m])})
    # LOO: choose each policy on 3 masks, then score held mask.  We report all
    # fixed policies and identify the minimum training RMSE; labels remain
    # confined to this audit.
    loo=[]
    for route in ROUTES:
        for held in seeds:
            trm = df.seed.to_numpy(int) != held; tem = ~trm
            vals=[]
            for pol in policies:
                p = policy_pred(df, route, pol); vals.append((rmse(y[trm],p[trm]), pol, p))
            vals.sort(key=lambda x:x[0]); best=vals[0][1]
            for pol in [best, "fixed040", "distance_50_40_30", "cohort_year_dist"]:
                p = policy_pred(df, route, pol)
                loo.append({"route":route,"held_seed":int(held),"selected_train_policy":best,"policy":pol,"train_rmse":rmse(y[trm],p[trm]),"test_rmse":rmse(y[tem],p[tem]),"test_base":rmse(y[tem],df.baseline.to_numpy(float)[tem])})
    # Analytic alpha by slice and mask for the principal route.
    ao=[]
    for route in ROUTES[:3]:
        e=df[f"e_{route}"].to_numpy(float); b=df.baseline.to_numpy(float)
        for sl,m in ms.items():
            if m.sum()<10: continue
            ao.append({"route":route,"slice":sl,"n":int(m.sum()),"alpha_opt":alpha_opt(y[m],b[m],e[m]),"rmse_opt":rmse(y[m],(1-alpha_opt(y[m],b[m],e[m]))*b[m]+alpha_opt(y[m],b[m],e[m])*e[m]),"rmse040":rmse(y[m],.6*b[m]+.4*e[m]),"rmse_base":rmse(y[m],b[m])})
    pd.DataFrame(rec).to_csv(R/"source_expert_route_v2_seed2_policy_metrics.csv",index=False,float_format="%.10f")
    pd.DataFrame(loo).to_csv(R/"source_expert_route_v2_seed2_policy_loo.csv",index=False,float_format="%.10f")
    pd.DataFrame(ao).to_csv(R/"source_expert_route_v2_seed2_policy_alpha.csv",index=False,float_format="%.10f")
    best = pd.DataFrame(rec).query("slice == 'all'").sort_values("rmse").head(30)
    lines=["# Source-route v2 four-mask policy audit", "", f"Masks: {seeds}; n={len(df)}. Source labels are scoring-only. Seed2 baseline is independently rebuilt.", "", "## Pooled policy shortlist", "", best.to_string(index=False), "", "## LOO policy checks", "", pd.DataFrame(loo).to_string(index=False), "", "## Analytic alpha slices", "", pd.DataFrame(ao).query("route == 'crop_hier_n1_p67'").to_string(index=False), "", "No candidate was overwritten."]
    (R/"source_expert_route_v2_seed2_policy_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(best.to_string(index=False)); print(pd.DataFrame(loo).to_string(index=False)); print(pd.DataFrame(ao).query("route == 'crop_hier_n1_p67'").to_string(index=False))


if __name__ == "__main__": main()

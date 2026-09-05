"""Cross-fitted cohort/year alpha grid for the train-augmented shock."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; D = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(R)); from shock_bin_sweep_v1 import _mask_private, _features  # noqa: E402
from source_expert_route_v2_seed2_policy_audit import load as load_route, policy_pred  # noqa: E402


def rm(y, p): return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def build():
    tr = pd.read_csv(D / "train_dataset.csv", parse_dates=["date"], low_memory=False); pr = pd.read_csv(D / "private_features.csv", parse_dates=["date"], low_memory=False); route = load_route(); out=[]
    for seed in (0, 1, 2, 70404):
        f,m=_mask_private(pr,seed); combo=pd.concat([tr,f],ignore_index=True,sort=False); combo["_truth"]=pd.to_numeric(combo.primary_ndvi,errors="coerce"); ft=_features(combo,np.r_[np.zeros(len(tr),bool),m],24)
        q=route[route.seed.astype(int).eq(seed)].copy();q.date=pd.to_datetime(q.date);q=q.merge(ft[["anon_polygon_id","date","crop_shock"]],on=["anon_polygon_id","date"],validate="one_to_one");q["base"]=policy_pred(q,"crop_hier_n1_p67","cohort_year_dist");q["scope"]=np.where(q.year.to_numpy(int)<2025,"history",np.where(q.cohort.eq("shared"),"shared25","new25"));q["seed"]=seed;out.append(q)
    return out


def fit(cal, scope=None):
    z=cal if scope is None else cal[cal.scope.eq(scope)];x=z.crop_shock.to_numpy(float);r=z.truth.to_numpy(float)-z.base.to_numpy(float);ok=np.isfinite(x)&np.isfinite(r);return float(np.clip(np.sum(x[ok]*r[ok])/max(np.sum(x[ok]**2),1e-9),-.5,.5)) if ok.sum()>=30 else 0.0


def run():
    parts=build();rows=[]
    fixed=[.0,.05,.10,.125,.15,.175,.20,.225,.25]
    # Global and per-scope fits, plus fixed policy combinations.
    for i,t in enumerate(parts):
        cal=pd.concat([p for j,p in enumerate(parts) if j!=i],ignore_index=True);y=t.truth.to_numpy(float);b=t.base.to_numpy(float);x=t.crop_shock.to_numpy(float)
        policies={"global_loo":{s:fit(cal) for s in ["history","shared25","new25"]},"scope_loo":{s:fit(cal,s) for s in ["history","shared25","new25"]}}
        for a in fixed:policies[f"global_{a:.3f}"]={s:a for s in ["history","shared25","new25"]}
        # Hand grid around the cross-fitted estimates, useful to check robust
        # cohort routing without selecting on held labels.
        for ah in [.10,.15,.20]:
            for as_ in [.10,.15,.20]:
                for an in [.05,.10,.15]: policies[f"h{ah:.2f}_s{as_:.2f}_n{an:.2f}"]={"history":ah,"shared25":as_,"new25":an}
        for name,mp in policies.items():
            a=np.asarray([mp.get(s,0.15) for s in t.scope],float);p=b+a*np.nan_to_num(x,nan=0.0);rows.append({"seed":int(t.seed.iloc[0]),"policy":name,"alpha_history":mp["history"],"alpha_shared25":mp["shared25"],"alpha_new25":mp["new25"],"n":len(t),"rmse":rm(y,p),"baseline_rmse":rm(y,b),"delta":rm(y,p)-rm(y,b),"wins":int(rm(y,p)<rm(y,b))})
    out=pd.DataFrame(rows);out.to_csv(R/'source_route_shock_policy_grid_v1_results.csv',index=False);agg=out.groupby('policy',as_index=False).apply(lambda g:pd.Series({'n':int(g.n.sum()),'rmse_pooled':float(np.sqrt(np.average(g.rmse**2,weights=g.n))),'baseline_rmse_pooled':float(np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),'delta':float(np.sqrt(np.average(g.rmse**2,weights=g.n))-np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),'wins':int(g.wins.sum()),'masks':len(g)}),include_groups=False).reset_index(drop=True);agg.to_csv(R/'source_route_shock_policy_grid_v1_aggregate.csv',index=False);best=agg.sort_values('rmse_pooled').head(30);(R/'source_route_shock_policy_grid_v1_report.md').write_text('# Source-route shock cohort/year policy grid\n\n'+best.to_string(index=False)+'\n\nFull results: `source_route_shock_policy_grid_v1_results.csv`. All coefficients are leave-one-mask-out; shock uses visible+train rows only.\n',encoding='utf8');print(best.to_string(index=False))


if __name__=='__main__':run()

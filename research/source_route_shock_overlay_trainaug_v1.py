"""Four-mask audit of source-route + 24-day shock with train-row augmentation."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; D = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(R))
from source_expert_route_v2_seed2_policy_audit import load as load_route, policy_pred  # noqa: E402
from shock_bin_sweep_v1 import _mask_private, _features  # noqa: E402


def rm(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); return float(np.sqrt(np.mean((p - y) ** 2)))


def run():
    tr = pd.read_csv(D / "train_dataset.csv", parse_dates=["date"], low_memory=False); pr = pd.read_csv(D / "private_features.csv", parse_dates=["date"], low_memory=False)
    routes = load_route(); parts = []
    for seed in (0, 1, 2, 70404):
        f, m = _mask_private(pr, seed); combo = pd.concat([tr, f], ignore_index=True, sort=False); combo["_truth"] = pd.to_numeric(combo.primary_ndvi, errors="coerce"); mc = np.r_[np.zeros(len(tr), bool), m]; ft = _features(combo, mc, 24)
        q = routes[routes.seed.astype(int).eq(seed)].copy(); q.date = pd.to_datetime(q.date); q = q.merge(ft[["anon_polygon_id", "date", "crop_shock"]], on=["anon_polygon_id", "date"], validate="one_to_one"); q["route_pred"] = policy_pred(q, "crop_hier_n1_p67", "cohort_year_dist"); q["seed"] = seed; parts.append(q)
    rows=[]
    for i,t in enumerate(parts):
        c=pd.concat([p for j,p in enumerate(parts) if j!=i]);x=c.crop_shock.to_numpy(float);r=c.truth.to_numpy(float)-c.route_pred.to_numpy(float);ok=np.isfinite(x)&np.isfinite(r);a=float(np.clip(np.sum(x[ok]*r[ok])/max(np.sum(x[ok]**2),1e-9),-.8,.8));xx=t.crop_shock.to_numpy(float);b=t.route_pred.to_numpy(float);y=t.truth.to_numpy(float); 
        for variant,aa in [("loo",a),("fixed010",.10),("fixed015",.15),("fixed020",.20),("fixed025",.25)]:
            p=b.copy();good=np.isfinite(xx);p[good]+=aa*xx[good];rows.append({"seed":int(t.seed.iloc[0]),"variant":variant,"alpha":aa,"n":len(t),"rmse":rm(y,p),"baseline_rmse":rm(y,b),"delta":rm(y,p)-rm(y,b)})
        for scope,sel in {"2025":t.year.to_numpy()==2025,"new2025":(t.year.to_numpy()==2025)&t.cohort.eq('new').to_numpy(),"shared2025":(t.year.to_numpy()==2025)&t.cohort.eq('shared').to_numpy(),"history":t.year.to_numpy()<2025}.items():
            p=b.copy();good=sel&np.isfinite(xx);p[good]+=a*xx[good];rows.append({"seed":int(t.seed.iloc[0]),"variant":"loo_"+scope,"alpha":a,"n":int(sel.sum()),"rmse":rm(y[sel],p[sel]),"baseline_rmse":rm(y[sel],b[sel]),"delta":rm(y[sel],p[sel])-rm(y[sel],b[sel])})
    out=pd.DataFrame(rows);out.to_csv(R/'source_route_shock_overlay_trainaug_v1_results.csv',index=False);main=out[out.variant.isin(['loo','fixed010','fixed015','fixed020','fixed025'])];agg=main.groupby('variant',as_index=False).apply(lambda g:pd.Series({'n':int(g.n.sum()),'rmse_pooled':float(np.sqrt(np.average(g.rmse**2,weights=g.n))),'baseline_rmse_pooled':float(np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),'delta':float(np.sqrt(np.average(g.rmse**2,weights=g.n))-np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),'wins':int((g.rmse<g.baseline_rmse).sum())}),include_groups=False).reset_index(drop=True);agg.to_csv(R/'source_route_shock_overlay_trainaug_v1_aggregate.csv',index=False);sl=out[out.variant.str.startswith('loo_')];sl.to_csv(R/'source_route_shock_overlay_trainaug_v1_slices.csv',index=False);(R/'source_route_shock_overlay_trainaug_v1_report.md').write_text('# Source-route shock overlay with train augmentation\n\n'+agg.to_string(index=False)+'\n\nSlices\n'+sl.to_string(index=False)+'\n',encoding='utf8');print(agg.to_string(index=False));print(sl.to_string(index=False))


if __name__=='__main__':run()

"""Check whether adding visible train rows improves the 24-day shock profile."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; D = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(R)); from shock_bin_sweep_v1 import _mask_private, _features  # noqa: E402


def rm(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def run():
    tr = pd.read_csv(D / "train_dataset.csv", parse_dates=["date"], low_memory=False); pr = pd.read_csv(D / "private_features.csv", parse_dates=["date"], low_memory=False)
    base = pd.read_csv(R / "teammate_sweep_postcorr_preds.csv", parse_dates=["date"], low_memory=False); base = base[base.method.eq("blend_lag_0.20")]
    parts = []
    for seed in (0, 1, 2):
        f, m = _mask_private(pr, seed); combo = pd.concat([tr, f], ignore_index=True, sort=False); combo["_truth"] = pd.to_numeric(combo.primary_ndvi, errors="coerce"); mc = np.r_[np.zeros(len(tr), bool), m]; ft = _features(combo, mc, 24)
        q = f.loc[m, ["anon_polygon_id", "date", "_truth"]].copy().reset_index(drop=True); q = q.merge(base[base.partition.eq(f"random{seed}")][["anon_polygon_id", "date", "pred"]].rename(columns={"pred": "baseline"}), on=["anon_polygon_id", "date"], validate="one_to_one"); q = q.merge(ft[["anon_polygon_id", "date", "crop_shock"]], on=["anon_polygon_id", "date"], validate="one_to_one"); q["year"] = q.date.dt.year; q["cohort"] = np.where(q.anon_polygon_id.astype(str).isin(set(tr.anon_polygon_id.astype(str))), "shared", "new"); q["seed"] = seed; parts.append(q)
    rows=[]
    for i,t in enumerate(parts):
        c=pd.concat([p for j,p in enumerate(parts) if j!=i]); x=c.crop_shock.to_numpy(float);r=c._truth.to_numpy(float)-c.baseline.to_numpy(float);ok=np.isfinite(x)&np.isfinite(r);a=np.clip(np.sum(x[ok]*r[ok])/max(np.sum(x[ok]**2),1e-9),-.8,.8);xx=t.crop_shock.to_numpy(float);y=t._truth.to_numpy(float);b=t.baseline.to_numpy(float)
        for scope,sel in {"all":np.ones(len(t),bool),"2025":t.year.to_numpy()==2025,"shared25":(t.year.to_numpy()==2025)&t.cohort.eq("shared").to_numpy(),"new25":(t.year.to_numpy()==2025)&t.cohort.eq("new").to_numpy(),"history":t.year.to_numpy()<2025}.items():
            good=sel&np.isfinite(xx);p=b.copy();p[good]+=a*xx[good];rows.append({"seed":t.seed.iloc[0],"scope":scope,"n":int(sel.sum()),"finite":int(good.sum()),"alpha":a,"baseline_rmse":rm(y[sel],b[sel]),"corrected_rmse":rm(y[sel],p[sel])})
    out=pd.DataFrame(rows);out.to_csv(R/'shock_bin_trainaugment_v1_results.csv',index=False);po=out.groupby('scope',as_index=False).apply(lambda g:pd.Series({'n':int(g.n.sum()),'baseline_rmse':float(np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),'corrected_rmse':float(np.sqrt(np.average(g.corrected_rmse**2,weights=g.n))),'delta':float(np.sqrt(np.average(g.corrected_rmse**2,weights=g.n))-np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),'alphas':','.join(f'{x:.3f}' for x in g.alpha)}),include_groups=False).reset_index(drop=True);po.to_csv(R/'shock_bin_trainaugment_v1_aggregate.csv',index=False);(R/'shock_bin_trainaugment_v1_report.md').write_text('# 24-day shock with train augmentation\n\n'+po.to_string(index=False)+'\n',encoding='utf8');print(po.to_string(index=False))


if __name__=='__main__':run()

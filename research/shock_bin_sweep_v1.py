"""Sweep seasonal-profile widths for an observable common-date residual.

The earlier overnight evaluator fixed a 16-day profile bin.  This diagnostic
rebuilds the profile with several widths and evaluates date/date+crop shocks
and a local state correction under the same leave-partition-out protocol.
No production or prior artifact is overwritten.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
R = ROOT / "research"
sys.path.insert(0, str(R)); sys.path.insert(0, str(ROOT / "src"))
from teammate_sweep_postcorr import _mask_private  # noqa: E402
from validate import make_fold  # noqa: E402

KEY = ["anon_polygon_id", "date"]
BASE = "blend_lag_0.20"


def _seasonal(frame: pd.DataFrame, known: np.ndarray, width: int):
    d = frame.copy().reset_index(drop=True); d.date = pd.to_datetime(d.date)
    d["_yr"] = d.date.dt.year.astype(int); d["_doy"] = d.date.dt.dayofyear.astype(int)
    d["_bin"] = ((d._doy - 1) // int(width)).astype(int)
    y = pd.to_numeric(d.get("_truth", d.primary_ndvi), errors="coerce").to_numpy(float).copy()
    known = np.asarray(known, bool); y[~known] = np.nan
    obs = d.loc[known & np.isfinite(y), ["anon_polygon_id", "_yr", "_bin"]].copy(); obs["_y"] = y[obs.index]
    obs = obs[obs._y.between(-0.5, 1.2)]
    if obs.empty: return np.full(len(d), np.nan), np.full(len(d), np.nan)
    p1 = obs.groupby(["anon_polygon_id", "_yr", "_bin"], observed=True)._y.median().rename("p1").reset_index()
    p2 = obs.groupby(["anon_polygon_id", "_bin"], observed=True)._y.median().rename("p2").reset_index()
    p3 = obs.groupby(["_yr", "_bin"], observed=True)._y.median().rename("p3").reset_index()
    p4 = obs.groupby(["_bin"], observed=True)._y.median().rename("p4").reset_index()
    z = d[["anon_polygon_id", "_yr", "_bin"]].merge(p1, on=["anon_polygon_id", "_yr", "_bin"], how="left")
    z = z.merge(p2, on=["anon_polygon_id", "_bin"], how="left").merge(p3, on=["_yr", "_bin"], how="left").merge(p4, on=["_bin"], how="left")
    prof = z.p1.combine_first(z.p2).combine_first(z.p3).combine_first(z.p4).fillna(float(obs._y.median())).to_numpy(float)
    res = np.clip(y - prof, -0.5, 0.5); res[~known] = np.nan
    return prof, res


def _features(frame: pd.DataFrame, mask: np.ndarray, width: int) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True); d.date = pd.to_datetime(d.date)
    known = d.primary_ndvi.notna().to_numpy(bool) & ~d.is_synthetic_gap.fillna(False).to_numpy(bool)
    qi = np.flatnonzero(mask); ids = d.anon_polygon_id.astype(str).to_numpy(); dates = d.date.to_numpy()
    crop = d.get("crop_type", pd.Series("", index=d.index)).fillna("").astype(str).to_numpy()
    _, residual = _seasonal(d, known, width)
    # Deduplicate residuals by date/AOI/crop before aggregation.
    vi = np.flatnonzero(known & np.isfinite(residual))
    b = pd.DataFrame({"date": d.date.iloc[vi].to_numpy(), "id": ids[vi], "crop": crop[vi], "r": residual[vi]})
    if len(b): b = b.groupby(["date", "id", "crop"], as_index=False, observed=True).r.median()
    # Exact date and date+crop medians, requiring at least three independent
    # AOIs; query AOI is removed at application time.
    dm = b.groupby("date", observed=True).agg(shock=("r", "median"), n=("id", "nunique")).reset_index() if len(b) else pd.DataFrame(columns=["date", "shock", "n"])
    cm = b.groupby(["date", "crop"], observed=True).agg(cshock=("r", "median"), cn=("id", "nunique")).reset_index() if len(b) else pd.DataFrame(columns=["date", "crop", "cshock", "cn"])
    out = d.loc[qi, KEY].copy().reset_index(drop=True); out["idx"] = qi; out["crop"] = crop[qi]
    out = out.merge(dm, on="date", how="left").merge(cm, on=["date", "crop"], how="left")
    out.rename(columns={"shock": "date_shock", "n": "date_n", "cshock": "crop_shock", "cn": "crop_n"}, inplace=True)
    # Local residual state.  Explicit groups keep this independent of any
    # hidden/query targets.
    d["_yr"] = d.date.dt.year.astype(int); ords = d.date.map(pd.Timestamp.toordinal).to_numpy(float)
    groups = {}
    for key, ix0 in d.loc[known & np.isfinite(residual)].groupby(["anon_polygon_id", "_yr"], sort=False).groups.items():
        ix = np.asarray(ix0, int); groups[(str(key[0]), int(key[1]))] = ix[np.isfinite(residual[ix])]
    state = np.full(len(qi), np.nan); state_n = np.zeros(len(qi), int)
    for j, i in enumerate(qi):
        ix = groups.get((ids[i], int(d._yr.iat[i])))
        if ix is None or len(ix) < 2: continue
        dist = np.abs(ords[ix] - ords[i]); take = np.argsort(dist)[: min(10, len(ix))]; take = take[dist[take] <= 120]
        if len(take) >= 2:
            w = np.exp(-dist[take] / 45.0); state[j] = np.average(np.clip(residual[ix[take]], -0.3, 0.3), weights=w); state_n[j] = len(take)
    out["state"] = state; out["state_n"] = state_n
    # A date group containing the query AOI itself can bias the median only if
    # that AOI has visible rows on the same date.  In these data each AOI/date
    # key is unique and the query row is masked, so this is already excluded.
    out.loc[out.date_n < 3, "date_shock"] = np.nan; out.loc[out.crop_n < 3, "crop_shock"] = np.nan
    return out


def _predmap_random(seed: int):
    p = pd.read_csv(R / "teammate_sweep_postcorr_preds.csv", parse_dates=["date"], low_memory=False)
    return p[(p.partition == f"random{seed}") & (p.method == BASE)][KEY + ["pred"]].rename(columns={"pred": "baseline"})


def build_parts():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    parts = []
    for seed in (0, 1, 2):
        f, m = _mask_private(pr, seed); q = f.loc[m, KEY + ["_truth"]].copy().reset_index(drop=True); q.date = pd.to_datetime(q.date)
        q = q.merge(_predmap_random(seed), on=KEY, how="left", validate="one_to_one"); q["dataset"] = "random_private_like"; q["partition"] = f"random{seed}"
        parts.append((q, f, m))
    ex = pd.read_csv(R / "exact_compare_preds.csv", parse_dates=["date"])
    for yr, g in ex.groupby("year", sort=True):
        f, _ = make_fold(tr.copy(), pr.copy(), int(yr)); m = f.is_synthetic_gap.fillna(False).to_numpy(bool)
        keys = set(zip(g.anon_polygon_id.astype(str), pd.to_datetime(g.date))); mm = np.array([a and (str(i), pd.Timestamp(dt)) in keys for a, i, dt in zip(m, f.anon_polygon_id, f.date)], bool)
        q = f.loc[mm, KEY + ["_truth"]].copy().reset_index(drop=True); q = q.merge(g[KEY + ["hgb", "lag_k16_d3"]], on=KEY, how="left", validate="one_to_one"); q["baseline"] = .8*q.hgb + .2*q.lag_k16_d3; q["dataset"] = "exact_hidden_doy"; q["partition"] = f"exact{int(yr)}"; q.date = pd.to_datetime(q.date)
        parts.append((q, f, mm))
    return parts


def fitcoef(train: pd.DataFrame, col: str) -> float:
    x = train[col].to_numpy(float); y = train._truth.to_numpy(float) - train.baseline.to_numpy(float); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.sum(x[ok]**2) < 1e-8: return 0.0
    return float(np.clip(np.sum(x[ok]*y[ok])/np.sum(x[ok]**2), -0.8, 0.8))


def run():
    parts = build_parts(); rows=[]; predrows=[]
    for width in (8, 12, 16, 24, 32, 45):
        featparts=[]
        for q,f,m in parts:
            ft = _features(f,m,width); z=q.merge(ft.drop(columns=["idx"],errors="ignore"),on=KEY,how="left",validate="one_to_one"); featparts.append(z)
        for i,test in enumerate(featparts):
            same=[j for j,p in enumerate(featparts) if j!=i and p.dataset.iloc[0]==test.dataset.iloc[0]]
            tr=pd.concat([featparts[j] for j in same],ignore_index=True) if same else pd.concat([p for j,p in enumerate(featparts) if j!=i],ignore_index=True)
            y=test._truth.to_numpy(float); b=test.baseline.to_numpy(float)
            for name,col in [("date","date_shock"),("crop","crop_shock"),("state","state")]:
                a=fitcoef(tr,col); x=test[col].to_numpy(float); ok=np.isfinite(x); p=b.copy(); p[ok]+=a*x[ok]; p=np.clip(p,-.5,1.2); e=p-y; rows.append({"width":width,"dataset":test.dataset.iloc[0],"partition":test.partition.iloc[0],"method":name,"alpha":a,"n":len(test),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(np.abs(e))),"finite":int(ok.sum())})
            # Conservative fixed joint state/date fit (ridge) for a useful
            # interaction check; coefficients are learned off-partition.
            X=np.c_[tr.date_shock.fillna(0),tr.state.fillna(0)]; yy=tr._truth.to_numpy(float)-tr.baseline.to_numpy(float); ok=np.isfinite(yy); coef=np.linalg.solve(X[ok].T@X[ok]+.15*np.eye(2),X[ok].T@yy[ok]); coef=np.clip(coef,-.8,.8); Xt=np.c_[test.date_shock.fillna(0),test.state.fillna(0)]; p=np.clip(b+Xt@coef,-.5,1.2);e=p-y;rows.append({"width":width,"dataset":test.dataset.iloc[0],"partition":test.partition.iloc[0],"method":"joint","alpha":float(np.linalg.norm(coef)),"coef_date":coef[0],"coef_state":coef[1],"n":len(test),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(np.abs(e))),"finite":int(np.isfinite(test.date_shock).sum())})
            if width==16:
                for nm,col in [("date", "date_shock"),("crop","crop_shock"),("state","state")]:
                    z=test[KEY+['dataset','partition','_truth','baseline',col]].copy(); z['method']=nm; z['width']=width; predrows.append(z)
    res=pd.DataFrame(rows); res.to_csv(R/'shock_bin_sweep_v1_results.csv',index=False)
    if predrows: pd.concat(predrows,ignore_index=True).to_csv(R/'shock_bin_sweep_v1_preds.csv',index=False)
    agg=res.groupby(['width','dataset','method'],as_index=False).apply(lambda g:pd.Series({'n':int(g.n.sum()),'rmse_pooled':float(np.sqrt(np.average(g.rmse**2,weights=g.n))),'rmse_mean':float(g.rmse.mean()),'mae_mean':float(g.mae.mean())}),include_groups=False).reset_index(drop=True); agg.to_csv(R/'shock_bin_sweep_v1_aggregate.csv',index=False)
    lines=['# Seasonal shock-bin sweep v1','','Only visible targets build seasonal profiles; coefficients are leave-partition-out.','',agg.sort_values(['dataset','rmse_pooled']).to_string(index=False),'','Files: `research/shock_bin_sweep_v1_results.csv`, `research/shock_bin_sweep_v1_aggregate.csv`, `research/shock_bin_sweep_v1_preds.csv`']
    (R/'shock_bin_sweep_v1_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(agg.sort_values(['dataset','rmse_pooled']).head(40).to_string(index=False),flush=True)


if __name__=='__main__': run()

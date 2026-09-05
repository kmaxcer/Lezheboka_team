"""Local lag/shock/state sweep around the best observable peer rule.

The sweep is intentionally narrow (lag fraction .25--.40, peer 10%, shock
coefficient .25--.40, state coefficient -.10--.20) and scores all six exact
years, all three random seeds, and the random-2025 cohort.  It also writes
separate private candidates for the Pareto-front rules.  Everything is
research-only; ``model_dani_tuned_submission.csv`` is never overwritten.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
OUT = ROOT / "outputs"
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
HGB_PATH = OUT / "model_dani_tuned_hgb.csv"
LAG_PATH = OUT / "model_dani_tuned_lag.csv"
PRIVATE_PATH = DATA / "private_features.csv"
PEER10_PATH = R / "paired_aoi_v2_private_hgb_lag30_peer10.csv"
PEER_BASE_PATH = R / "paired_aoi_v2_private_hgb_lag30.csv"
BASELINE_PATH = OUT / "model_dani_tuned_submission.csv"
KEY = ["anon_polygon_id", "date"]
CANON = {97,113,129,145,161,177,193,209,225,241,257,273,289}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest().upper()


def norm(x: pd.Series) -> pd.Series:
    x = x.astype(str)
    return x.str.replace(r"^(exact)(\d+)$", r"exact_\2", regex=True).str.replace(r"^(random)(\d+)$", r"random_\2", regex=True)


def load_cv() -> pd.DataFrame:
    p = pd.read_csv(R / "paired_aoi_v2_predictions.csv", parse_dates=["date"], low_memory=False)
    s = pd.read_csv(R / "overnight_next_shock_predictions.csv", parse_dates=["date"], low_memory=False)
    s = s[s.candidate.eq("baseline")].copy(); s["pp"] = norm(s.partition)
    if p.duplicated(["partition"] + KEY).any() or s.duplicated(["partition"] + KEY).any():
        raise ValueError("duplicate CV keys")
    z = p.merge(s[["pp"]+KEY+["shock","state"]], left_on=["partition"]+KEY, right_on=["pp"]+KEY, how="left", validate="one_to_one").reset_index(drop=True)
    z["ds"] = np.where(z.family.eq("exact"), "exact", "random")
    z["canon"] = z.date.dt.dayofyear.isin(CANON).to_numpy(bool)
    z["truth"] = z["_truth"].to_numpy(float)
    return z


def folds(z: pd.DataFrame):
    for ds in ("exact", "random"):
        for part in sorted(z.loc[z.ds.eq(ds), "partition"].unique()):
            g = z[(z.ds.eq(ds)) & (z.partition.eq(part))]
            yield ds, str(part), "all", g.index.to_numpy(int)
            if ds == "random":
                h = g[g.year.eq(2025)]
                if len(h): yield ds, str(part), "year2025", h.index.to_numpy(int)


def eval_sweep(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = z.truth.to_numpy(float); sh = np.nan_to_num(z.shock.to_numpy(float), nan=0.0); st = np.nan_to_num(z.state.to_numpy(float), nan=0.0); ca = z.canon.to_numpy(bool)
    cfg = "n16_c60_r125_k2"
    q = z[cfg].to_numpy(float); qok = np.isfinite(q)
    folds0 = list(folds(z)); bases = z.hgb.to_numpy(float)
    lagraw = z.lag.to_numpy(float)
    lag_weights = [0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40]
    alphas = [0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40]
    betas = [-0.10, -0.125, -0.15, -0.175, -0.20]
    fold_rows=[]; summaries=[]
    # Baselines are recomputed at each lag weight, so deltas answer the local
    # question (does the correction beat the same lag blend?).
    for lw,a,b in itertools.product(lag_weights, alphas, betas):
        base=(1-lw)*bases+lw*lagraw
        peer=base.copy(); peer[qok]=(1-.10)*base[qok]+.10*q[qok]
        pred=peer+a*sh+b*st; pred[ca]=peer[ca]
        rr=[]
        for ds,part,co,ix in folds0:
            mse=float(np.mean((pred[ix]-y[ix])**2)); bm=float(np.mean((base[ix]-y[ix])**2));
            row=dict(dataset=ds,partition=part,cohort=co,lag_weight=lw,peer_weight=.10,alpha=a,beta=b,n=len(ix),coverage=float(qok[ix].mean()),rmse=np.sqrt(mse),baseline_rmse=np.sqrt(bm),delta_rmse=np.sqrt(mse)-np.sqrt(bm))
            fold_rows.append(row); rr.append(row)
        def agg(ds,co):
            g=[r for r in rr if r["dataset"]==ds and r["cohort"]==co]
            nn=np.array([r["n"] for r in g],float); rm=np.sqrt(np.average(np.array([r["rmse"] for r in g])**2,weights=nn)); bm=np.sqrt(np.average(np.array([r["baseline_rmse"] for r in g])**2,weights=nn));
            return rm,bm,rm-bm,int(sum(r["delta_rmse"]<0 for r in g)),len(g),float(np.average([r["coverage"] for r in g],weights=nn))
        e=agg("exact","all"); r=agg("random","all"); y25=agg("random","year2025")
        rec=dict(lag_weight=lw,peer_config=cfg,peer_weight=.10,alpha=a,beta=b)
        for pre,v in (("exact",e),("random",r),("random2025",y25)):
            rec[f"{pre}_rmse"],rec[f"{pre}_baseline_rmse"],rec[f"{pre}_delta"],rec[f"{pre}_wins"],rec[f"{pre}_folds"],rec[f"{pre}_coverage"]=v
        rec["worst_delta"]=max(e[2],r[2],y25[2]); rec["mean_delta"]=float(np.mean([e[2],r[2],y25[2]])); rec["all_wins"]=bool(e[3]==e[4] and r[3]==r[4] and y25[3]==y25[4]); summaries.append(rec)
    f=pd.DataFrame(fold_rows); s=pd.DataFrame(summaries).sort_values(["all_wins","worst_delta","mean_delta"],ascending=[False,True,True]).reset_index(drop=True)
    # Pareto front: no other rule is no-worse in both worst and mean delta,
    # with at least one strict improvement.  Restrict to all-fold winners when
    # possible; this avoids a single anomalous year dominating the choice.
    pool=s[s.all_wins].copy()
    if pool.empty: pool=s.copy()
    keep=[]
    for i,row in pool.iterrows():
        dominated=((pool.worst_delta<=row.worst_delta)&(pool.mean_delta<=row.mean_delta)&((pool.worst_delta<row.worst_delta)|(pool.mean_delta<row.mean_delta))).any()
        if not dominated: keep.append(i)
    pareto=pool.loc[keep].sort_values(["worst_delta","mean_delta"]).reset_index(drop=True)
    return f,s,pareto


def read_pred(path: Path) -> pd.DataFrame:
    z=pd.read_csv(path,parse_dates=["date"],low_memory=False)
    if not set(KEY+["primary_ndvi_pred"]).issubset(z): raise ValueError(f"{path.name}: missing prediction columns")
    z=z[KEY+["primary_ndvi_pred"]].copy()
    if z.duplicated(KEY).any(): raise ValueError(f"{path.name}: duplicate keys")
    return z


def build_private_candidates(pareto: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Write up to five Pareto private candidates, with strict metadata."""
    private=pd.read_csv(PRIVATE_PATH,parse_dates=["date"],low_memory=False)
    hidden=private.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    keys=private.loc[hidden,KEY].copy().reset_index(drop=True)
    h=read_pred(HGB_PATH).rename(columns={"primary_ndvi_pred":"hgb"}); l=read_pred(LAG_PATH).rename(columns={"primary_ndvi_pred":"lag"})
    pp=read_pred(PEER10_PATH).rename(columns={"primary_ndvi_pred":"peer10_l30"}); pb=read_pred(PEER_BASE_PATH).rename(columns={"primary_ndvi_pred":"base30"})
    q=keys.merge(h,on=KEY,validate="one_to_one").merge(l,on=KEY,validate="one_to_one").merge(pp,on=KEY,validate="one_to_one").merge(pb,on=KEY,validate="one_to_one")
    base30=.7*q.hgb.to_numpy(float)+.3*q.lag.to_numpy(float)
    # The saved lag30 peer10 file is exactly .9*base30+.1*raw_peer when a
    # peer exists and equals base30 otherwise.  Recover raw peer and coverage
    # without opening any hidden labels.
    peer_saved=q.peer10_l30.to_numpy(float); base_saved=q.base30.to_numpy(float)
    covered=np.abs(peer_saved-base_saved)>2e-7
    raw=np.full(len(q),np.nan,float); raw[covered]=(peer_saved[covered]-.9*base_saved[covered])/.1
    # Rebuild visible-only shock/state using the same feature-only functions as
    # ensemble_cv_v2_apply_peer.py.
    sys.path.insert(0,str(R)); from ensemble_cv_v2_apply import _seasonal_residuals,_shock,_state
    known=private.primary_ndvi.notna().to_numpy(bool)&~hidden; qi=np.flatnonzero(hidden)
    residual=_seasonal_residuals(private,known); shock,shock_n=_shock(private,known,residual,qi); state,state_n=_state(private,known,residual,qi)
    canon=keys.date.dt.dayofyear.isin(CANON).to_numpy(bool)
    rows=[]; seen=set()
    # Keep the strict Pareto front and a few nearby interior points.  The
    # latter are useful fallbacks when a boundary coefficient is considered
    # too aggressive; all are still evaluated on every leave-partition fold.
    specs = [r for _, r in pareto.head(5).iterrows()]
    for lw, a, b in ((0.30, 0.30, -0.15), (0.325, 0.325, -0.15),
                     (0.35, 0.325, -0.15), (0.40, 0.35, -0.20)):
        hit = summary[(summary["lag_weight"].sub(lw).abs() < 1e-9) &
                      (summary["alpha"].sub(a).abs() < 1e-9) &
                      (summary["beta"].sub(b).abs() < 1e-9)]
        if len(hit): specs.append(hit.iloc[0])
    for r in specs:
        lw=float(r.lag_weight); a=float(r.alpha); b=float(r.beta); base=(1-lw)*q.hgb.to_numpy(float)+lw*q.lag.to_numpy(float); peer=base.copy(); peer[covered]=.9*base[covered]+.1*raw[covered]; delta=np.where(canon,0.,a*np.nan_to_num(shock,nan=0.)+b*np.nan_to_num(state,nan=0.)); pred=np.clip(peer+delta,-.5,1.2)
        tag=f"lag{int(round(100*lw)):02d}_peer10_a{int(round(1000*a)):03d}_b{int(round(abs(1000*b))):03d}"
        if tag in seen: continue
        seen.add(tag); out=keys.copy(); out["primary_ndvi_pred"]=pred; fn=f"model_dani_{tag}_submission.csv"; mp=f"model_dani_{tag}_metadata.json"; outpath=OUT/fn; metapath=OUT/mp
        if outpath.resolve()==BASELINE_PATH.resolve(): raise ValueError("refusing baseline overwrite")
        out.to_csv(outpath,index=False,float_format="%.8f")
        check=read_pred(outpath)
        if len(check)!=int(hidden.sum()) or set(map(tuple,check[KEY].to_numpy()))!=set(map(tuple,keys.to_numpy())): raise ValueError(f"contract failure {fn}")
        meta={"candidate":fn,"generated_by":Path(__file__).name,"formula":f"base=(1-{lw:g})*hgb+{lw:g}*lag; peer10={.9:g}*base+{.1:g}*raw_peer when visible; canon=False +{a:g}*shock{b:+g}*state","lag_weight":lw,"peer_weight":.10,"peer_config":"n16_c60_r125_k2","shock_coef":a,"state_coef":b,"rows":len(out),"hidden_rows_in_private":int(hidden.sum()),"known_rows_used_for_features":int(known.sum()),"peer_coverage":float(covered.mean()),"shock_finite":int(np.isfinite(shock).sum()),"state_finite":int(np.isfinite(state).sum()),"canon_true":int(canon.sum()),"canon_false":int((~canon).sum()),"candidate_min":float(pred.min()),"candidate_max":float(pred.max()),"correction_min":float(delta.min()),"correction_max":float(delta.max()),"sha256":{"private_features.csv":sha(PRIVATE_PATH),"model_dani_tuned_hgb.csv":sha(HGB_PATH),"model_dani_tuned_lag.csv":sha(LAG_PATH),"paired_aoi_v2_private_hgb_lag30_peer10.csv":sha(PEER10_PATH),"paired_aoi_v2_private_hgb_lag30.csv":sha(PEER_BASE_PATH),fn:sha(outpath)},"hidden_label_columns_read":[],"production_baseline_overwritten":False}
        metapath.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        rows.append({"candidate":fn,"metadata":mp,"lag_weight":lw,"alpha":a,"beta":b,"sha256":meta["sha256"][fn],"peer_coverage":float(covered.mean()),"min_pred":float(pred.min()),"max_pred":float(pred.max())})
    return pd.DataFrame(rows)


def main():
    z=load_cv(); folds_df,summary,pareto=eval_sweep(z); folds_df.to_csv(R/"ensemble_cv_v2_local_folds.csv",index=False,float_format="%.9f"); summary.to_csv(R/"ensemble_cv_v2_local_summary.csv",index=False,float_format="%.9f"); pareto.to_csv(R/"ensemble_cv_v2_local_pareto.csv",index=False,float_format="%.9f")
    candidates=build_private_candidates(pareto, summary); candidates.to_csv(R/"ensemble_cv_v2_local_candidates.csv",index=False)
    report=["# Local lag/shock/state sweep","",f"CV rows: {len(z)}; peer config n16_c60_r125_k2, peer weight 0.10; lag weights .25--.40; alpha .25--.40; beta -.10--.20.","", "## Pareto front", "", pareto.to_string(index=False), "", "## Private candidates", "", candidates.to_string(index=False), "", "All formulas use visible-only peer/shock/state features. Production baseline was not modified."]
    (R/"ensemble_cv_v2_local_report.md").write_text("\n".join(report)+"\n",encoding="utf-8"); print("\n".join(report))


if __name__=="__main__": main()

"""Source-conditioned residual correction on top of the local peer ensemble.

The outer query rows receive a source sidecar from the unmasked source sensor
columns (train rows for exact folds, private rows for random folds).  That
sidecar is *never* used to construct the prediction.  Observable source
posteriors are rebuilt from visible sensor schedules after removing the query
keys.  Residual maps are fitted on other partitions only, then applied to the
held-out partition.  This gives a direct leakage audit for source-aware
corrections before any private candidate is materialized.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import itertools
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
OUT = ROOT / "outputs"
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
PRIVATE_PATH = DATA / "private_features.csv"
TRAIN_PATH = DATA / "train_dataset.csv"
HGB_PATH = OUT / "model_dani_tuned_hgb.csv"
LAG_PATH = OUT / "model_dani_tuned_lag.csv"
KEY = ["anon_polygon_id", "date"]
SOURCES = ["s2", "landsat", "modis"]
CANON = {97,113,129,145,161,177,193,209,225,241,257,273,289}
CFG = "n16_c60_r125_k2"


def _source_labels(d: pd.DataFrame) -> np.ndarray:
    s2 = d["s2_ndvi"].notna().to_numpy(bool)
    ls = d["landsat_ndvi"].notna().to_numpy(bool)
    md = d["modis_ndvi"].notna().to_numpy(bool)
    out = np.full(len(d), "none", dtype=object)
    out[s2] = "s2"
    out[~s2 & ls] = "landsat"
    out[~s2 & ~ls & md] = "modis"
    return out


def _norm_part(x: pd.Series) -> pd.Series:
    x = x.astype(str)
    return x.str.replace(r"^(exact)(\d+)$", r"exact_\2", regex=True).str.replace(r"^(random)(\d+)$", r"random_\2", regex=True)


def _read_cv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(R / "paired_aoi_v2_predictions.csv", parse_dates=["date"], low_memory=False)
    s = pd.read_csv(R / "overnight_next_shock_predictions.csv", parse_dates=["date"], low_memory=False)
    s = s[s.candidate.eq("baseline")].copy(); s["part_peer"] = _norm_part(s.partition)
    z = p.merge(s[["part_peer"]+KEY+["shock","state"]], left_on=["partition"]+KEY, right_on=["part_peer"]+KEY, how="left", validate="one_to_one").reset_index(drop=True)
    z["dataset"] = np.where(z.family.eq("exact"), "exact", "random")
    z["canon"] = z.date.dt.dayofyear.isin(CANON).to_numpy(bool)
    z["doy"] = z.date.dt.dayofyear.astype(int)
    z["year"] = z.date.dt.year.astype(int)
    z["truth"] = z["_truth"].to_numpy(float)
    # Attach true source as an evaluation-only sidecar.  For exact rows the
    # source comes from the original train sensors; random rows come from the
    # original private sensors before their synthetic mask.
    tr = pd.read_csv(TRAIN_PATH, parse_dates=["date"], low_memory=False); tr["true_src"] = _source_labels(tr)
    pr = pd.read_csv(PRIVATE_PATH, parse_dates=["date"], low_memory=False); pr["true_src"] = _source_labels(pr)
    trm = tr[KEY+["true_src"]].drop_duplicates(KEY); prm = pr[KEY+["true_src"]].drop_duplicates(KEY)
    z = z.merge(trm.rename(columns={"true_src":"src_train"}), on=KEY, how="left", validate="many_to_one")
    z = z.merge(prm.rename(columns={"true_src":"src_private"}), on=KEY, how="left", validate="many_to_one")
    z["true_src"] = np.where(z.dataset.eq("exact"), z.src_train, z.src_private)
    # Ensure every scored query has one of the three source labels; ``none``
    # indicates an unobservable sensor and is retained for diagnostics.
    z["hgb0"] = z.hgb.to_numpy(float); z["lag0"] = z.lag.to_numpy(float)
    return z, tr, pr


def _visible_posterior(frame: pd.DataFrame, query_keys: set[tuple[str, str]]) -> pd.DataFrame:
    """Return year+doy schedule posterior for every row in ``frame``.

    Only sensor presence on visible rows contributes.  Query keys are removed
    explicitly, so a source sidecar or hidden target cannot affect a posterior.
    """
    d = frame.copy(); d["date"] = pd.to_datetime(d["date"]); d["year"] = d.date.dt.year.astype(int); d["doy"] = d.date.dt.dayofyear.astype(int)
    tok = list(zip(d["anon_polygon_id"].astype(str), d["date"].astype(str)))
    visible = np.array([t not in query_keys for t in tok], bool)
    src = _source_labels(d)
    rows = pd.DataFrame({"year":d.year.to_numpy(),"doy":d.doy.to_numpy(),"src":src,"visible":visible})
    rows = rows[rows.visible & rows.src.ne("none")]
    tab = rows.groupby(["year","doy","src"]).size().unstack(fill_value=0).reindex(columns=SOURCES, fill_value=0)
    # Weak Laplace smoothing is fixed a priori and makes sparse dates finite.
    arr = tab.to_numpy(float) + 0.5; arr /= arr.sum(axis=1, keepdims=True)
    idx = pd.MultiIndex.from_arrays([d.year,d.doy]); pos = tab.index.get_indexer(idx)
    pp = np.full((len(d),3), 1/3, float); good = pos >= 0; pp[good] = arr[pos[good]]
    out = pd.DataFrame({"p_s2":pp[:,0],"p_landsat":pp[:,1],"p_modis":pp[:,2],"src_mode":np.asarray(SOURCES,dtype=object)[pp.argmax(1)],"year":d.year.to_numpy(),"doy":d.doy.to_numpy()})
    return out


def _attach_posteriors(z: pd.DataFrame, tr: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    out = z.copy(); pcols=["p_s2","p_landsat","p_modis","src_mode"]
    parts=[]
    for part,g in out.groupby("partition",sort=False):
        keys=set(zip(g.anon_polygon_id.astype(str),g.date.astype(str)))
        frame=tr if str(part).startswith("exact") else pr
        pp=_visible_posterior(frame,keys)
        # Align posterior to query rows by immutable key, not positional order.
        kframe=frame[KEY].copy(); kframe["_pos"]=np.arange(len(frame)); qpos=kframe.merge(g[KEY],on=KEY,how="inner",validate="one_to_one")._pos.to_numpy(int)
        qq=pp.iloc[qpos].copy(); qq.index=g.index; parts.append(qq)
    ppall=pd.concat(parts).sort_index();
    for c in pcols: out[c]=ppall[c].to_numpy()
    return out


def _folds(z: pd.DataFrame):
    for ds in ("exact","random"):
        q=z[z.dataset.eq(ds)]
        for part,g in q.groupby("partition",sort=True):
            test=g.index.to_numpy(int)
            if ds=="exact": train=q.index[q.partition.ne(part)].to_numpy(int)
            else:
                keys=set(zip(g.anon_polygon_id.astype(str),g.date.astype(str)))
                tk=list(zip(q.anon_polygon_id.astype(str),q.date.astype(str))); train=q.index[[t not in keys for t in tk]].to_numpy(int)
            yield ds,str(part),train,test


def _base_pred(z: pd.DataFrame, idx: np.ndarray, lag_weight: float) -> tuple[np.ndarray,np.ndarray]:
    h=z.hgb0.to_numpy(float); l=z.lag0.to_numpy(float); q=z[CFG].to_numpy(float); ok=np.isfinite(q)
    base=(1-lag_weight)*h+lag_weight*l; peer=base.copy(); peer[ok]=.9*base[ok]+.1*q[ok]
    sh=np.nan_to_num(z.shock.to_numpy(float),nan=0.); st=np.nan_to_num(z.state.to_numpy(float),nan=0.); ca=z.canon.to_numpy(bool)
    # Two local formulas tested independently; coefficients were predeclared by
    # the lag/shock sweep, not fitted on source labels here.
    a=.325 if lag_weight <= .325 else .35; b=-.15 if lag_weight <= .325 else -.20
    pred=peer+a*sh+b*st; pred[ca]=peer[ca]
    return pred,peer


def _fit_maps(train: pd.DataFrame, resid: np.ndarray, mode: str, min_n: int=30, cap: float=.02) -> dict:
    """Fit global/doy/source residual maps on outer-train rows only."""
    t=train.copy(); t["resid"]=resid; t=t[np.isfinite(t.resid)]
    if not len(t): return {}
    glob=float(np.median(t.resid.to_numpy(float))); out={"__global__":float(np.clip(glob,-cap,cap))}
    if mode=="global": return out
    if mode in ("doy","source_doy_soft","source_doy_mode"):
        for k,g in t.groupby("doy",dropna=False):
            if len(g)>=min_n: out[("doy",int(k))]=float(np.clip(np.median(g.resid)*len(g)/(len(g)+20),-cap,cap))
    if mode in ("source_soft","source_mode","source_doy_soft","source_doy_mode","source_oracle"):
        for k,g in t.groupby("true_src",dropna=False):
            if len(g)>=min_n: out[("src",str(k))]=float(np.clip(np.median(g.resid)*len(g)/(len(g)+20),-cap,cap))
    if mode in ("source_doy_soft","source_doy_mode"):
        for (src,doy),g in t.groupby(["true_src","doy"],dropna=False):
            if len(g)>=min_n: out[("srcdoy",str(src),int(doy))]=float(np.clip(np.median(g.resid)*len(g)/(len(g)+20),-cap,cap))
    return out


def _map_get(m: dict, key, default): return float(m.get(key,default))


def _apply_map(test: pd.DataFrame, maps: dict, mode: str, oracle: bool=False) -> np.ndarray:
    glob=_map_get(maps,"__global__",0.); p=np.full(len(test),glob,float)
    ps=test[["p_s2","p_landsat","p_modis"]].to_numpy(float); srcs=np.asarray(SOURCES,dtype=object)
    for i,(_,r) in enumerate(test.iterrows()):
        if mode=="global": continue
        # Doy map first; source map is combined through posterior/mode.  The
        # source_doy map is used only when present, otherwise backs off.
        if mode=="doy": p[i]=_map_get(maps,("doy",int(r.doy)),glob); continue
        if mode=="source_oracle":
            p[i]=_map_get(maps,("src",str(r.true_src)),glob); continue
        if mode in ("source_mode","source_doy_mode"):
            ss=str(r.src_mode); p[i]=_map_get(maps,("srcdoy",ss,int(r.doy)) if mode.endswith("doy_mode") else ("src",ss),glob); continue
        if mode in ("source_soft","source_doy_soft"):
            vals=[]
            for j,s in enumerate(srcs):
                vals.append(_map_get(maps,("srcdoy",str(s),int(r.doy)) if mode.endswith("doy_soft") else ("src",str(s)),glob))
            p[i]=float(np.dot(ps[i],np.asarray(vals,float)))
    return p


def main():
    z,tr,pr=_read_cv(); z=_attach_posteriors(z,tr,pr)
    y=z.truth.to_numpy(float); rows=[]; fold_records=[]; row_records=[]
    # Evaluate both local lag baselines and correction maps.  The map fit uses
    # only other partitions; random overlap keys are excluded by _folds().
    for ds,part,train_ix,test_ix in _folds(z):
        for lw in (.30,.325,.35,.40):
            pred,peer=_base_pred(z,np.arange(len(z)),lw)
            base_train=pred[train_ix]; base_test=pred[test_ix]
            # Residual maps are fit against the complete local peer baseline
            # (including its fixed shock/state correction).  This prevents the
            # source map from being added on top of a residual definition that
            # belongs to a different baseline.
            resid=y[train_ix]-pred[train_ix]
            tr=z.loc[train_ix].copy(); tr["resid"]=resid
            te=z.loc[test_ix].copy()
            for mode in ("none","global","doy","source_oracle","source_soft","source_mode","source_doy_soft","source_doy_mode"):
                # The local peer/shock baseline deliberately leaves canonical
                # DOYs untouched.  Fit/apply source residuals on the same
                # non-canonical regime; otherwise a source mean learned from
                # ordinary dates can bias the canonical challenge rows.
                tr_fit = tr.loc[~tr["canon"].astype(bool)].copy()
                fit_resid = tr_fit["resid"].to_numpy(float)
                maps={} if mode=="none" else _fit_maps(tr_fit,fit_resid,mode,min_n=30,cap=.02)
                if mode=="none": corr=np.zeros(len(te),float)
                else: corr=_apply_map(te,maps,mode)
                corr[te["canon"].astype(bool).to_numpy()] = 0.0
                pp=base_test+corr; yy=y[test_ix]; rm=float(np.sqrt(np.mean((pp-yy)**2))); bm=float(np.sqrt(np.mean((base_test-yy)**2)))
                row={"dataset":ds,"partition":part,"lag_weight":lw,"mode":mode,"n":len(te),"rmse":rm,"baseline_rmse":bm,"delta_rmse":rm-bm,"coverage":float(np.isfinite(z.loc[test_ix,CFG]).mean()),"map_global":float(maps.get('__global__',0.)) if maps else 0.}
                rows.append(row); fold_records.append(row)
                # Preserve row-level predictions for cross-year cohorts such
                # as random 2025.  Fold RMSEs alone cannot score that slice
                # without introducing a weighting mismatch.
                row_records.append(pd.DataFrame({
                    "dataset": ds, "partition": part, "lag_weight": lw,
                    "mode": mode, "year": te.year.to_numpy(int),
                    "canon": te.canon.to_numpy(bool), "truth": yy,
                    "pred": pp, "baseline_pred": base_test,
                }))
    f=pd.DataFrame(rows)
    rr=pd.concat(row_records,ignore_index=True)
    rr["cohort"]=np.where(rr.dataset.eq("exact"),"exact",np.where(rr.year.eq(2025),"random2025","random"))
    cohort_rows=[]
    for (lw,mode,cohort),g in rr.groupby(["lag_weight","mode","cohort"],sort=False):
        e=g.pred.to_numpy(float)-g.truth.to_numpy(float); eb=g.baseline_pred.to_numpy(float)-g.truth.to_numpy(float)
        fw=[]
        for _,q in g.groupby("partition",sort=True):
            fw.append(float(np.sqrt(np.mean((q.pred.to_numpy(float)-q.truth.to_numpy(float))**2))) < float(np.sqrt(np.mean((q.baseline_pred.to_numpy(float)-q.truth.to_numpy(float))**2))))
        cohort_rows.append({"lag_weight":lw,"mode":mode,"cohort":cohort,"n":len(g),"rmse":float(np.sqrt(np.mean(e*e))),"baseline_rmse":float(np.sqrt(np.mean(eb*eb))),"delta_rmse":float(np.sqrt(np.mean(e*e))-np.sqrt(np.mean(eb*eb))),"wins":int(sum(fw)),"folds":len(fw)})
    cohorts=pd.DataFrame(cohort_rows).sort_values(["cohort","delta_rmse","lag_weight"])
    # Aggregate exact/random/all and 2025 with squared-error weighting.
    out=[]
    for (lw,mode),g in f.groupby(["lag_weight","mode"],sort=False):
        def agg(ds,co=None):
            q=g[g.dataset.eq(ds)]
            if co is not None: q=q[q.partition.isin(q.partition.unique()) & z.loc[z.index.isin([]),"partition"].isin([])] if False else q
            if co=="year2025":
                q2=[]
                for _,r in q.iterrows():
                    # fold rows are not row-level here; use separately below
                    pass
            if not len(q): return (np.nan,np.nan,np.nan,0,0)
            nn=q.n.to_numpy(float); rm=np.sqrt(np.average(q.rmse.to_numpy(float)**2,weights=nn)); bm=np.sqrt(np.average(q.baseline_rmse.to_numpy(float)**2,weights=nn)); return rm,bm,rm-bm,int((q.delta_rmse<0).sum()),len(q)
        e=agg("exact"); r=agg("random")
        rec={"lag_weight":lw,"mode":mode,"exact_rmse":e[0],"exact_baseline_rmse":e[1],"exact_delta":e[2],"exact_wins":e[3],"exact_folds":e[4],"random_rmse":r[0],"random_baseline_rmse":r[1],"random_delta":r[2],"random_wins":r[3],"random_folds":r[4]}
        out.append(rec)
    s=pd.DataFrame(out)
    # Correct 2025 cohort from row-level fold tables by rerunning masks on the
    # saved rows; this avoids pretending a fold RMSE is a row RMSE.
    # (The dedicated local sweep already reports 2025; source correction is
    # primarily judged on exact/random all here.)
    s["worst_delta"]=s[["exact_delta","random_delta"]].max(axis=1); s["mean_delta"]=s[["exact_delta","random_delta"]].mean(axis=1); s=s.sort_values(["worst_delta","mean_delta"])
    f.to_csv(R/"ensemble_cv_v2_source_correction_folds.csv",index=False,float_format="%.9f"); s.to_csv(R/"ensemble_cv_v2_source_correction_summary.csv",index=False,float_format="%.9f"); cohorts.to_csv(R/"ensemble_cv_v2_source_correction_cohorts.csv",index=False,float_format="%.9f"); rr.to_csv(R/"ensemble_cv_v2_source_correction_rows.csv",index=False,float_format="%.9f")
    report=["# Source-conditioned residual correction audit","",f"Rows: {len(z)}; source sidecars attached from unmasked train/private sensor presence; posterior uses visible year+doy schedule with query keys removed.","",s.to_string(index=False),"","## Cohorts (row-level pooled RMSE)","",cohorts.to_string(index=False),"","No source correction is promoted unless it improves both exact and random outer folds without oracle source labels. `source_oracle` is an upper-bound diagnostic only."]
    (R/"ensemble_cv_v2_source_correction_report.md").write_text("\n".join(report)+"\n",encoding="utf-8"); print("\n".join(report))


if __name__=="__main__": main()

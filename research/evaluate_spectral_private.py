"""Leakage-safe private holdout screen for temporal spectral features.

This intentionally trains only one compact HGB family on top of the archive
features plus nearest EVI/NDWI/ratio anchors.  Existing v3/ext40 predictions
are joined from their saved holdout table, so the screen measures whether the
new signal adds a useful residual correction without refitting every baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
ARCH = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
R = ROOT / "research"
sys.path.insert(0, str(ARCH)); sys.path.insert(0, str(R))
from agropulse.pipeline import build_features  # noqa: E402
from feature_hgb_v2 import _clear  # noqa: E402
from spectral_features_v1 import spectral_features, TRANSFORMS, LOCAL_SUFFIXES  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"


def make_holdout(pr: pd.DataFrame, seed: int = 70404) -> np.ndarray:
    known = pr[TARGET].notna().to_numpy(bool) & ~pr[GAP].fillna(False).to_numpy(bool)
    out = np.zeros(len(pr), bool); rng = np.random.default_rng(seed)
    yy = pd.to_datetime(pr[DATE]).dt.year
    for _, ix0 in pr.loc[known].groupby([ID, yy], sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int); n = max(1, int(round(.15 * len(ix))))
        out[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
    return out


def matrix(d: pd.DataFrame, observed: pd.Series, mask: np.ndarray, add_base: bool = True) -> pd.DataFrame:
    m = np.asarray(mask, bool)
    fr = _clear(d, m)
    sp = spectral_features(fr, observed, m)
    if add_base:
        bx = build_features(fr, observed, pd.Series(m, index=fr.index))
        x = pd.concat([bx.reset_index(drop=True), sp.reset_index(drop=True)], axis=1)
    else:
        x = sp
    return x.replace([np.inf, -np.inf], np.nan)


def fit_screen(ref: pd.DataFrame, gaps: np.ndarray, n_masks: int = 2, add_base: bool = True):
    d = ref.copy().reset_index(drop=True); d[DATE] = pd.to_datetime(d[DATE])
    d["year"] = d["year"].fillna(d[DATE].dt.year).astype(int); d["doy"] = d["doy"].fillna(d[DATE].dt.dayofyear).astype(int)
    d["_truth"] = pd.to_numeric(d[TARGET], errors="coerce")
    gaps = np.asarray(gaps, bool); known = d[TARGET].notna().to_numpy(bool) & ~gaps
    tab = pd.DataFrame({"id": d[ID].astype(str), "year": d[DATE].dt.year.to_numpy(int)})
    blocks=[]; ys=[]; t0=time.time(); feature_count=None
    for rep in range(n_masks):
        rng=np.random.default_rng(20261000+rep); pm=np.zeros(len(d),bool)
        for _,ix0 in tab.loc[known].groupby(["id","year"],sort=False).groups.items():
            ix=np.asarray(ix0,dtype=int); nn=max(1,int(round(.18*len(ix))))
            pm[rng.choice(ix,size=min(nn,len(ix)),replace=False)]=True
        comb=gaps|pm; obs=d[TARGET].where(~comb)
        print(f"spectral block {rep+1}/{n_masks}: pseudo={int(pm.sum())}",flush=True)
        x=matrix(d,obs,comb,add_base=add_base); feature_count=x.shape[1]
        blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,"_truth"].reset_index(drop=True))
    obs=d[TARGET].where(~gaps); print(f"spectral query={int(gaps.sum())}",flush=True)
    qx=matrix(d,obs,gaps,add_base=add_base).loc[gaps].reset_index(drop=True)
    X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float)
    model=HistGradientBoostingRegressor(loss="squared_error",random_state=42,learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=35,l2_regularization=10.0)
    print("spectral fit",X.shape,flush=True); model.fit(X,y)
    p=np.clip(model.predict(qx),-.2,1.1)
    return d,p,{"features":int(feature_count or 0),"train_rows":int(len(X)),"seconds":round(time.time()-t0,1)}


def metric(g: pd.DataFrame, c: str):
    y=g.truth.to_numpy(float); p=g[c].to_numpy(float); ok=np.isfinite(y)&np.isfinite(p)
    return int(ok.sum()), float(np.sqrt(np.mean((p[ok]-y[ok])**2))), float(np.mean(np.abs(p[ok]-y[ok])))


def main():
    t0=time.time(); tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False)
    tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool); hold=make_holdout(pr); hidden=pr[GAP].to_numpy(bool); gaps_pr=hold|hidden
    tr2=tr.copy(); p2=pr.copy(); tr2["_origin"]="train"; p2["_origin"]="private"
    ref=pd.concat([tr2,p2],ignore_index=True,sort=False); ref[DATE]=pd.to_datetime(ref[DATE]); ref["year"]=ref["year"].fillna(ref[DATE].dt.year).astype(int); ref["doy"]=ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    ref["_truth"]=pd.to_numeric(ref[TARGET],errors="coerce")
    hk=set(map(tuple,pr.loc[gaps_pr,[ID,DATE]].to_numpy())); gaps_ref=np.array([tuple(x) in hk for x in ref[[ID,DATE]].to_numpy()],bool)
    ref.loc[gaps_ref,TARGET]=np.nan
    # Raw dynamic fields on gaps are cleared by _clear.  This sidecar is kept
    # only for scoring and never passed to a feature builder.
    print("reference",len(ref),"gaps",int(gaps_ref.sum()),"holdout",int(hold.sum()),flush=True)
    dfit, ps, info = fit_screen(ref,gaps_ref,n_masks=2,add_base=True)
    # ``ps`` covers organiser gaps and the added holdout.  Map by key before
    # scoring, retaining only the additional holdout rows.
    allq = dfit.loc[gaps_ref, [ID, DATE]].copy().reset_index(drop=True); allq["spectral"] = ps
    qkeys=pr.loc[hold,[ID,DATE]].copy().reset_index(drop=True)
    qkeys=qkeys.merge(allq,on=[ID,DATE],how="left",validate="one_to_one")
    qkeys["truth"]=pr.loc[hold,TARGET].to_numpy(float)
    train_ids=set(tr[ID].astype(str)); qkeys["cohort"]=np.where(qkeys[ID].astype(str).isin(train_ids),"shared","new"); qkeys["year"]=pd.to_datetime(qkeys[DATE]).dt.year.astype(int)
    old=pd.read_csv(R/"private_cohort_blend_holdout_predictions.csv",parse_dates=[DATE],low_memory=False)
    v3=pd.read_csv(R/"v3_private_holdout_predictions.csv",parse_dates=[DATE],low_memory=False)
    # v3 table contains one row per holdout key; use its prediction and the
    # already evaluated ext40 component for apples-to-apples residual blends.
    qkeys=qkeys.merge(old[[ID,DATE,"ext40","joint40"]],on=[ID,DATE],how="left",validate="one_to_one")
    qkeys=qkeys.merge(v3[[ID,DATE,"v3"]],on=[ID,DATE],how="left",validate="one_to_one")
    qkeys["ext40_v3_30"]=.7*qkeys.ext40+.3*qkeys.v3
    qkeys["ext40_v3_40"]=.6*qkeys.ext40+.4*qkeys.v3
    # Grid is intentionally conservative: a standalone spectral model can be
    # noisy, so promotion requires a stable improvement over ext40_v3_30.
    for w in (0.05,.10,.15,.20,.25,.30,.40):
        qkeys[f"blend_{int(round(100*w)):02d}"]=(1-w)*qkeys.ext40_v3_30+w*qkeys.spectral
        qkeys[f"joint_blend_{int(round(100*w)):02d}"]=(1-w)*qkeys.joint40+w*qkeys.spectral
    rows=[]
    groups={"all":qkeys,"history":qkeys[qkeys.year<2025],"2025":qkeys[qkeys.year==2025],"new_history":qkeys[(qkeys.cohort=="new")&(qkeys.year<2025)],"new_2025":qkeys[(qkeys.cohort=="new")&(qkeys.year==2025)],"shared_2025":qkeys[(qkeys.cohort=="shared")&(qkeys.year==2025)]}
    cols=["spectral","ext40_v3_30","ext40_v3_40"]+[c for c in qkeys if c.startswith("blend_") or c.startswith("joint_blend_")]
    for gn,g in groups.items():
        for c in cols:
            n,rm,ma=metric(g,c); rows.append({"cohort":gn,"method":c,"n":n,"rmse":rm,"mae":ma})
    res=pd.DataFrame(rows); qkeys.to_csv(R/"spectral_private_holdout_predictions.csv",index=False); res.to_csv(R/"spectral_private_holdout_results.csv",index=False)
    meta={"holdout_seed":70404,"holdout_rows":int(hold.sum()),"actual_hidden_rows":int(hidden.sum()),"n_masks":2,**info,"seconds_total":round(time.time()-t0,1)}; (R/"spectral_private_holdout_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(res[res.cohort.isin(["all","history","2025","new_history","new_2025","shared_2025"])].sort_values(["cohort","rmse"]).to_string(index=False),flush=True); print(json.dumps(meta,indent=2),flush=True)


if __name__=="__main__": main()

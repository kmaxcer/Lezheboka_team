"""Small leakage-safe audit for feature_hgb_v3 (research only).

Runs one exact private-DOY fold and one random private-like fold.  Only the
wide model is fitted; the purpose is to reject/promote the feature family
quickly before paying the cost of a full private fit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
sys.path.insert(0, str(ROOT / "research"))
from validate import make_fold  # noqa: E402
from agropulse.pipeline import build_features  # noqa: E402
from feature_hgb_v2 import _clear  # noqa: E402
from feature_hgb_v3 import extra_features_v3  # noqa: E402

TARGET = "primary_ndvi"


def matrix(frame: pd.DataFrame, observed: pd.Series, mask: np.ndarray) -> pd.DataFrame:
    fr = _clear(frame, mask)
    bx = build_features(fr, observed, pd.Series(np.asarray(mask, bool)))
    ex = extra_features_v3(fr, observed, np.asarray(mask, bool))
    return pd.concat([bx.reset_index(drop=True), ex.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)


def fit_wide(x: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(loss="squared_error", random_state=42,
        learning_rate=.03, max_iter=350, max_leaf_nodes=63,
        min_samples_leaf=30, l2_regularization=8.0)
    m.fit(x, y.astype(float))
    return m


def run_case(name: str, d: pd.DataFrame, query: np.ndarray, truth: np.ndarray,
             pseudo_pool: np.ndarray, rng_seed: int) -> dict:
    t0 = time.time(); blocks=[]; ys=[]
    tab = pd.DataFrame({"id": d.anon_polygon_id.astype(str), "year": pd.to_datetime(d.date).dt.year})
    for rep in range(2):
        pm = np.zeros(len(d), dtype=bool)
        rng = np.random.default_rng(rng_seed + rep)
        for _, ix0 in tab.loc[pseudo_pool].groupby(["id", "year"], sort=False).groups.items():
            ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix))))
            pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
        comb = query | pm
        obs = d[TARGET].where(~comb)
        print(name, "features block", rep+1, "query", int(query.sum()), "pseudo", int(pm.sum()), flush=True)
        x=matrix(d, obs, comb)
        blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,"_truth"].reset_index(drop=True))
    obs=d[TARGET].where(~query)
    print(name, "features query", flush=True)
    qx=matrix(d,obs,query).loc[query].reset_index(drop=True)
    xall=pd.concat(blocks,ignore_index=True); yall=pd.concat(ys,ignore_index=True)
    m=fit_wide(xall,yall); p=np.clip(m.predict(qx),-.2,1.1)
    rm=float(np.sqrt(np.mean((p-truth)**2))); mae=float(np.mean(np.abs(p-truth)))
    out={"case":name,"n":int(len(truth)),"features":int(xall.shape[1]),"rmse":rm,"mae":mae,"seconds":round(time.time()-t0,1)}
    print(out,flush=True); return out


def main():
    tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=["date"],low_memory=False)
    pr=pd.read_csv(DATA/"private_features.csv",parse_dates=["date"],low_memory=False)
    tr["_truth"]=tr[TARGET].astype(float)
    # Exact 2024 private hidden DOYs, same mask protocol as the established v2 CV.
    fold,truth=make_fold(tr.copy(),pr.copy(),2024)
    qm=fold.is_synthetic_gap.fillna(False).to_numpy(bool)
    pool=fold[TARGET].notna().to_numpy(bool)&~qm
    pool &= fold.date.dt.year.ne(2024).to_numpy()
    exact=run_case("exact2024",fold,qm,truth.to_numpy(float),pool,20261001)
    # Random private-like mask over all visible rows, with the exact synthetic
    # rows excluded from both the query and the training pool.
    d=tr.copy(); d["is_synthetic_gap"]=False
    rng=np.random.default_rng(20261020); base=d[TARGET].notna().to_numpy(bool)
    q=np.zeros(len(d),bool); tab=pd.DataFrame({"id":d.anon_polygon_id.astype(str),"year":d.date.dt.year})
    for _,ix0 in tab.loc[base].groupby(["id","year"],sort=False).groups.items():
        ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.15*len(ix))))
        q[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
    d["_truth"]=d[TARGET].astype(float)
    random=run_case("random15",d,q,d.loc[q,"_truth"].to_numpy(float),base&~q,20261101)
    out=pd.DataFrame([exact,random]); out.to_csv(ROOT/"research"/"feature_hgb_v3_quick_results.csv",index=False)
    print(out.to_string(index=False),flush=True)

if __name__=="__main__": main()

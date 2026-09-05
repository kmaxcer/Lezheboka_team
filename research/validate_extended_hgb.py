"""Compact outer checks for the extended-HGB feature component.

Runs one representative exact fold, one random train mask, and a 2025
private-known mask.  Each fit is leakage-safe: query rows and pseudo-gaps are
blanked before *both* base and extra features are built.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(RESEARCH))
from build_extended_hgb_private import _clear, _fit, _matrix, _pseudo_masks  # noqa: E402

TARGET = "primary_ndvi"


def fit_eval(d: pd.DataFrame, query: np.ndarray, exclude: np.ndarray,
             seed: int, n_masks: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date)
    d["year"] = d["year"].fillna(d.date.dt.year).astype(int)
    d["doy"] = d["doy"].fillna(d.date.dt.dayofyear).astype(int)
    # make_fold already carries the unmasked labels in _truth; preserve them
    # even though TARGET is intentionally NaN on the exact query rows.
    if "_truth" not in d.columns:
        d["_truth"] = pd.to_numeric(d[TARGET], errors="coerce")
    blocks = []; ys = []
    for pm in _pseudo_masks(d, exclude, n_masks=n_masks, fraction=.18):
        comb = np.asarray(exclude, bool) | pm
        fr = _clear(d, comb); obs = fr[TARGET].where(~comb)
        x = _matrix(d, obs, comb)
        blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    vf = _clear(d, exclude); obs = vf[TARGET].where(~exclude)
    qx = _matrix(d, obs, exclude).loc[query].reset_index(drop=True)
    xa = pd.concat(blocks, ignore_index=True); ya = pd.concat(ys, ignore_index=True).astype(float)
    y = d.loc[query, "_truth"].to_numpy(float)
    rows = []; pred = []
    for kind in ("regular", "wide"):
        m = _fit(kind, xa, ya, 42)
        p = np.clip(m.predict(qx), -.2, 1.1)
        e = p-y
        rows.append({"kind": kind, "n": len(y), "rmse": float(np.sqrt(np.mean(e*e))), "mae": float(np.mean(np.abs(e))), "train_n": len(xa), "features": xa.shape[1]})
        pred.append(pd.DataFrame({"anon_polygon_id": d.loc[query,"anon_polygon_id"].to_numpy(), "date": d.loc[query,"date"].to_numpy(), "truth":y, "kind":kind, "pred":p}))
    return pd.DataFrame(rows), pd.concat(pred, ignore_index=True)


def main() -> None:
    t0 = time.time()
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    all_rows = []; all_pred = []
    # Representative exact fold (2024); the six-fold train-only run is in
    # feature_hgb_v2_results.csv and is retained for pooled comparison.
    fold, truth = make_fold(tr.copy(), pr.copy(), 2024)
    qm = fold.is_synthetic_gap.fillna(False).to_numpy(bool)
    r, p = fit_eval(fold, qm, qm, 202401, n_masks=2); r["protocol"]="exact2024"; all_rows.append(r); p["protocol"]="exact2024"; all_pred.append(p)
    # Random 15% mask over train, stratified by AOI/year.
    d = tr.copy(); d["is_synthetic_gap"] = False; d["_truth"] = d[TARGET].astype(float)
    rng = np.random.default_rng(20260905)
    hold = np.zeros(len(d), bool)
    for _, ix0 in d.groupby(["anon_polygon_id", d.date.dt.year], sort=False).groups.items():
        ix=np.asarray(ix0,int); ix=ix[d.loc[ix,TARGET].notna().to_numpy()]
        if len(ix):
            n=max(1,int(round(.15*len(ix)))); hold[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
    r,p=fit_eval(d,hold,hold,202402,n_masks=2); r["protocol"]="random"; all_rows.append(r); p["protocol"]="random"; all_pred.append(p)
    # 2025 proxy: mask 30% of visible private 2025 rows, while keeping the
    # organiser's actual hidden rows unavailable to every feature statistic.
    tr["is_synthetic_gap"] = False; pr["is_synthetic_gap"] = pr["is_synthetic_gap"].fillna(False).astype(bool)
    d=pd.concat([tr,pr],ignore_index=True,sort=False); d["date"]=pd.to_datetime(d.date); d["year"]=d.year.fillna(d.date.dt.year).astype(int); d["doy"]=d.doy.fillna(d.date.dt.dayofyear).astype(int); d["_truth"]=d[TARGET].astype(float)
    hidden=d.is_synthetic_gap.to_numpy(bool); target=(d.date.dt.year.eq(2025)&d[TARGET].notna()&~hidden).to_numpy(); hold=np.zeros(len(d),bool); rr=np.random.default_rng(202403); 
    for _,ix0 in d.loc[target].groupby("anon_polygon_id",sort=False).groups.items():
        ix=np.asarray(ix0,int); n=max(1,int(round(.30*len(ix)))); hold[rr.choice(ix,size=min(n,len(ix)),replace=False)]=True
    exclude=hidden|hold
    r,p=fit_eval(d,hold,exclude,202403,n_masks=1); r["protocol"]="proxy2025"; all_rows.append(r); p["protocol"]="proxy2025"; all_pred.append(p)
    out=pd.concat(all_rows,ignore_index=True); pred=pd.concat(all_pred,ignore_index=True); out.to_csv(RESEARCH/'extended_hgb_validation_results.csv',index=False); pred.to_csv(RESEARCH/'extended_hgb_validation_predictions.csv',index=False)
    print(out.to_string(index=False)); print("elapsed",round(time.time()-t0,1),flush=True)


if __name__ == "__main__": main()

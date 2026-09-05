"""Fit the compact spectral HGB on the full visible reference and route it.

Research output only: the established extwide40_v3_30 component is preserved,
and spectral corrections are emitted as separate candidates.  All dynamic
fields on organiser gaps are cleared before feature construction.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
OUT = ROOT / "outputs"; R = ROOT / "research"
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src")); sys.path.insert(0, str(R))
from agropulse.pipeline import build_features  # noqa: E402
from feature_hgb_v2 import _clear  # noqa: E402
from spectral_features_v1 import spectral_features  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
BASE_NAME = "model_dani_lag40_peer10_extwide40_v3_30_submission.csv"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def matrix(d: pd.DataFrame, obs: pd.Series, mask: np.ndarray) -> pd.DataFrame:
    m = np.asarray(mask, bool); fr = _clear(d, m)
    bx = build_features(fr, obs, pd.Series(m, index=fr.index))
    sp = spectral_features(fr, obs, m)
    return pd.concat([bx.reset_index(drop=True), sp.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)


def pseudo_masks(d: pd.DataFrame, hidden: np.ndarray, count: int = 2) -> list[np.ndarray]:
    known = d[TARGET].notna().to_numpy(bool) & ~hidden
    tab = pd.DataFrame({"id": d[ID].astype(str), "year": d[DATE].dt.year.to_numpy(int)})
    out=[]
    for rep in range(count):
        rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(d),bool)
        for _, ix0 in tab.loc[known].groupby(["id","year"],sort=False).groups.items():
            ix=np.asarray(ix0,dtype=int); nn=max(1,int(round(.18*len(ix))))
            pm[rng.choice(ix,size=min(nn,len(ix)),replace=False)]=True
        out.append(pm)
    return out


def write_candidate(keys: pd.DataFrame, p: np.ndarray, name: str, meta: dict) -> dict:
    p=np.clip(np.asarray(p,float),-.2,1.1); out=keys.copy(); out["primary_ndvi_pred"]=p
    if list(out.columns)!=[ID,DATE,"primary_ndvi_pred"] or out.duplicated([ID,DATE]).any() or not np.isfinite(p).all(): raise RuntimeError("contract failure")
    path=OUT/name; out.to_csv(path,index=False,float_format="%.8f")
    return {"candidate":name,"rows":int(len(out)),"min":float(p.min()),"max":float(p.max()),"mean":float(p.mean()),"sha256":sha(path),**meta}


def main():
    t0=time.time(); tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False)
    tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool); tr["_origin"]="train"; pr["_origin"]="private"
    d=pd.concat([tr,pr],ignore_index=True,sort=False); d[DATE]=pd.to_datetime(d[DATE]); d["year"]=d["year"].fillna(d[DATE].dt.year).astype(int); d["doy"]=d["doy"].fillna(d[DATE].dt.dayofyear).astype(int); d["_truth"]=pd.to_numeric(d[TARGET],errors="coerce")
    hidden=d[GAP].to_numpy(bool); qi=np.flatnonzero(hidden)
    keys=d.loc[hidden,[ID,DATE]].copy().reset_index(drop=True)
    checkpoint = R / "spectral_full_predictions_checkpoint.csv"
    sp = None; feature_count = 219
    if checkpoint.exists():
        cp=pd.read_csv(checkpoint,parse_dates=[DATE],low_memory=False)
        key0=keys.copy(); key0[ID]=key0[ID].astype(str); cp[ID]=cp[ID].astype(str)
        if len(cp)==len(key0) and cp[[ID,DATE]].reset_index(drop=True).equals(key0[[ID,DATE]].reset_index(drop=True)) and np.isfinite(cp.spectral_pred).all():
            print("using spectral checkpoint",len(cp),flush=True); sp=cp.spectral_pred.to_numpy(float); X=pd.DataFrame()
    if sp is None:
        blocks=[]; ys=[]
        # One disjoint pseudo block is sufficient for the routed candidate and
        # halves the expensive feature construction; the two-block screen above
        # established the direction of the blend.
        for no,pm in enumerate(pseudo_masks(d,hidden,1),1):
            comb=hidden|pm; obs=d[TARGET].where(~comb); print("spectral features block",no,"pseudo",int(pm.sum()),flush=True)
            x=matrix(d,obs,comb); feature_count=int(x.shape[1]); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,"_truth"].reset_index(drop=True))
        obs=d[TARGET].where(~hidden); print("spectral features query",len(qi),flush=True); qx=matrix(d,obs,hidden).loc[hidden].reset_index(drop=True)
        X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float)
        m=HistGradientBoostingRegressor(loss="squared_error",random_state=42,learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=35,l2_regularization=10.0)
        print("spectral fit",X.shape,flush=True); m.fit(X,y); sp=np.clip(m.predict(qx),-.2,1.1)
        # Crash-safe checkpoint: expensive feature construction need not be lost
        # if a later metadata/hash operation fails.
        pd.DataFrame({ID: keys[ID].astype(str), DATE: keys[DATE], "spectral_pred": sp}).to_csv(checkpoint, index=False, float_format="%.8f")
    base=pd.read_csv(OUT/BASE_NAME,parse_dates=[DATE],low_memory=False); base[DATE]=pd.to_datetime(base[DATE]); bm=keys.merge(base,on=[ID,DATE],how="left",validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
    train_ids=set(tr[ID].astype(str)); years=keys[DATE].dt.year.to_numpy(int); shared=keys[ID].astype(str).isin(train_ids).to_numpy(bool); history=(years<2025); shared25=shared&~history; new25=(~shared)&~history
    infos=[]
    # Requested route: spectral correction on all pre-2025 history and AOIs
    # with prior train history, but zero weight on genuinely new 2025 AOIs.
    for w in (.20,.30,.40):
        mask=history|shared25
        ww=np.where(mask,w,0.0); p=(1-ww)*bm+ww*sp
        name=f"model_dani_lag40_peer10_extwide40_v3_30_spectral{int(round(100*w)):02d}_routed_submission.csv"
        infos.append(write_candidate(keys,p,name,{"spectral_weight":w,"routing":"history_or_shared2025;zero_new2025","spectral_component":"base+nearest-EVI-NDWI-HGB","base_component":BASE_NAME,"base_sha256":sha(OUT/BASE_NAME),"private_sha256":sha(DATA/"private_features.csv"),"pseudo_masks":1,"features":feature_count,"history_rows":int(history.sum()),"shared2025_rows":int(shared25.sum()),"new2025_rows":int(new25.sum()),"production_baseline_overwritten":False}))
    # Also emit a conservative history-only variant; holdout showed spectral
    # gains concentrated before 2025, while shared-2025 was slightly negative.
    for w in (.30,.40):
        ww=np.where(history,w,0.0); p=(1-ww)*bm+ww*sp
        name=f"model_dani_lag40_peer10_extwide40_v3_30_spectral{int(round(100*w)):02d}_historyonly_submission.csv"
        infos.append(write_candidate(keys,p,name,{"spectral_weight":w,"routing":"history_only;zero_all2025","spectral_component":"base+nearest-EVI-NDWI-HGB","base_component":BASE_NAME,"base_sha256":sha(OUT/BASE_NAME),"private_sha256":sha(DATA/"private_features.csv"),"pseudo_masks":1,"features":feature_count,"history_rows":int(history.sum()),"shared2025_rows":int(shared25.sum()),"new2025_rows":int(new25.sum()),"production_baseline_overwritten":False}))
    # Holdout routing showed the spectral correction is beneficial before 2025
    # but mildly harmful on the noisier 2025 rows; advertise history-only as
    # the primary artifact and retain routed variants as alternatives.
    meta={"recommended":next(x["candidate"] for x in infos if x.get("routing")=="history_only;zero_all2025" and x.get("spectral_weight")==.30),"candidates":infos,"spectral_model":"HistGradientBoostingRegressor(base_features+spectral_features_v1)","seconds":round(time.time()-t0,1),"private_sha256":sha(DATA/"private_features.csv"),"production_baseline_overwritten":False}
    (OUT/"model_dani_spectral_routed_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()

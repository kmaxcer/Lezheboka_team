"""Fit feature_hgb_v3 on all visible train/private rows (research output only)."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
OUT=ROOT/"outputs"; R=ROOT/"research"
sys.path.insert(0,str(ROOT/"_archive_inspect"/"agropulse_max_score"/"src")); sys.path.insert(0,str(R))
from agropulse.pipeline import build_features
from feature_hgb_v2 import _clear
from feature_hgb_v3 import extra_features_v3

TARGET="primary_ndvi"

def sha(p):
 h=hashlib.sha256();
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def mat(d,obs,mask):
 fr=_clear(d,mask); bx=build_features(fr,obs,pd.Series(np.asarray(mask,bool)))
 ex=extra_features_v3(fr,obs,np.asarray(mask,bool))
 return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)

def main():
 t0=time.time(); tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=["date"],low_memory=False); pr=pd.read_csv(DATA/"private_features.csv",parse_dates=["date"],low_memory=False)
 tr["is_synthetic_gap"]=False; pr["is_synthetic_gap"]=pr["is_synthetic_gap"].fillna(False).astype(bool)
 tr["_origin"]="train"; pr["_origin"]="private"; d=pd.concat([tr,pr],ignore_index=True,sort=False); d["date"]=pd.to_datetime(d.date); d["year"]=d.year.fillna(d.date.dt.year).astype(int); d["doy"]=d.doy.fillna(d.date.dt.dayofyear).astype(int); d["_truth"]=pd.to_numeric(d[TARGET],errors="coerce")
 hidden=d.is_synthetic_gap.to_numpy(bool); qi=np.flatnonzero(hidden); known=d[TARGET].notna().to_numpy(bool)&~hidden
 years=d.date.dt.year.to_numpy(int); tab=pd.DataFrame({"id":d.anon_polygon_id.astype(str),"year":years})
 blocks=[]; ys=[]
 for rep in range(3):
  rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(d),bool)
  for _,ix0 in tab.loc[known].groupby(["id","year"],sort=False).groups.items():
   ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
  comb=hidden|pm; obs=d[TARGET].where(~comb); print("features block",rep+1,"pseudo",int(pm.sum()),flush=True); x=mat(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,"_truth"].reset_index(drop=True))
 obs=d[TARGET].where(~hidden); print("features query",len(qi),flush=True); qx=mat(d,obs,hidden).loc[hidden].reset_index(drop=True); xall=pd.concat(blocks,ignore_index=True); yall=pd.concat(ys,ignore_index=True).astype(float)
 m=HistGradientBoostingRegressor(loss="squared_error",random_state=42,learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=8.0); print("fit",xall.shape,flush=True); m.fit(xall,yall); p=np.clip(m.predict(qx),-.2,1.1)
 keys=d.loc[hidden,["anon_polygon_id","date"]].copy().reset_index(drop=True); out=keys.copy(); out["primary_ndvi_pred"]=p; path=OUT/"model_dani_extended_hgb_v3_wide.csv"; out.to_csv(path,index=False,float_format="%.8f")
 meta={"rows":int(len(qi)),"features":int(xall.shape[1]),"pseudo_masks":3,"model":"wide","seconds":round(time.time()-t0,1),"train_sha256":sha(DATA/"train_dataset.csv"),"private_sha256":sha(DATA/"private_features.csv"),"production_baseline_overwritten":False}
 (OUT/"model_dani_extended_hgb_v3_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()

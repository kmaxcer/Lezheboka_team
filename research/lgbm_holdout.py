"""Quick leakage-safe LightGBM screen on the private-like holdout.

Uses the same extended feature construction as the production research HGB,
but fits LightGBM variants and compares them against saved holdout blends.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
RES = ROOT / "research"
sys.path.insert(0, str(RES))
from evaluate_private_cohort_blend import make_holdout
from build_extended_hgb_private import _clear, _matrix

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"

def fit_lgb(X, y, kind):
    specs = {
        "lgbm_fast": dict(n_estimators=500, learning_rate=.025, num_leaves=31, min_child_samples=80,
                           reg_lambda=12., reg_alpha=.5, max_bin=127),
        "lgbm_wide": dict(n_estimators=650, learning_rate=.02, num_leaves=63, min_child_samples=60,
                           reg_lambda=16., reg_alpha=.5, max_bin=127),
        "lgbm_deep": dict(n_estimators=700, learning_rate=.018, num_leaves=127, min_child_samples=80,
                           reg_lambda=24., reg_alpha=1., max_bin=127),
    }[kind]
    return LGBMRegressor(objective="regression", verbosity=-1, random_state=42,
                         n_jobs=-1, subsample=.9, colsample_bytree=.9, **specs).fit(X, y)

def main():
    t0=time.time()
    tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False)
    pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False)
    tr[GAP]=False; pr[GAP]=pr[GAP].fillna(False).astype(bool)
    hold=make_holdout(pr); gaps=pr[GAP].to_numpy(bool)|hold
    truth=pr[TARGET].to_numpy(float)
    pmask=pr.copy(); pmask.loc[gaps,TARGET]=np.nan; pmask.loc[gaps,GAP]=True
    # Build the same sorted train+private reference as the validated evaluator.
    tr2=tr.copy(); p2=pmask.copy(); tr2["_origin"]="train"; p2["_origin"]="private"
    ref=pd.concat([tr2,p2],ignore_index=True,sort=False).sort_values([ID,DATE,"_origin"]).reset_index(drop=True)
    ref["year"]=ref["year"].fillna(ref[DATE].dt.year).astype(int); ref["doy"]=ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    labels=pd.concat([tr[[ID,DATE,TARGET]],pr[[ID,DATE,TARGET]]],ignore_index=True)
    ref=ref.merge(labels.rename(columns={TARGET:"_truth"}),on=[ID,DATE],how="left",validate="one_to_one")
    hk=set(map(tuple,pr.loc[hold,[ID,DATE]].to_numpy())); gm=ref[GAP].fillna(False).to_numpy(bool)
    gm=gm|np.array([tuple(x) in hk for x in ref[[ID,DATE]].to_numpy()]); ref.loc[gm,TARGET]=np.nan
    known=ref[TARGET].notna().to_numpy(bool)&~gm; years=pd.to_datetime(ref[DATE]).dt.year.to_numpy(int)
    blocks=[]; ys=[]
    for rep in range(2):
        rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(ref),bool)
        tab=pd.DataFrame({"id":ref[ID].astype(str),"year":years})
        for _,ix0 in tab.loc[known].groupby(["id","year"],sort=False).groups.items():
            ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
        comb=gm|pm; obs=ref[TARGET].where(~comb)
        print("features block",rep, int(pm.sum()), flush=True)
        x=_matrix(ref,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(ref.loc[pm,"_truth"].reset_index(drop=True))
    obs=ref[TARGET].where(~gm); print("features query",int(gm.sum()),flush=True)
    qx=_matrix(ref,obs,gm).loc[gm].reset_index(drop=True)
    X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float)
    qi_ref=ref.loc[gm,[ID,DATE]].copy(); qi_ref["pred_i"]=np.arange(len(qi_ref))
    # Map only random holdout queries (not organiser hidden rows) by key.
    qkeys=pr.loc[hold,[ID,DATE]].copy(); qkeys=qkeys.merge(qi_ref,on=[ID,DATE],how="left",validate="one_to_one")
    out=pd.DataFrame({"anon_polygon_id":qkeys[ID],"date":qkeys[DATE],"truth":truth[np.flatnonzero(hold)]})
    for kind in ("lgbm_fast","lgbm_wide","lgbm_deep"):
        print("fit",kind, X.shape, flush=True); m=fit_lgb(X,y,kind); allp=np.clip(m.predict(qx),-.5,1.2)
        out[kind]=qkeys.pred_i.map(pd.Series(allp,index=np.arange(len(allp)))).to_numpy()
    base=pd.read_csv(RES/"private_cohort_blend_holdout_predictions.csv",parse_dates=[DATE])
    out=out.merge(base[[ID,DATE,"cohort","year","ext40","ext20","joint40"]],on=[ID,DATE],how="left",validate="one_to_one")
    rows=[]
    for cohort,g in [("all",out),("history",out[out.year<2025]),("2025",out[out.year==2025]),("new_history",out[(out.cohort=="new")&(out.year<2025)]),("new_2025",out[(out.cohort=="new")&(out.year==2025)]),("shared_2025",out[(out.cohort=="shared")&(out.year==2025)])]:
        for c in ["ext40","lgbm_fast","lgbm_wide","lgbm_deep"]:
            e=g[c].to_numpy(float)-g.truth.to_numpy(float); rows.append({"cohort":cohort,"method":c,"n":len(g),"rmse":float(np.sqrt(np.mean(e*e)))})
    rr=pd.DataFrame(rows); rr.to_csv(RES/"lgbm_holdout_results.csv",index=False); out.to_csv(RES/"lgbm_holdout_predictions.csv",index=False)
    print(rr.to_string(index=False)); print("elapsed",round(time.time()-t0,1),flush=True)

if __name__=="__main__": main()

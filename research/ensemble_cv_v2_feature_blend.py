"""Evaluate the ready ``feature_hgb_v2`` exact OOF predictions as a blend.

The feature-HGB artifact currently covers exact hidden-DOY folds only.  This
script therefore reports exact-only evidence and explicitly does not create a
private submission from it.
"""
from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"

def rm(y,p): return float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2)))

def main():
    f=pd.read_csv(R/"feature_hgb_v2_predictions.csv",parse_dates=["date"],low_memory=False)
    p=f.pivot_table(index=["year","anon_polygon_id","date","truth"],columns="kind",values="pred",aggfunc="first").reset_index(); p.columns.name=None
    # Exact peer/shock rows carry the same truth/key.  Normalize partition by
    # year and join only on immutable AOI/date keys.
    q=pd.read_csv(R/"paired_aoi_v2_predictions.csv",parse_dates=["date"],low_memory=False)
    q=q[q.family.eq("exact")].copy(); q["year2"]=q.date.dt.year
    s=pd.read_csv(R/"overnight_next_shock_predictions.csv",parse_dates=["date"],low_memory=False); s=s[s.candidate.eq("baseline") & s.dataset.eq("exact_hidden_doy")].copy(); s["year2"]=s.date.dt.year
    q=q.merge(s[["anon_polygon_id","date","year2","shock","state"]],on=["anon_polygon_id","date","year2"],how="left",validate="one_to_one")
    q=q.merge(p,on=["anon_polygon_id","date","year"],how="inner",validate="one_to_one")
    if len(q)!=len(p): raise ValueError(f"join rows {len(q)} != {len(p)}")
    q["base20"]=.8*q.hgb+.2*q.lag; q["base30"]=.7*q.hgb+.3*q.lag; q["base40"]=.6*q.hgb+.4*q.lag
    q["canon"]=q.date.dt.dayofyear.isin({97,113,129,145,161,177,193,209,225,241,257,273,289})
    q["local40"]=q.base40.copy(); ok=q.n16_c60_r125_k2.notna(); q.loc[ok,"local40"]=.9*q.loc[ok,"base40"]+.1*q.loc[ok,"n16_c60_r125_k2"]; q["local40"] += np.where(q.canon,0,.35*q.shock.fillna(0)-.2*q.state.fillna(0))
    rows=[]
    methods={"hgb":q.hgb,"base20":q.base20,"base30":q.base30,"local40":q.local40,"feature_default":q.default,"feature_regular":q.regular,"feature_wide":q.wide}
    # Basic models and feature-HGB blends.  Blend weights denote feature-HGB
    # mass; no weight is selected for private deployment here.
    for name,v in methods.items(): rows.append({"method":name,"weight":np.nan,"rmse":rm(q.truth,v),"n":len(q)})
    for kind in ["default","regular","wide"]:
      for base_name in ["hgb","base20","base30","local40"]:
        for w in np.arange(.0,1.01,.05):
          pred=(1-w)*q[base_name].to_numpy(float)+w*q[kind].to_numpy(float)
          rows.append({"method":f"{base_name}+feature_{kind}","weight":round(float(w),2),"rmse":rm(q.truth,pred),"n":len(q)})
    out=pd.DataFrame(rows).sort_values("rmse"); out.to_csv(R/"ensemble_cv_v2_feature_blend_results.csv",index=False,float_format="%.9f")
    # Per-year diagnostics for the best blend of each feature kind/base.
    best=[]
    for kind in ["default","regular","wide"]:
      for base_name in ["hgb","base20","base30","local40"]:
        z=out[(out.method==f"{base_name}+feature_{kind}")].iloc[0]
        w=float(z.weight); pred=(1-w)*q[base_name].to_numpy(float)+w*q[kind].to_numpy(float)
        for yr,g in q.assign(_pred=pred).groupby("year"):
          best.append({"kind":kind,"base":base_name,"weight":w,"year":yr,"rmse":rm(g.truth,g._pred),"n":len(g)})
    pd.DataFrame(best).to_csv(R/"ensemble_cv_v2_feature_blend_by_year.csv",index=False,float_format="%.9f")
    report=["# Exact OOF feature-HGB blend audit","",f"Rows joined: {len(q)} (six exact years). feature_hgb_v2 has no random/private-like OOF yet, so results are exact-only.","", "## Best pooled methods", "", out.head(30).to_string(index=False), "", "Feature-HGB is not promoted to private output until a visible-only full-private fit and random-protocol check exist."]
    (R/"ensemble_cv_v2_feature_blend_report.md").write_text("\n".join(report)+"\n",encoding="utf-8"); print("\n".join(report))

if __name__=="__main__": main()

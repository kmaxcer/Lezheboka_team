"""Compact diagnostics for the synthetic NDVI benchmark."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")

def main() -> None:
    tr = pd.read_csv(ROOT / "train_dataset.csv", parse_dates=["date"])
    pr = pd.read_csv(ROOT / "private_features.csv", parse_dates=["date"])
    for d in (tr, pr):
        d["_src"] = np.select(
            [d.s2_ndvi.notna(), d.landsat_ndvi.notna(), d.modis_ndvi.notna()],
            ["s2", "landsat", "modis"], default="none")
        d["_year"] = d.date.dt.year
        d["_doy"] = d.date.dt.dayofyear
    print("shapes", tr.shape, pr.shape)
    print("weather uniqueness by date", tr.groupby("date")["era5_temp_c"].nunique().describe().to_dict(), tr.groupby("date")["era5_precip_mm"].nunique().describe().to_dict())
    print("sources", tr._src.value_counts().to_dict())
    known = tr[tr.primary_ndvi.notna()].copy()
    # adjacent known observations and raw first differences by AOI/year
    rows=[]
    for (pid,year), g in known.sort_values("date").groupby(["anon_polygon_id","_year"]):
        y=g.primary_ndvi.to_numpy(float); dt=g.date.diff().dt.days.to_numpy(float)
        dy=np.diff(y)
        rows.append((pid,year,len(y),np.nanmedian(dt[1:]),np.nanmedian(np.abs(dy)),np.quantile(np.abs(dy),[.5,.9,.95,.99]).tolist(),np.corrcoef(y[:-1],y[1:])[0,1] if len(y)>2 else np.nan))
    rr=pd.DataFrame(rows,columns=["pid","year","n","med_dt","med_absdy","q_absdy","ac1"])
    print("group obs",rr[["n","med_dt","med_absdy","ac1"]].describe().to_string())
    print("abs diff quantiles",np.quantile(np.concatenate([np.asarray(x) for x in rr.q_absdy]),[0,.1,.25,.5,.75,.9,1]))
    print("target corr lag by source")
    print(known.groupby("_src").primary_ndvi.agg(["count","mean","std","min","max"]).to_string())
    # sensor-target residuals and cross-sensor residuals
    for c in ["s2_ndvi","landsat_ndvi","modis_ndvi"]:
        z=known[c].notna()
        e=known.loc[z,c]-known.loc[z,"primary_ndvi"]
        print(c,"n",int(z.sum()),"bias",float(e.mean()),"sd",float(e.std()),"q",np.quantile(e,[0,.01,.05,.5,.95,.99,1]).tolist())
    # source schedule conditional on date/day-of-year and aoi
    src=known["_src"]
    for k in ["_doy","date"]:
        tab=pd.crosstab(known[k],src,normalize="index")
        ent=-(tab.replace(0,np.nan)*np.log2(tab.replace(0,np.nan))).sum(axis=1)
        print("source entropy",k,ent.describe().to_dict(),"mode accuracy",tab.max(axis=1).describe().to_dict())
    # hidden mask selection pattern vs available rows
    h=pr[pr.is_synthetic_gap].copy()
    print("hidden by aoi",h.groupby("anon_polygon_id").size().describe().to_dict())
    print("hidden day modulo",(h.date.dt.dayofyear%16).value_counts().sort_index().to_dict())
    print("hidden date doy",h.date.dt.dayofyear.value_counts().head(30).to_dict())
    print("hidden crop",h.crop_type.value_counts().to_dict())

if __name__ == "__main__":
    main()

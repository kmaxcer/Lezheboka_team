"""Fast exploratory diagnostics for target reconstruction.

This module is research-only: it never edits competition inputs or production
submissions.  It compares several leakage-safe analogue predictors on the
same exact hidden-DOY and random private-like masks used elsewhere.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(R))
from teammate_sweep_postcorr import _mask_private  # noqa: E402

TARGET = "primary_ndvi"


def source(d: pd.DataFrame) -> np.ndarray:
    return np.select([d.s2_ndvi.notna(), d.landsat_ndvi.notna(), d.modis_ndvi.notna()], [0, 1, 2], -1)


def metric(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any(): return np.nan
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def mask_dynamic(d: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    x = d.copy().reset_index(drop=True)
    x["date"] = pd.to_datetime(x.date)
    dynamic = ["s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi", "landsat_ndwi",
               "modis_ndvi", "modis_evi", "modis_ndwi", "era5_temp_c", "era5_precip_mm", "year",
               TARGET, "doy", "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
               "n_reference_years", "status"]
    for c in dynamic:
        if c in x: x.loc[mask, c] = np.nan
    x["year"] = x.year.fillna(x.date.dt.year).astype(int)
    x["doy"] = x.doy.fillna(x.date.dt.dayofyear).astype(int)
    return x


def predictor_table(d: pd.DataFrame, mask: np.ndarray, use_all_years=True) -> pd.DataFrame:
    """Build analogue predictions from visible target values only.

    Includes same-AOI/year interpolation, same-AOI cross-year seasonal median,
    and date/crop peer medians.  All maps are formed after query rows are
    blanked, so they are valid at inference time.
    """
    x = d.copy().reset_index(drop=True); x.date = pd.to_datetime(x.date)
    if "_truth" not in x: x["_truth"] = pd.to_numeric(x[TARGET], errors="coerce")
    known = x[TARGET].notna().to_numpy(bool) & ~np.asarray(mask, bool)
    ids = x.anon_polygon_id.astype(str); yrs = x.date.dt.year.astype(int); doys = x.date.dt.dayofyear.astype(int)
    ords = x.date.map(pd.Timestamp.toordinal).to_numpy(float)
    y = pd.to_numeric(x[TARGET], errors="coerce").to_numpy(float)
    out = pd.DataFrame(index=np.arange(len(x)))
    # Same AOI/year linear interpolation using nearest visible target points.
    p = np.full(len(x), np.nan); nearest = np.full(len(x), np.nan); span = np.full(len(x), np.nan)
    for _, ix0 in pd.DataFrame({"id":ids,"yr":yrs}).groupby(["id","yr"],sort=False).groups.items():
        ix=np.asarray(ix0,int); k=ix[known[ix]]; q=ix[np.asarray(mask)[ix]]
        if len(k)==0: continue
        ko=np.argsort(ords[k]); k=k[ko]; kt=ords[k]; kv=y[k]
        for qi in q:
            pos=int(np.searchsorted(kt,ords[qi],side="left")); li=pos-1 if pos>0 else -1; ri=pos if pos<len(k) else -1
            if li>=0 and ri>=0:
                span[qi]=kt[ri]-kt[li]; p[qi]=kv[li]+(kv[ri]-kv[li])*(ords[qi]-kt[li])/(kt[ri]-kt[li]) if kt[ri]>kt[li] else (kv[li]+kv[ri])/2
                nearest[qi]=min(ords[qi]-kt[li],kt[ri]-ords[qi])
            elif li>=0: p[qi]=kv[li]; nearest[qi]=ords[qi]-kt[li]
            elif ri>=0: p[qi]=kv[ri]; nearest[qi]=kt[ri]-ords[qi]
    out["interp"] = p; out["span"] = span; out["nearest"] = nearest
    # Cross-year same-AOI DOY profile, with a range of windows.
    obs = pd.DataFrame({"id":ids[known].to_numpy(),"yr":yrs[known].to_numpy(),"doy":doys[known].to_numpy(),"y":y[known]})
    qd = pd.DataFrame({"id":ids.to_numpy(),"yr":yrs.to_numpy(),"doy":doys.to_numpy()})
    for rad in (0, 2, 4, 7, 10, 15, 21, 30):
        # robust median among other years; when rad=0 exact DOY only
        vals=[]
        for _,r in qd.loc[np.asarray(mask)].iterrows():
            z=obs[(obs.id==r.id) & (obs.yr!=r.yr) & (abs(obs.doy-r.doy)<=rad if rad else (obs.doy==r.doy))]
            vals.append(float(z.y.median()) if len(z) else np.nan)
        out.loc[np.asarray(mask),f"crossyr_{rad}"] = vals
    # Date/crop and date all peers.  Crop is static and can be missing.
    crop=x.crop_type.astype(str)
    oo=pd.DataFrame({"date":x.date[known].to_numpy(),"crop":crop[known].to_numpy(),"y":y[known]})
    for key,name in [("date","date_peer"),("crop","date_crop_peer")]:
        if key=="date": mp=oo.groupby("date").y.median(); out[name]=x.date.map(mp).to_numpy()
        else:
            mp=oo.groupby(["date","crop"]).y.median(); out[name]=pd.MultiIndex.from_arrays([x.date,crop]).map(mp).to_numpy()
    # AOI season profile pooled by DOY (optionally crop-conditioned globally).
    for rad in (0,2,4,7,10,15,21,30):
        vals=[]
        for dd in doys[np.asarray(mask)]:
            z=obs[(abs(obs.doy-dd)<=rad if rad else (obs.doy==dd))]
            vals.append(float(z.y.median()) if len(z) else np.nan)
        out.loc[np.asarray(mask),f"global_doy_{rad}"]=vals
    return out.loc[np.asarray(mask)].reset_index(drop=True)


def evaluate_one(name, frame, mask):
    truth=frame.loc[mask,"_truth"].to_numpy(float)
    tab=predictor_table(frame,mask)
    rows=[]
    for c in tab.columns:
        if c in ("span","nearest"): continue
        rows.append({"method":c,"rmse":metric(truth,tab[c]),"coverage":float(np.isfinite(tab[c]).mean()),"n":len(truth)})
    # Simple per-row selection/blends based only on observable span/coverage.
    for a,b in [("interp","crossyr_0"),("interp","crossyr_7"),("interp","date_crop_peer"),("crossyr_7","date_crop_peer")]:
        for w in (.1,.25,.5,.75,.9):
            pa=tab[a].to_numpy(float); pb=tab[b].to_numpy(float); p=np.where(np.isfinite(pa)&np.isfinite(pb),(1-w)*pa+w*pb,np.where(np.isfinite(pa),pa,pb))
            rows.append({"method":f"{a}+{b}_{w}","rmse":metric(truth,p),"coverage":float(np.isfinite(p).mean()),"n":len(truth)})
    z=frame.loc[mask,["anon_polygon_id","date"]].copy().reset_index(drop=True); z["truth"]=truth; z=pd.concat([z,tab],axis=1); z["partition"]=name
    return pd.DataFrame(rows),z


def main():
    tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=["date"],low_memory=False); pr=pd.read_csv(DATA/"private_features.csv",parse_dates=["date"],low_memory=False)
    allr=[]; allz=[]
    for yr in (2019,2020,2021,2022,2023,2024):
        f,t=make_fold(tr.copy(),pr.copy(),yr); f=f.reset_index(drop=True)
        # make_fold retains the unmasked labels in ``_truth`` even though the
        # visible target column is blanked on query rows.
        f["_truth"]=pd.to_numeric(f["_truth"],errors="coerce"); m=f.is_synthetic_gap.fillna(False).to_numpy(bool)
        r,z=evaluate_one(f"exact{yr}",f,m); allr.append(r); allz.append(z)
    for seed in (0,1,2):
        f,m=_mask_private(pr.copy(),seed); f=f.reset_index(drop=True); f["_truth"]=pd.to_numeric(f["_truth"],errors="coerce")
        r,z=evaluate_one(f"random{seed}",f,np.asarray(m,bool)); allr.append(r); allz.append(z)
    rr=pd.concat(allr,ignore_index=True); agg=rr.groupby("method",as_index=False).apply(lambda g:pd.Series({"n":g.n.sum(),"rmse":np.sqrt(np.average(g.rmse**2,weights=g.n)),"coverage":np.average(g.coverage,weights=g.n)}),include_groups=False).reset_index(drop=True).sort_values("rmse")
    rr.to_csv(R/"deep_signal_eval_rows.csv",index=False); agg.to_csv(R/"deep_signal_eval_aggregate.csv",index=False); pd.concat(allz,ignore_index=True).to_csv(R/"deep_signal_eval_predictions.csv",index=False)
    print(agg.head(40).to_string(index=False))


if __name__=="__main__": main()

"""Evaluate a cross-AOI date-factor correction for hidden NDVI rows.

The same satellite acquisition date is shared by many AOIs.  After removing
an AOI's seasonal profile, the remaining date-level anomaly can be estimated
from other fields.  This is a deliberately small, leakage-safe experiment:
the query row is masked before profiles and factors are built.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
from infer import predict_private  # noqa: E402
from validate import make_fold  # noqa: E402


def _profile_predict(z: pd.DataFrame, known: np.ndarray, q: pd.DataFrame, *, bin_days: int = 8):
    """Return AOI seasonal profile values for q and residual factors.

    Profiles are medians by AOI/DOY bin, then linearly interpolated within the
    available bins.  A global/crop fallback keeps sparse AOIs usable.
    """
    x = z.loc[known, ["anon_polygon_id", "_doy", "primary_ndvi", "crop_type", "_year", "date"]].copy()
    x["_bin"] = (x["_doy"] // bin_days).astype(int)
    # Trim physically implausible target outliers for the latent profile only;
    # hidden target itself is still scored without clipping.
    x.loc[~x.primary_ndvi.between(-0.5, 1.2), "primary_ndvi"] = np.nan
    x = x.dropna(subset=["primary_ndvi"])
    aoi_bin = x.groupby(["anon_polygon_id", "_bin"], observed=True).primary_ndvi.median()
    crop_bin = x.groupby(["crop_type", "_bin"], observed=True).primary_ndvi.median()
    glob_bin = x.groupby("_bin", observed=True).primary_ndvi.median()

    def lookup(pid, crop, doy):
        b = int(doy // bin_days)
        # Use a local weighted profile across nearby bins, not only one bin.
        cand = []
        for db in range(-2, 3):
            bb = b + db
            val = aoi_bin.get((pid, bb), np.nan)
            if np.isfinite(val): cand.append((abs(db), float(val)))
        if not cand:
            for db in range(-2, 3):
                val = crop_bin.get((crop, b + db), np.nan)
                if np.isfinite(val): cand.append((abs(db), float(val)))
        if not cand:
            for db in range(-2, 3):
                val = glob_bin.get(b + db, np.nan)
                if np.isfinite(val): cand.append((abs(db), float(val)))
        if not cand: return np.nan
        w = np.array([1.0 / (1.0 + d) for d, _ in cand]);v=np.array([v for _,v in cand])
        return float(np.average(v, weights=w))

    prof = np.array([lookup(pid, crop, doy) for pid, crop, doy in zip(q.anon_polygon_id, q.crop_type.astype(str), q._doy)], float)
    # Date factors from all known observations, after subtracting their AOI
    # profile.  Reuse profile lookup for each known row (small enough for CV).
    xp = np.array([lookup(pid, crop, doy) for pid, crop, doy in zip(x.anon_polygon_id, x.crop_type.astype(str), x._doy)], float)
    x["_res"] = x.primary_ndvi.to_numpy(float) - xp
    # Robust factors by exact date; use medians, and also same-crop medians.
    x["_date"] = x["date"].to_numpy()
    x["_factor"] = x._res
    allfac = x.groupby("_date", observed=True)._factor.median()
    cropfac = x.groupby(["_date", "crop_type"], observed=True)._factor.median()
    factors=[]
    for dt,crop in zip(q.date,q.crop_type.astype(str)):
        v=cropfac.get((dt,crop),np.nan)
        if not np.isfinite(v):v=allfac.get(dt,np.nan)
        factors.append(v)
    return prof, np.asarray(factors,float)


def evaluate():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    tr["_year"] = tr.date.dt.year; tr["_doy"] = tr.date.dt.dayofyear
    rec=[]
    for year in range(2019, 2025):
        f, truth = make_fold(tr, pr, year)
        z=f.copy();z["_year"]=z.date.dt.year;z["_doy"]=z.date.dt.dayofyear
        syn=z.is_synthetic_gap.to_numpy(bool);known=np.isfinite(z.primary_ndvi.to_numpy(float)) & ~syn
        q=z.loc[syn].copy().reset_index(drop=True); y=truth.to_numpy(float)
        base=predict_private(f,k=8,bin_days=30).primary_ndvi_pred.to_numpy(float)
        for bd in [4,8,12,16,24,30]:
            prof,fac=_profile_predict(z,known,q,bin_days=bd)
            for alpha in [0,.15,.25,.35,.5,.7,1.0]:
                # Factor-corrected profile; blend with local production.  If
                # no exact-date peers exist, retain base.
                p=np.array(base,copy=True)
                valid=np.isfinite(prof)
                cand=prof+alpha*np.nan_to_num(fac,nan=0.0)
                # adaptive blend: factor is most useful when date has many
                # peer observations; fixed .35 is intentionally conservative.
                p[valid]=.65*base[valid]+.35*cand[valid]
                e=p-y
                rec.append((year,bd,alpha,len(y),float(np.sqrt(np.mean(e*e))),float(np.mean(abs(e)))))
    out=pd.DataFrame(rec,columns=['year','bin_days','alpha','n','rmse','mae'])
    def _agg(z):
        return pd.Series({'n': z['n'].sum(), 'rmse': np.sqrt(np.average(z['rmse'] ** 2, weights=z['n'])), 'mae': np.average(z['mae'], weights=z['n'])})
    agg=out.groupby(['bin_days','alpha']).apply(_agg).reset_index().sort_values('rmse')
    print(agg.head(30).to_string(index=False));out.to_csv(ROOT/'research'/'factor_eval_results.csv',index=False)


if __name__=='__main__':evaluate()

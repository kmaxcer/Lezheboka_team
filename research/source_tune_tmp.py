"""Experimental source posterior/local-k tuning on private-like folds.

This is a disposable research script; it does not alter production inference.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import (_prepare, _fit_source_maps, _local_source_prediction,
                   _mode_posteriors, _query_posterior, SOURCES)
from validate import make_fold

DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")


def truth_source(d: pd.DataFrame) -> np.ndarray:
    return np.select([d.s2_ndvi.notna(), d.landsat_ndvi.notna(), d.modis_ndvi.notna()],
                     ["s2", "landsat", "modis"], default="none")


def source_date_prior(d: pd.DataFrame, known: np.ndarray, smoothing: float = .25):
    z = pd.DataFrame({"date": d.date.astype(str), "src": d._src.astype(str), "known": known})
    z = z[z.known & z.src.isin(SOURCES)]
    tab = z.pivot_table(index="date", columns="src", values="known", aggfunc="sum", fill_value=0)
    for s in SOURCES:
        if s not in tab: tab[s] = 0
    p = tab[list(SOURCES)].to_numpy(float) + smoothing
    p /= p.sum(axis=1, keepdims=True)
    return {k: v for k, v in zip(tab.index, p)}


def collect(fold: pd.DataFrame, truth: pd.Series, year: int):
    d = _prepare(fold.reset_index(drop=True))
    syn = d.is_synthetic_gap.to_numpy(bool)
    y = d.primary_ndvi.to_numpy(float)
    known = np.isfinite(y)
    x = d._ord.to_numpy(float); src = d._src.to_numpy(object)
    maps = _fit_source_maps(d, known, 30)
    aoi, crop, glob, date = _mode_posteriors(d, known)
    dp = source_date_prior(d, known)
    groups = d.groupby(["anon_polygon_id", "_year"], sort=False).groups
    ts = truth_source(fold.reset_index(drop=True))
    qidx = np.flatnonzero(syn)
    rec = []
    ks = [4, 6, 8, 10, 12, 16, 20]
    for q in qidx:
        key = (d.anon_polygon_id.iat[q], d._year.iat[q]); ii = np.asarray(groups[key], dtype=int)
        kk = ii[known[ii]]
        # date prior from known same-date rows (not the query itself)
        datep = dp.get(str(d.date.iat[q]), np.array([1/3]*3))
        basep = _query_posterior(d, int(q), aoi, crop, glob, date, date_weight=1.0)
        row = {"year": year, "truth": float(truth.iloc[len(rec)]), "src": ts[q],
               "doy": int(d._doy.iat[q]), "can": int(((int(d._doy.iat[q])-97)%16)==0),
               "date_n": int((d.date.astype(str).to_numpy()[known] == str(d.date.iat[q])).sum()),
               "p_s2": basep[0], "p_ls": basep[1], "p_md": basep[2],
               "dp_s2": datep[0], "dp_ls": datep[1], "dp_md": datep[2]}
        for k in ks:
            vals=[]
            for s in SOURCES:
                v=_local_source_prediction(x[q],kk,x,y,src,s,maps,int(d._doy.iat[q]),30,k)
                row[f"{s}_{k}"]=v; vals.append(v)
            # base and date-prior mixes
            for nm,p in [("base",basep),("date",datep)]:
                good=np.isfinite(vals); row[f"mix_{nm}_{k}"]=float(np.average(np.asarray(vals)[good],weights=np.asarray(p)[good])) if good.any() else np.nan
            # phase prior: preserve date mix but add canonical MODIS knowledge
            pp=np.asarray(datep,float).copy()
            if row["can"]:
                pp=(pp + np.array([.10,.10,.80]))/2
            else:
                pp[2]=0.0; pp/=pp.sum()
            good=np.isfinite(vals); row[f"mix_phase_{k}"]=float(np.average(np.asarray(vals)[good],weights=pp[good])) if good.any() else np.nan
        # source-specific k choices, using base p
        for combo in [(4,10,20),(4,8,16),(6,10,20),(6,12,20),(4,12,20)]:
            vals=np.array([row[f"{s}_{k}"] for s,k in zip(SOURCES,combo)])
            good=np.isfinite(vals); row["mix_k_"+"_".join(map(str,combo))]=float(np.average(vals[good],weights=basep[good])) if good.any() else np.nan
        rec.append(row)
    return pd.DataFrame(rec)


def main():
    tr=pd.read_csv(DATA/"train_dataset.csv", parse_dates=["date"])
    pr=pd.read_csv(DATA/"private_features.csv", parse_dates=["date"])
    allr=[]
    for yr in [2019,2020,2021,2022,2023,2024]:
        f,t=make_fold(tr,pr,yr); r=collect(f,t,yr); allr.append(r); print(yr,len(r),flush=True)
    z=pd.concat(allr,ignore_index=True); z.to_csv(ROOT/"research"/"source_tune_tmp.csv",index=False)
    out=[]
    for c in z.columns:
        if c in {"year","truth","src","doy","can","date_n","p_s2","p_ls","p_md","dp_s2","dp_ls","dp_md"}: continue
        q=z[["truth",c]].dropna(); out.append((c,len(q),float(np.sqrt(np.mean((q[c]-q.truth)**2))),float(np.mean(q[c]-q.truth))))
    print(pd.DataFrame(out,columns=["method","n","rmse","bias"]).sort_values("rmse").head(80).to_string(index=False))
    print("BY SOURCE")
    for s in SOURCES:
        qz=z[z.src==s]; rr=[]
        for c in z.columns:
            if c in {"year","truth","src","doy","can","date_n","p_s2","p_ls","p_md","dp_s2","dp_ls","dp_md"}: continue
            q=qz[["truth",c]].dropna(); rr.append((c,len(q),float(np.sqrt(np.mean((q[c]-q.truth)**2)))))
        print(s,sorted(rr,key=lambda x:x[2])[:25])


if __name__ == "__main__": main()

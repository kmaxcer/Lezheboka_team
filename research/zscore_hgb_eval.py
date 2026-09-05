"""Quick leakage-safe comparison of target vs climatology-normalized HGB.

This is intentionally standalone: it reuses the archive feature builder and
tests whether predicting a local z-score (then reconstructing NDVI) helps on
the same private-like date masks used by the production audit.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
from agropulse.pipeline import build_features, FULL_FEATURES  # noqa: E402

TARGET = "primary_ndvi"
DYN = ["s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
       "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi",
       "era5_temp_c", "era5_precip_mm", "year", TARGET, "doy",
       "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
       "n_reference_years", "status"]


def clear(d: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    z = d.copy().reset_index(drop=True)
    m = np.asarray(mask, bool)
    for c in DYN:
        if c in z:
            z.loc[m, c] = np.nan
    z["year"] = z["year"].fillna(z.date.dt.year).astype(int)
    z["doy"] = z["doy"].fillna(z.date.dt.dayofyear).astype(int)
    return z


def model(seed=42):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=.03, max_iter=350,
        max_leaf_nodes=48, min_samples_leaf=40,
        l2_regularization=10., random_state=seed)


def run_fold(train: pd.DataFrame, private: pd.DataFrame, year: int):
    fold, truth = make_fold(train.copy(), private.copy(), year)
    qm = fold.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    # Build three pseudo masks from other years, matching the production v2
    # protocol.  Their labels are transformed using the *masked-frame*
    # interpolated climatology, never the hidden row's original fields.
    base_known = fold.primary_ndvi.notna().to_numpy(bool) & ~qm
    blocks_x=[]; blocks_y=[]; blocks_z=[]
    rng=np.random.default_rng(100+year)
    for seed in (0,1,2):
        pm=np.zeros(len(fold),bool); pool=base_known & fold.date.dt.year.ne(year).to_numpy()
        for _,ix0 in fold.loc[pool].groupby(["anon_polygon_id",fold.date.dt.year],sort=False).groups.items():
            ix=np.asarray(ix0,int); n=max(1,int(round(.18*len(ix))))
            pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
        comb=qm|pm; fr=clear(fold,comb); obs=fr[TARGET].where(~comb)
        x=build_features(fr,obs,pd.Series(comb,index=fr.index))
        ci=x["ndvi_climatology_mean_interp"].to_numpy(float); cs=x["ndvi_climatology_std_interp"].to_numpy(float)
        yy=fold[TARGET].to_numpy(float)
        good=np.isfinite(ci)&np.isfinite(cs)&(cs>.02)
        z=np.full(len(fold),np.nan);z[good]=(yy[good]-ci[good])/cs[good]
        blocks_x.append(x.loc[pm].reset_index(drop=True)); blocks_y.append(fold.loc[pm,TARGET].reset_index(drop=True));blocks_z.append(pd.Series(z[pm]))
    vf=clear(fold,qm);obs=vf[TARGET].where(~qm);xv=build_features(vf,obs,pd.Series(qm,index=vf.index)).loc[qm]
    y=truth.to_numpy(float); X=pd.concat(blocks_x,ignore_index=True); yt=pd.concat(blocks_y,ignore_index=True); zt=pd.concat(blocks_z,ignore_index=True)
    out=[]
    # direct target model
    md=model(1);md.fit(X,yt);pdirect=np.clip(md.predict(xv),-.3,1.2)
    out.append(("direct",pdirect))
    # z-score model, train only finite transformed labels
    ok=np.isfinite(zt.to_numpy(float)); mz=model(2);mz.fit(X.loc[ok],zt.loc[ok]); zz=mz.predict(xv)
    ci=xv["ndvi_climatology_mean_interp"].to_numpy(float);cs=xv["ndvi_climatology_std_interp"].to_numpy(float)
    pz=np.where(np.isfinite(ci)&np.isfinite(cs)&(cs>.02),ci+zz*cs,pdirect); pz=np.clip(pz,-.5,1.3)
    out.append(("zscore",pz))
    # damped blend, useful if z model only captures anomalies partly
    for w in (.25,.5,.75): out.append((f"blend{w}",np.clip((1-w)*pdirect+w*pz,-.5,1.3)))
    rows=[]
    for n,p in out: rows.append({"year":year,"kind":n,"n":len(y),"rmse":float(np.sqrt(np.mean((p-y)**2))),"mae":float(np.mean(np.abs(p-y)))})
    return rows


def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False)
    pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rows=[]
    for y in (2019,2020,2021,2022,2023,2024):
        print('year',y,flush=True);rows.extend(run_fold(tr,pr,y));print(pd.DataFrame(rows).tail(5).to_string(index=False),flush=True)
    out=pd.DataFrame(rows);out.to_csv(ROOT/'research/zscore_hgb_results.csv',index=False)
    agg=out.groupby('kind',as_index=False).apply(lambda g:pd.Series(n=int(g.n.sum()),rmse=float(np.sqrt(np.average(g.rmse**2,weights=g.n))),mae=float(np.average(g.mae,weights=g.n))),include_groups=False).reset_index(drop=True).sort_values('rmse')
    agg.to_csv(ROOT/'research/zscore_hgb_aggregate.csv',index=False);print(agg.to_string(index=False))


if __name__=='__main__':main()

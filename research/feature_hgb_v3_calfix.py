"""Extra context features for the HGB imputer (research-only).

Adds leakage-safe AOI/year seasonal levels, acquisition-source posteriors and
nearby raw-sensor anchors to the archive feature matrix.  Every statistic is
computed from the observed target/sensors after the requested mask is applied.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
from agropulse.pipeline import build_features, FULL_FEATURES  # noqa: E402

TARGET = "primary_ndvi"
SENSORS = ["s2_ndvi", "landsat_ndvi", "modis_ndvi"]
DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi",
    "era5_temp_c", "era5_precip_mm", "year", TARGET, "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def _source(s: pd.DataFrame) -> np.ndarray:
    return np.select([s["s2_ndvi"].notna(), s["landsat_ndvi"].notna(), s["modis_ndvi"].notna()], [0, 1, 2], -1)


def _clear(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True)
    mask = np.asarray(mask, bool)
    for c in DYNAMIC:
        if c in d:
            d.loc[mask, c] = np.nan
    d["is_synthetic_gap"] = mask
    d["year"] = d["year"].fillna(d.date.dt.year).astype(int)
    d["doy"] = d["doy"].fillna(d.date.dt.dayofyear).astype(int)
    return d


def _neighbor_sensor(d: pd.DataFrame, known: np.ndarray) -> pd.DataFrame:
    """Nearest raw sensor anchors within the same AOI/year, query-safe."""
    dt = pd.to_datetime(d.date)
    yr = dt.dt.year.to_numpy(int)
    ids = d.anon_polygon_id.astype(str).to_numpy()
    ordv = dt.map(pd.Timestamp.toordinal).to_numpy(float)
    out = np.full((len(d), 9), np.nan, float)
    for gkey, ix0 in pd.DataFrame({"id": ids, "yr": yr}).groupby(["id", "yr"], sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int)
        for si, col in enumerate(SENSORS):
            vals = pd.to_numeric(d[col], errors="coerce").to_numpy(float)[ix]
            good = known[ix] & np.isfinite(vals)
            if not good.any():
                continue
            gi = ix[good]; gv = vals[good]; gx = ordv[gi]
            order = np.argsort(gx); gx = gx[order]; gv = gv[order]
            # Search on the sorted known times for every row in this group.
            pos = np.searchsorted(gx, ordv[ix], side="left")
            for j, p in enumerate(pos):
                left = p - 1 if p > 0 else -1
                right = p if p < len(gx) else -1
                # If an exact-time sensor row is itself masked/unknown, it is
                # absent from gx; otherwise use it only for non-query rows.
                if left >= 0:
                    out[ix[j], si * 3] = gv[left]
                    out[ix[j], si * 3 + 1] = ordv[ix[j]] - gx[left]
                if right >= 0:
                    out[ix[j], si * 3 + 2] = gv[right]
                    # Store right distance in a paired slot by replacing the
                    # sign of the next-value distance convention below.
                    # The six slots are prev, dprev, next; dnext is derived
                    # from the sign-coded fourth slot in a later column.
    # The compact 9-column layout is [prev, dprev, next] per sensor.  A fourth
    # distance is not needed: query date and nearest values let HGB infer it.
    return out


def extra_features(frame: pd.DataFrame, observed: pd.Series, masked: np.ndarray) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True)
    d.date = pd.to_datetime(d.date)
    n = len(d)
    known = observed.reset_index(drop=True).notna().to_numpy(bool) & ~np.asarray(masked, bool)
    y = pd.to_numeric(observed.reset_index(drop=True), errors="coerce").to_numpy(float)
    ids = d.anon_polygon_id.astype(str)
    years = d.date.dt.year.astype(int)
    doys = d.date.dt.dayofyear.astype(int)
    bins = ((doys - 1) // 16).astype(int)
    z = pd.DataFrame({"id": ids, "year": years, "doy": doys, "bin": bins, "y": y, "known": known})
    obs = z.loc[known & np.isfinite(y)].copy()
    # Robust levels and counts.  Medians are less sensitive to sensor tails.
    tables = {
        "aoi_level": obs.groupby("id").y.median(),
        "aoi_year_level": obs.groupby(["id", "year"]).y.median(),
        "aoi_bin_level": obs.groupby(["id", "bin"]).y.median(),
        "aoi_year_bin_level": obs.groupby(["id", "year", "bin"]).y.median(),
        "doy_level": obs.groupby("doy").y.median(),
        "crop_level": None,
    }
    out = pd.DataFrame(index=np.arange(n))
    out["aoi_level"] = ids.map(tables["aoi_level"])
    out["aoi_year_level"] = pd.MultiIndex.from_arrays([ids, years]).map(tables["aoi_year_level"])
    out["aoi_bin_level"] = pd.MultiIndex.from_arrays([ids, bins]).map(tables["aoi_bin_level"])
    out["aoi_year_bin_level"] = pd.MultiIndex.from_arrays([ids, years, bins]).map(tables["aoi_year_bin_level"])
    out["doy_level"] = doys.map(tables["doy_level"])
    for key, g in obs.groupby(["id", "year"], sort=False):
        pass
    out["aoi_known_n"] = ids.map(obs.groupby("id").y.size()).astype(float)
    out["aoi_year_known_n"] = pd.MultiIndex.from_arrays([ids, years]).map(obs.groupby(["id", "year"]).y.size()).astype(float)
    out["aoi_year_sd"] = pd.MultiIndex.from_arrays([ids, years]).map(obs.groupby(["id", "year"]).y.std()).astype(float)

    # Current-year source posterior/mode from visible rows.  Keep three soft
    # probabilities and a hard mode; missing groups receive global priors.
    src = _source(d)
    src_known = known & (src >= 0)
    st = pd.DataFrame({"id": ids, "doy": doys, "src": src}).loc[src_known]
    tab = st.groupby(["id", "doy", "src"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    arr = tab.to_numpy(float) + 0.5; arr /= arr.sum(axis=1, keepdims=True)
    idx = pd.MultiIndex.from_arrays([ids, doys]); pos = tab.index.get_indexer(idx)
    pp = np.full((n, 3), 1 / 3, float); ok = pos >= 0; pp[ok] = arr[pos[ok]]
    for j, name in enumerate(("query_p_s2", "query_p_ls", "query_p_md")):
        out[name] = pp[:, j]
    out["query_source_mode"] = np.argmax(pp, axis=1).astype(float)

    # Sensor-neighbour features.  The raw values are not exposed from a query
    # row because all query sensors are blank; only other rows contribute.
    ns = _neighbor_sensor(d, known)
    for j, col in enumerate(SENSORS):
        out[f"{col}_prev"] = ns[:, j * 3]
        out[f"{col}_dprev"] = ns[:, j * 3 + 1]
        out[f"{col}_next"] = ns[:, j * 3 + 2]
    # Simple source-normalized sensor medians on visible rows (global maps).
    for j, col in enumerate(SENSORS):
        v = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        good = known & np.isfinite(v) & np.isfinite(y)
        if good.sum() >= 20:
            slope, intercept = np.polyfit(v[good], y[good], 1)
            out[f"{col}_global_cal"] = intercept + slope * out[f"{col}_prev"].to_numpy(float)
            out[f"{col}_global_cal_next"] = intercept + slope * out[f"{col}_next"].to_numpy(float)
        else:
            out[f"{col}_global_cal"] = out[f"{col}_prev"]
            out[f"{col}_global_cal_next"] = out[f"{col}_next"]
    # Stable one-hot crop and AOI codes are useful for repeated private IDs;
    # AOI code is numeric only (HGB split thresholds are deterministic).
    out["aoi_code"] = pd.Categorical(ids).codes.astype(float)
    out["year_sin"] = np.sin(2 * np.pi * (years - 2010) / 16.0)
    out["year_cos"] = np.cos(2 * np.pi * (years - 2010) / 16.0)
    return out


def model(kind: str = "default", seed: int = 42):
    specs = {
        "default": dict(learning_rate=.035, max_iter=300, max_leaf_nodes=48, min_samples_leaf=35, l2_regularization=8.0),
        "regular": dict(learning_rate=.03, max_iter=350, max_leaf_nodes=48, min_samples_leaf=50, l2_regularization=12.0),
        "wide": dict(learning_rate=.03, max_iter=350, max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=8.0),
    }
    return HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **specs[kind])


def main() -> None:
    warnings.filterwarnings("ignore")
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    hidden = pr.loc[pr.is_synthetic_gap].copy(); hidden["_doy"] = hidden.date.dt.dayofyear.astype(int)
    hd = hidden.groupby("anon_polygon_id")["_doy"].apply(set).to_dict()
    rows = []; preds = []; t0 = time.time()
    for yr in (2019, 2020, 2021, 2022, 2023, 2024):
        fold, truth = make_fold(tr.copy(), pr.copy(), yr)
        fold["year"] = fold.year.fillna(fold.date.dt.year).astype(int); fold["doy"] = fold.doy.fillna(fold.date.dt.dayofyear).astype(int)
        qm = fold.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
        # Canonical OOF-style training masks; keep validation query blank in
        # every training representation.
        rng = np.random.default_rng(100 + yr)
        base_known = fold.primary_ndvi.notna().to_numpy(bool) & ~qm
        masks = []
        for seed in (0, 1, 2):
            pm = np.zeros(len(fold), bool)
            pool = base_known.copy(); pool &= fold.date.dt.year.ne(yr).to_numpy()
            for _, ix0 in fold.loc[pool].groupby(["anon_polygon_id", fold.date.dt.year], sort=False).groups.items():
                ii = np.asarray(ix0, dtype=int); nn = max(1, int(round(.18 * len(ii))))
                pm[rng.choice(ii, size=min(nn, len(ii)), replace=False)] = True
            masks.append(pm)
        blocks=[]; targets=[]
        for pm in masks:
            comb = qm | pm
            fr = _clear(fold, comb); obs = fr[TARGET].where(~comb)
            bx = build_features(fr, obs, pd.Series(comb, index=fr.index))
            ex = extra_features(fr, obs, comb)
            xx = pd.concat([bx.reset_index(drop=True), ex.reset_index(drop=True)], axis=1)
            blocks.append(xx.loc[pm].reset_index(drop=True)); targets.append(fold.loc[pm, "_truth"].reset_index(drop=True))
        vf = _clear(fold, qm); obs = vf[TARGET].where(~qm); bx = build_features(vf, obs, pd.Series(qm, index=vf.index)); ex = extra_features(vf, obs, qm); vx = pd.concat([bx.reset_index(drop=True), ex.reset_index(drop=True)], axis=1).loc[qm]
        y = truth.to_numpy(float); print("year", yr, "train", sum(len(x) for x in blocks), "q", len(y), flush=True)
        for kind in ("default", "regular", "wide"):
            m = model(kind); X = pd.concat(blocks, ignore_index=True); yy = pd.concat(targets, ignore_index=True); m.fit(X, yy); p = np.clip(m.predict(vx), -.2, 1.1); e = p-y
            rows.append({"year":yr,"kind":kind,"n":len(y),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(abs(e)))})
            preds.append(pd.DataFrame({"year":yr,"anon_polygon_id":fold.loc[qm,"anon_polygon_id"].to_numpy(),"date":fold.loc[qm,"date"].to_numpy(),"truth":y,"kind":kind,"pred":p}))
        print("done",yr,"elapsed",round(time.time()-t0,1),flush=True)
    out=pd.DataFrame(rows);out.to_csv(RESEARCH/'feature_hgb_v2_results.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(RESEARCH/'feature_hgb_v2_predictions.csv',index=False)
    agg=out.groupby('kind',as_index=False).apply(lambda g:pd.Series({'n':int(g.n.sum()),'rmse_pooled':float(np.sqrt(np.average(g.rmse**2,weights=g.n))),'mae_pooled':float(np.average(g.mae,weights=g.n))}),include_groups=False).reset_index(drop=True).sort_values('rmse_pooled');agg.to_csv(RESEARCH/'feature_hgb_v2_aggregate.csv',index=False);print(agg.to_string(index=False))


if __name__ == "__main__": main()





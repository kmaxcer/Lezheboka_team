"""Additional leakage-safe features for the NDVI imputer (research only).

The archive model mostly sees target anchors.  This module adds two pieces of
context that are available even when every dynamic field on a query row is
blank: raw-sensor/weather aggregates on the same acquisition date and
cross-year seasonal residual anchors for the same AOI.  All tables are built
after the requested mask is applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_hgb_v2 import extra_features

TARGET = "primary_ndvi"
SENSORS = ["s2_ndvi", "landsat_ndvi", "modis_ndvi"]
RAW = SENSORS + [
    "s2_evi", "s2_ndwi", "landsat_evi", "landsat_ndwi", "modis_evi", "modis_ndwi",
    "era5_temp_c", "era5_precip_mm",
]


def _src(d: pd.DataFrame) -> np.ndarray:
    return np.select([d["s2_ndvi"].notna(), d["landsat_ndvi"].notna(), d["modis_ndvi"].notna()], [0, 1, 2], -1)


def _nearest(values_x: np.ndarray, values_y: np.ndarray, qx: float, k: int = 3):
    """Weighted nearest values and distances, with circular DOY handled by caller."""
    if len(values_x) == 0:
        return np.nan, np.nan, np.nan
    dist = np.abs(values_x - qx)
    take = np.argsort(dist)[: min(k, len(dist))]
    dd = dist[take]; yy = values_y[take]
    good = np.isfinite(yy)
    if not good.any():
        return np.nan, np.nan, np.nan
    dd = dd[good]; yy = yy[good]
    w = 1.0 / (1.0 + dd)
    return float(np.average(yy, weights=w)), float(np.min(dd)), float(np.std(yy))


def extra_features_v3(frame: pd.DataFrame, observed: pd.Series, masked: np.ndarray) -> pd.DataFrame:
    """Return the v2 features plus date/sensor/cross-year context."""
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    n = len(d)
    masked = np.asarray(masked, bool)
    obs_y = pd.to_numeric(observed.reset_index(drop=True), errors="coerce").to_numpy(float)
    known = np.isfinite(obs_y) & ~masked
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    years = d["date"].dt.year.to_numpy(int)
    doys = d["date"].dt.dayofyear.to_numpy(int)
    # v2 contains the strong anchor/peer features.  It is intentionally called
    # with the same observed series and mask, so it cannot see query labels.
    out = extra_features(d, pd.Series(obs_y), masked).reset_index(drop=True)

    # Date-level raw channels.  Sensor observations on other AOIs at the same
    # date are legal evidence and often reveal a common acquisition condition.
    date_key = d["date"]
    for col in RAW:
        if col not in d.columns:
            continue
        val = pd.to_numeric(d[col], errors="coerce").to_numpy(float).copy()
        val[masked] = np.nan
        z = pd.DataFrame({"date": date_key, "v": val})
        g = z.groupby("date", observed=True)["v"]
        out[f"date_{col}_median"] = date_key.map(g.median())
        out[f"date_{col}_mean"] = date_key.map(g.mean())
        out[f"date_{col}_std"] = date_key.map(g.std())
        out[f"date_{col}_count"] = date_key.map(g.count()).astype(float)

    # Target date/source summaries (source is inferred only from visible raw
    # sensor availability; masked rows have no source label).
    src = _src(d)
    t = pd.DataFrame({"date": date_key, "src": src, "y": obs_y, "known": known})
    t = t[t.known & (t.src >= 0)]
    for s in (0, 1, 2):
        g = t[t.src == s].groupby("date", observed=True)["y"]
        out[f"date_target_s{s}_median"] = date_key.map(g.median())
        out[f"date_target_s{s}_mean"] = date_key.map(g.mean())
        out[f"date_target_s{s}_count"] = date_key.map(g.count()).astype(float)

    # Explicit cross-year anchors and residuals.  We use DOY rather than
    # ordinal date to align the same AOI across years; the circular distance
    # avoids a hard edge at Jan 1 (outside the growing season in this case).
    clim = pd.to_numeric(d.get("ndvi_climatology_mean", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    resid = obs_y - clim
    rows = np.full((n, 12), np.nan, float)
    # columns: cross-year weighted y/dist/sd, residual weighted/dist/sd,
    # all-year weighted y/dist/sd, number of cross-year observations.
    for aid, ix0 in pd.DataFrame({"id": ids, "year": years}).groupby(["id", "year"], sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int)
        # all observed rows of this AOI, split by current year for each query
        aid_mask = ids == str(aid[0])
        all_ix = np.flatnonzero(aid_mask & known)
        for i in ix:
            if not masked[i]:
                # Features for non-query rows are still built without their
                # own label; keep the same leave-one-out semantics.
                pass
            cross = all_ix[years[all_ix] != years[i]]
            # Exclude the current row even if a caller forgot to mark it.
            cross = cross[cross != i]
            if len(cross):
                dx = np.abs(doys[cross] - doys[i]); dx = np.minimum(dx, 366 - dx)
                take = np.argsort(dx)[: min(12, len(dx))]; yy = obs_y[cross[take]]; dd = dx[take]
                ok = np.isfinite(yy)
                if ok.any():
                    yy = yy[ok]; dd = dd[ok]; w = 1.0 / (1.0 + dd)
                    rows[i, 0] = np.average(yy, weights=w); rows[i, 1] = np.min(dd); rows[i, 2] = np.std(yy)
                    rr = resid[cross[take]][ok]; okr = np.isfinite(rr)
                    if okr.any():
                        rr = rr[okr]; wr = 1.0 / (1.0 + dd[okr]); rows[i, 3] = np.average(rr, weights=wr); rows[i, 4] = np.min(dd[okr]); rows[i, 5] = np.std(rr)
                    rows[i, 9] = float(len(yy))
            all2 = all_ix[all_ix != i]
            if len(all2):
                dx = np.abs(doys[all2] - doys[i]); dx = np.minimum(dx, 366 - dx); take = np.argsort(dx)[: min(12, len(dx))]; yy = obs_y[all2[take]]; dd = dx[take]; ok = np.isfinite(yy)
                if ok.any():
                    yy = yy[ok]; dd = dd[ok]; rows[i, 6] = np.average(yy, weights=1.0/(1.0+dd)); rows[i, 7] = np.min(dd); rows[i, 8] = np.std(yy)
    names = ["crossyear_y", "crossyear_dist", "crossyear_sd", "crossyear_resid", "crossyear_resid_dist", "crossyear_resid_sd", "aoi_season_y", "aoi_season_dist", "aoi_season_sd", "crossyear_n", "v3_doy_sin2", "v3_doy_cos2"]
    for j, name in enumerate(names[:10]): out[name] = rows[:, j]
    phase = 2.0 * np.pi * doys / 366.0
    out["v3_doy_sin2"] = np.sin(2 * phase); out["v3_doy_cos2"] = np.cos(2 * phase)
    return out.replace([np.inf, -np.inf], np.nan)

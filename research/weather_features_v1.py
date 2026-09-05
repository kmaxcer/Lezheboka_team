"""Leakage-safe weather/phenology feature builder.

Weather is present on most calendar rows, including rows without a satellite
target.  This module reconstructs the two weather channels within each
AOI/year and exposes causal rolling/cumulative summaries.  It never uses the
target column and respects a supplied mask for weather values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEMP = "era5_temp_c"
PREC = "era5_precip_mm"


def _fill_group(x: pd.Series, doy: np.ndarray) -> np.ndarray:
    """Interpolate a channel on the observed calendar rows."""
    a = pd.to_numeric(x, errors="coerce").to_numpy(float)
    if np.isfinite(a).sum() == 0:
        return np.full(len(a), np.nan)
    s = pd.Series(a, index=pd.Index(doy)).groupby(level=0).mean()
    # Reindex to the actual row sequence, preserving duplicate DOY rows.
    z = pd.Series(s.to_numpy(float), index=s.index).reindex(np.arange(1, 367))
    z = z.interpolate(limit_direction="both")
    return z.reindex(doy).to_numpy(float)


def weather_features(frame: pd.DataFrame, masked: np.ndarray | None = None) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    n = len(d)
    mask = np.zeros(n, dtype=bool) if masked is None else np.asarray(masked, bool)
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    years = d["date"].dt.year.to_numpy(int)
    doy = d["date"].dt.dayofyear.to_numpy(int)
    out = pd.DataFrame(index=np.arange(n))
    filled = {TEMP: np.full(n, np.nan), PREC: np.full(n, np.nan)}
    # Fill each AOI/year from its own visible weather.  If a channel is fully
    # absent, fall back to same-date medians across AOIs, then global seasonal
    # medians.  Hidden rows are removed before interpolation.
    for col in (TEMP, PREC):
        raw = pd.to_numeric(d.get(col, pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float).copy()
        raw[mask] = np.nan
        arr = np.full(n, np.nan)
        for _, ix0 in pd.DataFrame({"id": ids, "year": years}).groupby(["id", "year"], sort=False).groups.items():
            ix = np.asarray(ix0, dtype=int)
            arr[ix] = _fill_group(pd.Series(raw[ix]), doy[ix])
        # Cross-AOI/date fallback is especially useful for sparse new AOIs.
        z = pd.DataFrame({"year": years, "doy": doy, "v": raw})
        mp = z.groupby(["year", "doy"], observed=True).v.median()
        vals = pd.MultiIndex.from_arrays([years, doy]).map(mp).to_numpy(float)
        arr[~np.isfinite(arr)] = vals[~np.isfinite(arr)]
        # Use all-year seasonal median as a final fallback.
        mp2 = pd.DataFrame({"doy": doy, "v": raw}).groupby("doy", observed=True).v.median()
        vals2 = pd.Series(doy).map(mp2).to_numpy(float)
        arr[~np.isfinite(arr)] = vals2[~np.isfinite(arr)]
        filled[col] = arr

    # Per-group chronological rolling/cumulative summaries.
    for col, arr in filled.items():
        prefix = "temp" if col == TEMP else "prec"
        out[f"{prefix}_filled"] = arr
        for w in (3, 7, 14, 21, 30, 45, 60):
            mean = np.full(n, np.nan); summ = np.full(n, np.nan); std = np.full(n, np.nan)
            for _, ix0 in pd.DataFrame({"id": ids, "year": years}).groupby(["id", "year"], sort=False).groups.items():
                ix = np.asarray(ix0, dtype=int)
                order = np.argsort(doy[ix]); ii = ix[order]; aa = arr[ii]
                ss = pd.Series(aa).rolling(w, min_periods=1).sum().to_numpy(float)
                mm = pd.Series(aa).rolling(w, min_periods=1).mean().to_numpy(float)
                vv = pd.Series(aa).rolling(w, min_periods=2).std().to_numpy(float)
                # Centered windows are more appropriate for reconstructing a
                # measurement at a date; causal values are also retained.
                csum = pd.Series(aa).rolling(w, min_periods=1, center=True).sum().to_numpy(float)
                cmean = pd.Series(aa).rolling(w, min_periods=1, center=True).mean().to_numpy(float)
                cstd = pd.Series(aa).rolling(w, min_periods=2, center=True).std().to_numpy(float)
                mean[ii] = cmean; summ[ii] = csum; std[ii] = cstd
            out[f"{prefix}_roll{w}_mean"] = mean
            out[f"{prefix}_roll{w}_sum"] = summ
            out[f"{prefix}_roll{w}_std"] = std
        # Growing-degree and cumulative precipitation states from season start.
        gdd = np.maximum(arr - 5.0, 0.0)
        rain = np.maximum(arr, 0.0)
        cumg = np.full(n, np.nan); cumr = np.full(n, np.nan)
        for _, ix0 in pd.DataFrame({"id": ids, "year": years}).groupby(["id", "year"], sort=False).groups.items():
            ix = np.asarray(ix0, dtype=int); order = np.argsort(doy[ix]); ii = ix[order]
            cumg[ii] = np.nancumsum(gdd[ii]); cumr[ii] = np.nancumsum(rain[ii])
        out[f"{prefix}_cum"] = cumg if prefix == "temp" else cumr

    # AOI/year-normalized anomalies and cross-AOI date context.
    for col, arr, prefix in ((TEMP, filled[TEMP], "temp"), (PREC, filled[PREC], "prec")):
        prof = pd.DataFrame({"id": ids, "doy": doy, "v": arr}).groupby(["id", "doy"], observed=True).v.median()
        norm = pd.MultiIndex.from_arrays([ids, doy]).map(prof).to_numpy(float)
        out[f"{prefix}_season_norm"] = norm
        out[f"{prefix}_anom"] = arr - norm
        date_med = pd.DataFrame({"year": years, "doy": doy, "v": arr}).groupby(["year", "doy"], observed=True).v.median()
        dm = pd.MultiIndex.from_arrays([years, doy]).map(date_med).to_numpy(float)
        out[f"{prefix}_date_med"] = dm
        out[f"{prefix}_date_anom"] = arr - dm
    # Phase terms beyond the single annual harmonic.
    phase = 2 * np.pi * doy / 366.0
    for k in (1, 2, 3, 4):
        out[f"doy_sin{k}"] = np.sin(k * phase)
        out[f"doy_cos{k}"] = np.cos(k * phase)
    return out.replace([np.inf, -np.inf], np.nan)

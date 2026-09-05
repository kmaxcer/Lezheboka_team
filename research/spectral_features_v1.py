"""Leakage-safe temporal spectral features for NDVI gap filling.

The hidden rows have every dynamic channel removed, but neighbouring rows of
the same AOI/year still expose the raw sensor triplets.  This module turns
those neighbouring EVI/NDWI values into robust interpolation/ratio features.
It deliberately never reads a value from a row marked in ``masked``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SENSORS = ("s2", "landsat", "modis")
CHANNELS = ("ndvi", "evi", "ndwi")
# Keep the feature family deliberately compact.  Raw EVI ratios can be very
# heavy-tailed; the two physically interpretable combinations below are much
# more stable than exposing every possible ratio to the tree.
TRANSFORMS = ("ndvi", "evi", "ndwi", "ndvi_minus_ndwi", "evi_over_water")
LOCAL_SUFFIXES = ("prev", "dprev", "next", "dnext", "linear", "nearest", "dist", "sd", "n")


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 0.03) -> np.ndarray:
    """Finite, clipped ratio; spectral outliers are common in this data."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    z = a / (b + eps * np.where(b >= 0, 1.0, -1.0))
    return np.clip(z, -10.0, 10.0)


def _channel_transform(name: str, vals: dict[str, np.ndarray]) -> np.ndarray:
    """Derived spectral scalar used for a temporal neighbour search."""
    # NDVI itself is retained, while EVI/NDWI-derived transforms are robust to
    # the occasional enormous denominator/outlier in Sentinel-2 fields.
    nd = vals.get("ndvi", np.full(len(next(iter(vals.values()))), np.nan))
    ev = vals.get("evi", np.full(len(nd), np.nan))
    nw = vals.get("ndwi", np.full(len(nd), np.nan))
    if name == "ndvi":
        z = nd
    elif name == "evi":
        z = np.clip(ev, -2.0, 2.0)
    elif name == "ndwi":
        z = np.clip(nw, -2.0, 2.0)
    elif name == "ndvi_minus_ndwi":
        z = nd - nw
    elif name == "ndvi_plus_ndwi":
        z = nd + nw
    elif name == "evi_minus_ndvi":
        z = np.clip(ev, -2.0, 2.0) - nd
    elif name == "evi_over_ndvi":
        z = _safe_ratio(np.clip(ev, -2.0, 2.0), nd)
    elif name == "evi_over_water":
        z = _safe_ratio(np.clip(ev, -2.0, 2.0), 1.0 + nw)
    else:
        raise KeyError(name)
    z = np.asarray(z, float)
    z[~np.isfinite(z)] = np.nan
    return np.clip(z, -10.0, 10.0)


def _nearest_temporal(qx: np.ndarray, gx: np.ndarray, gv: np.ndarray):
    """Return prev/next/linear/nearest-k values for sorted time coordinates."""
    n = len(qx)
    out = np.full((n, 9), np.nan, float)
    if len(gx) == 0:
        return out
    order = np.argsort(gx)
    gx = np.asarray(gx, float)[order]
    gv = np.asarray(gv, float)[order]
    good = np.isfinite(gv) & np.isfinite(gx)
    gx, gv = gx[good], gv[good]
    if len(gx) == 0:
        return out
    pos = np.searchsorted(gx, qx, side="left")
    for j, p in enumerate(pos):
        li = p - 1 if p > 0 else -1
        ri = p if p < len(gx) else -1
        if li >= 0:
            out[j, 0] = gv[li]
            out[j, 1] = qx[j] - gx[li]
        if ri >= 0:
            out[j, 2] = gv[ri]
            out[j, 3] = gx[ri] - qx[j]
        # A linear interpolation is only meaningful with both sides.  For a
        # one-sided edge use the closest value and preserve its distance.
        if li >= 0 and ri >= 0:
            den = gx[ri] - gx[li]
            out[j, 4] = gv[li] + (gv[ri] - gv[li]) * (qx[j] - gx[li]) / den if den > 0 else (gv[li] + gv[ri]) / 2
        elif li >= 0:
            out[j, 4] = gv[li]
        elif ri >= 0:
            out[j, 4] = gv[ri]
        # Inverse-distance average of up to three nearest observations.
        lo = max(0, p - 3); hi = min(len(gx), p + 3)
        ii = np.arange(lo, hi)
        if len(ii):
            dd = np.abs(gx[ii] - qx[j]); take = ii[np.argsort(dd)[:3]]
            w = 1.0 / (1.0 + np.abs(gx[take] - qx[j]))
            out[j, 5] = np.average(gv[take], weights=w)
            out[j, 6] = np.min(np.abs(gx[take] - qx[j]))
            out[j, 7] = np.std(gv[take]) if len(take) > 1 else 0.0
            out[j, 8] = len(take)
    return out


def spectral_features(frame: pd.DataFrame, observed: pd.Series, masked: np.ndarray) -> pd.DataFrame:
    """Build spectral temporal/ratio features for every row in ``frame``.

    ``observed`` is the target sidecar after masking.  Raw channels from a
    masked row are ignored even if a caller accidentally left them populated.
    Features are computed separately within AOI/year and also over other years
    aligned by day-of-year, yielding useful context for long gaps.
    """
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    n = len(d)
    mask = np.asarray(masked, bool)
    # A row is usable as a spectral anchor only if its target is visible.  This
    # matches the competition construction (target exists iff a sensor exists)
    # and avoids leaking an organiser gap through raw channels.
    y = pd.to_numeric(observed.reset_index(drop=True), errors="coerce").to_numpy(float)
    known = np.isfinite(y) & ~mask
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    years = d["date"].dt.year.to_numpy(int)
    doy = d["date"].dt.dayofyear.to_numpy(int)
    # Continuous ordinal day for within-year interpolation; circular DOY for
    # cross-year search.
    ordinal = d["date"].map(pd.Timestamp.toordinal).to_numpy(float)
    # Build columns in numpy arrays and create the frame once at the end.  A
    # repeated ``out.loc[...] =`` insertion makes pandas' block manager highly
    # fragmented and is an order of magnitude slower on the full reference.
    arrays: dict[str, np.ndarray] = {}
    def _arr(name: str) -> np.ndarray:
        return arrays.setdefault(name, np.full(n, np.nan, float))

    # Source triplet values, clipped before deriving ratios.  Missing modis
    # NDWI is represented by an all-NaN channel and naturally ignored.
    transformed: dict[tuple[str, str], np.ndarray] = {}
    for s in SENSORS:
        vals = {}
        for c in CHANNELS:
            col = f"{s}_{c}"
            if col in d:
                z = pd.to_numeric(d[col], errors="coerce").to_numpy(float).copy()
                z[~known] = np.nan
            else:
                z = np.full(n, np.nan)
            vals[c] = z
        for name in TRANSFORMS:
            transformed[(s, name)] = _channel_transform(name, vals)

    # Within AOI/year nearest features.  The base set has 8 values plus a
    # source-specific confidence/availability indicator.
    groups = pd.DataFrame({"id": ids, "year": years}).groupby(["id", "year"], sort=False).groups
    for (aid, yr), ix0 in groups.items():
        ix = np.asarray(ix0, dtype=int)
        qx = ordinal[ix]
        for s in SENSORS:
            for name in TRANSFORMS:
                z = _nearest_temporal(qx, ordinal[ix][known[ix]], transformed[(s, name)][ix][known[ix]])
                pfx = f"sp_{s}_{name}"
                for j, suf in enumerate(LOCAL_SUFFIXES):
                    _arr(pfx + "_" + suf)[ix] = z[:, j]

    # Cross-year seasonal spectral anchors are intentionally omitted.  The v3
    # target feature set already supplies cross-year NDVI anchors; doing the
    # same exhaustive search for every spectral transform is expensive and
    # strongly collinear.  This module focuses on the local information that
    # is not present in the target-only anchor set.

    # Date-level robust spectral context across AOIs.  v3 already has raw
    # date aggregates, but ratios and source-calibrated summaries are novel.
    dates = d["date"]
    for s in SENSORS:
        for name in TRANSFORMS:
            z = transformed[(s, name)]
            q = pd.DataFrame({"date": dates, "z": z}).groupby("date", observed=True)["z"]
            _arr(f"sp_date_{s}_{name}_median")[:] = dates.map(q.median()).to_numpy(float)
            _arr(f"sp_date_{s}_{name}_mean")[:] = dates.map(q.mean()).to_numpy(float)
            _arr(f"sp_date_{s}_{name}_count")[:] = dates.map(q.count()).to_numpy(float)

    return pd.DataFrame(arrays, index=np.arange(n)).replace([np.inf, -np.inf], np.nan)


def compact_columns(columns: list[str]) -> list[str]:
    """A conservative subset for quick HGB screens (avoid 200-column noise)."""
    keep = []
    for c in columns:
        if "_xy_" in c or "_date_" in c:
            continue
        if c.endswith(("_prev", "_next", "_linear", "_nearest", "_dist", "_sd", "_dprev", "_dnext", "_n")):
            keep.append(c)
    return keep

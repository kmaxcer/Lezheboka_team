"""Overnight source-aware routing, residual correction and diversity audit.

This is a research-only evaluator.  It deliberately leaves ``outputs/`` and
the competition input CSVs untouched.  The estimator reuses the production
source calibration primitives from :mod:`src.infer` and the lag-aware local
primitive from :mod:`src.infer_lag`, but exposes the three source-domain
predictions so that soft and hard source assignment can be scored separately.

Two leakage-safe proxies are evaluated:

* ``exact_hidden_doy`` -- the private synthetic day-of-year pattern projected
  onto train years 2019--2024;
* ``private_like`` -- 15% random known private rows per AOI/year, seeds 0--2.

For every partition, source routing is evaluated with the true source retained
only in a sidecar.  A date/crop residual correction and a rank-1/2 PCA stack
are fitted on the other half of each partition, then scored on the held half.
These are intentionally conservative diagnostics rather than production
changes.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))

from infer import (  # noqa: E402
    SOURCES,
    _fit_source_maps,
    _local_source_prediction,
    _mode_posteriors,
    _prepare,
    _query_posterior,
    _safe_bool,
)
from infer_lag import _lagged_local_poly  # noqa: E402
from validate import make_fold  # noqa: E402


DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def _source_labels(frame: pd.DataFrame) -> np.ndarray:
    """Primary source labels before masking (evaluation sidecar only)."""
    s2 = np.isfinite(frame["s2_ndvi"].to_numpy(float))
    ls = np.isfinite(frame["landsat_ndvi"].to_numpy(float))
    md = np.isfinite(frame["modis_ndvi"].to_numpy(float))
    out = np.full(len(frame), "none", dtype=object)
    out[s2] = "s2"
    out[~s2 & ls] = "landsat"
    out[~s2 & ~ls & md] = "modis"
    return out


def _flag(s: pd.Series) -> np.ndarray:
    return _safe_bool(s)


def _mask_private_like(private: pd.DataFrame, seed: int, frac: float = 0.15) -> tuple[pd.DataFrame, np.ndarray]:
    """Mask observed private rows at the empirical 15% AOI/year rate."""
    d = private.copy().sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["_true_src"] = _source_labels(d)
    # Do not ask the predictor to score the organizer's already-hidden rows in
    # this proxy; they are simply unavailable calibration rows.
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(int(seed))
    mask = np.zeros(len(d), dtype=bool)
    pool = d["primary_ndvi"].notna().to_numpy(bool)
    years = d["date"].dt.year
    for _, ix in d.loc[pool].groupby(["anon_polygon_id", years], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        n = max(1, int(round(float(frac) * len(ii))))
        mask[rng.choice(ii, size=min(n, len(ii)), replace=False)] = True
    for col in DYNAMIC:
        if col in d:
            d.loc[mask, col] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    return d, mask


def _exact_fold(train: pd.DataFrame, private: pd.DataFrame, year: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Build exact hidden-DOY train fold with truth/source sidecars."""
    orig = train.copy().reset_index(drop=True)
    src = _source_labels(orig)
    fold, truth = make_fold(orig, private.copy(), int(year))
    fold = fold.reset_index(drop=True)
    hidden = fold["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
    # ``make_fold`` retains _truth but no source after dynamic masking.
    fold["_true_src"] = np.where(hidden, src, "none")
    fold["_truth"] = fold["_truth"].astype(float)
    return fold, hidden


def _make_2025_date_proxy(private: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Preserve real hidden-2025 date multiplicities using known peers."""
    d = private.copy().sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["_true_src"] = _source_labels(d)
    d["is_synthetic_gap"] = d["is_synthetic_gap"].fillna(False).astype(bool)
    y25 = d["date"].dt.year.eq(2025)
    real_hidden = d["is_synthetic_gap"].to_numpy(bool) & y25.to_numpy(bool)
    candidates = d[y25 & ~d["is_synthetic_gap"] & d["primary_ndvi"].notna()]
    hidden_counts = d.loc[real_hidden].groupby("date").size().to_dict()
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for date, count in hidden_counts.items():
        ix = candidates.index[candidates["date"].eq(date)].to_numpy()
        if len(ix):
            selected.extend(rng.choice(ix, size=min(int(count), len(ix)), replace=False).tolist())
    need = int(real_hidden.sum()) - len(selected)
    if need > 0:
        pool = candidates.index[~candidates.index.isin(selected)].to_numpy()
        if len(pool):
            selected.extend(rng.choice(pool, size=min(need, len(pool)), replace=False).tolist())
    hold = np.zeros(len(d), dtype=bool)
    hold[np.asarray(selected, dtype=int)] = True
    # Keep real hidden rows unavailable, but score only sampled known peers.
    allmask = hold | real_hidden
    for col in DYNAMIC:
        if col in d:
            d.loc[allmask, col] = np.nan
    d.loc[allmask, "is_synthetic_gap"] = True
    d.loc[hold, "_hold"] = True
    return d, hold


def _calibration_frame(frame: pd.DataFrame, train: pd.DataFrame | None) -> pd.DataFrame:
    """Return the same calibration population used by ``predict_private``."""
    if train is None:
        return _prepare(frame.copy().reset_index(drop=True))
    fr = frame.copy().reset_index(drop=True)
    tr = train.copy().reset_index(drop=True)
    cols = [c for c in fr.columns if c in tr.columns]
    return _prepare(pd.concat([tr[cols], fr[cols]], ignore_index=True, sort=False))


def _predict_matrix(
    frame: pd.DataFrame,
    train: pd.DataFrame | None = None,
    *,
    family: str = "base",
    k: int = 8,
    degree: int = 3,
    bin_days: int = 30,
    date_weight: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Produce per-source, soft and hard predictions for synthetic rows.

    Returns a query-aligned table and a 3-column posterior matrix.  ``soft``
    is the production-style posterior average; ``hard`` selects the modal
    source.  ``oracle`` is evaluation-only and selects the true source.
    """
    fr = frame.copy().reset_index(drop=True)
    df = _prepare(fr)
    syn = _flag(fr.get("is_synthetic_gap", pd.Series(False, index=fr.index)))
    y = df["primary_ndvi"].to_numpy(float)
    known = np.isfinite(y) & ~syn
    x = df["_ord"].to_numpy(float)
    src = df["_src"].to_numpy(object)

    calib = _calibration_frame(fr, train)
    known_cal = np.isfinite(calib["primary_ndvi"].to_numpy(float))
    maps = _fit_source_maps(calib, known_cal, bin_days=bin_days)
    aoi, crop, glob, date_prior = _mode_posteriors(calib, known_cal)

    qidx = np.flatnonzero(syn)
    pred = np.full((len(qidx), len(SOURCES)), np.nan, dtype=float)
    post = np.full((len(qidx), len(SOURCES)), np.nan, dtype=float)
    groups = df.groupby(["anon_polygon_id", "_year"], sort=False).groups
    qpos = {int(q): n for n, q in enumerate(qidx)}
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        kk = ii[known[ii]]
        for q in ii[syn[ii]]:
            n = qpos[int(q)]
            p = _query_posterior(
                df, int(q), aoi, crop, glob, date_prior,
                date_weight=float(date_weight),
            )
            post[n] = p
            for j, target in enumerate(SOURCES):
                if family == "lag":
                    value = _lagged_local_poly(
                        x[q], kk, x, y, src, target, maps,
                        int(df["_doy"].iat[q]), lags=None or {
                            ("s2", "landsat"): 0.0, ("landsat", "s2"): 0.0,
                            ("s2", "modis"): 8.0, ("modis", "s2"): -8.0,
                            ("landsat", "modis"): 5.0, ("modis", "landsat"): -5.0,
                        },
                        bin_days=bin_days, k=max(3, int(k)), degree=max(0, int(degree)),
                    )
                else:
                    value = _local_source_prediction(
                        x[q], kk, x, y, src, target, maps,
                        int(df["_doy"].iat[q]), bin_days=bin_days,
                        k=max(3, int(k)),
                    )
                pred[n, j] = value

    # Match infer.py's nearest-observation fallback for pathological groups.
    ids = df["anon_polygon_id"].to_numpy(object)
    for n, q in enumerate(qidx):
        if np.isfinite(pred[n]).any():
            continue
        same = np.flatnonzero(known & (ids == ids[q]))
        if len(same):
            v = y[same[np.argmin(np.abs(x[same] - x[q]))]]
        else:
            v = float(np.nanmedian(y[known])) if known.any() else 0.3
        pred[n, :] = v

    # Ensure a finite posterior even for very sparse calibration slices.
    badp = ~np.isfinite(post).all(axis=1)
    post[badp] = 1.0 / len(SOURCES)
    # Weighted and modal predictions.  A missing source estimate is ignored,
    # exactly as in the production routine.
    soft = np.full(len(qidx), np.nan)
    hard = np.full(len(qidx), np.nan)
    for n in range(len(qidx)):
        ok = np.isfinite(pred[n])
        if ok.any():
            pp = post[n].copy(); pp[~ok] = 0.0
            soft[n] = float(np.average(pred[n, ok], weights=pp[ok])) if pp[ok].sum() > 0 else float(np.mean(pred[n, ok]))
            hard[n] = float(pred[n, int(np.argmax(np.where(ok, post[n], -np.inf)))])
    out = pd.DataFrame({
        "row_index": qidx,
        "pred_s2": pred[:, 0], "pred_landsat": pred[:, 1], "pred_modis": pred[:, 2],
        "soft": soft, "hard": hard,
        "route": np.asarray(SOURCES, dtype=object)[np.argmax(post, axis=1)],
        "p_s2": post[:, 0], "p_landsat": post[:, 1], "p_modis": post[:, 2],
    })
    return out, post


def _attach_query(frame: pd.DataFrame, mask: np.ndarray, pred: pd.DataFrame, partition: str) -> pd.DataFrame:
    q = frame.loc[mask, ["anon_polygon_id", "date", "_truth", "_true_src"]].copy().reset_index()
    q = q.rename(columns={"index": "row_index"})
    q["date"] = pd.to_datetime(q["date"])
    z = q.merge(pred, on="row_index", how="left", validate="one_to_one")
    z["partition"] = partition
    z["year"] = z["date"].dt.year.astype(int)
    z["doy"] = z["date"].dt.dayofyear.astype(int)
    z["doy_bin"] = (z["doy"] // 16).astype(int)
    z["crop_type"] = frame.loc[mask, "crop_type"].fillna("unknown").astype(str).to_numpy()
    z["hash_parity"] = [int(hashlib.sha1(f"{a}|{d.date()}".encode()).hexdigest()[-1], 16) % 2 for a, d in zip(z.anon_polygon_id, z.date)]
    return z


def _metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return np.nan, np.nan
    e = p[ok] - y[ok]
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def _route_rows(q: pd.DataFrame, protocol: str, label_prefix: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y = q["_truth"].to_numpy(float)
    for method0 in ("soft", "hard"):
        method = f"{label_prefix}{method0}"
        p = q[method0].to_numpy(float)
        rm, ma = _metric(y, p)
        rows.append({"protocol": protocol, "partition": q["partition"].iloc[0], "method": method, "source": "all", "n": len(q), "rmse": rm, "mae": ma})
        for s in SOURCES:
            take = q["_true_src"].eq(s).to_numpy(bool)
            if take.any():
                rr, aa = _metric(y[take], p[take])
                rows.append({"protocol": protocol, "partition": q["partition"].iloc[0], "method": method, "source": s, "n": int(take.sum()), "rmse": rr, "mae": aa})
    # Oracle source and routing accuracy are useful bounds/diagnostics only.
    cols = {s: q[f"pred_{s if s != 'landsat' else 'landsat'}"].to_numpy(float) for s in SOURCES}
    oracle = np.full(len(q), np.nan)
    for s in SOURCES:
        take = q["_true_src"].eq(s).to_numpy(bool)
        oracle[take] = cols[s][take]
    rr, aa = _metric(y, oracle)
    rows.append({"protocol": protocol, "partition": q["partition"].iloc[0], "method": f"{label_prefix}oracle_source", "source": "all", "n": len(q), "rmse": rr, "mae": aa})
    route = q["route"].astype(str).to_numpy()
    true = q["_true_src"].astype(str).to_numpy()
    pmat = q[["p_s2", "p_landsat", "p_modis"]].to_numpy(float)
    smap = {s: i for i, s in enumerate(SOURCES)}
    tidx = np.asarray([smap.get(t, 0) for t in true], dtype=int)
    truep = pmat[np.arange(len(q)), tidx]
    rows.append({
        "protocol": protocol,
        "partition": q["partition"].iloc[0],
        "method": f"{label_prefix}route_accuracy",
        "source": "all",
        "n": len(q),
        "rmse": float(np.mean(route == true)),
        "mae": float(-np.mean(np.log(np.clip(truep, 1e-8, 1.0)))),
    })
    return rows


def _fit_group_residual(cal: pd.DataFrame, key_cols: list[str], min_n: int, clip: float = 0.03) -> dict[object, float]:
    z = cal.copy()
    z["resid"] = z["_truth"].to_numpy(float) - z["soft"].to_numpy(float)
    z = z[np.isfinite(z.resid)]
    if not len(z):
        return {}
    keys = key_cols[0] if len(key_cols) == 1 else key_cols
    out: dict[object, float] = {}
    for key, g in z.groupby(keys, dropna=False, observed=True):
        if len(g) < int(min_n):
            continue
        n = float(len(g))
        # Robust median plus count shrinkage; the bound is deliberately small.
        v = float(np.median(g.resid.to_numpy(float))) * min(1.0, n / (n + 8.0))
        out[key] = float(np.clip(v, -clip, clip))
    return out


def _apply_group_correction(cal: pd.DataFrame, test: pd.DataFrame, key_cols: list[str], weight: float, min_n: int) -> np.ndarray:
    vals = _fit_group_residual(cal, key_cols, min_n=min_n)
    p = test["soft"].to_numpy(float).copy()
    if not vals:
        return p
    for key, v in vals.items():
        if len(key_cols) == 1:
            take = test[key_cols[0]].eq(key).to_numpy(bool)
        else:
            take = np.ones(len(test), dtype=bool)
            for c, k in zip(key_cols, key if isinstance(key, tuple) else (key,)):
                take &= test[c].eq(k).to_numpy(bool)
        p[take] += float(weight) * v
    return np.clip(p, -0.5, 1.2)


def _pca_stack(cal: pd.DataFrame, test: pd.DataFrame, rank: int, blend: float = 1.0) -> np.ndarray:
    """Cross-fitted low-rank (PCA) stack over diverse source predictions."""
    names = ["soft", "hard", "pred_s2", "pred_landsat", "pred_modis"]
    # When available, include the lag-aware family as additional diversity;
    # this is what makes the stack useful beyond a trivial source average.
    for extra in ["lag_soft", "lag_hard", "lag_pred_s2", "lag_pred_landsat", "lag_pred_modis"]:
        if extra in cal.columns and extra in test.columns:
            names.append(extra)
    Xc = cal[names].to_numpy(float); Xt = test[names].to_numpy(float)
    # Deterministic column means; all source columns should be finite, but the
    # guard makes the routine safe for sparse edge groups.
    mu = np.nanmean(Xc, axis=0)
    Xc = np.where(np.isfinite(Xc), Xc, mu)
    Xt = np.where(np.isfinite(Xt), Xt, mu)
    X0 = Xc - mu
    try:
        _, _, vt = np.linalg.svd(X0, full_matrices=False)
    except np.linalg.LinAlgError:
        return test["soft"].to_numpy(float)
    r = max(1, min(int(rank), vt.shape[0]))
    basis = vt[:r]
    zc = X0 @ basis.T
    zt = (Xt - mu) @ basis.T
    yc = cal["_truth"].to_numpy(float)
    ok = np.isfinite(yc) & np.isfinite(zc).all(axis=1)
    if ok.sum() < max(30, 5 * r):
        return test["soft"].to_numpy(float)
    A = np.c_[np.ones(int(ok.sum())), zc[ok]]
    coef = np.linalg.lstsq(A, yc[ok], rcond=None)[0]
    pred = np.c_[np.ones(len(test)), zt] @ coef
    # A conservative blend toward the uncalibrated soft estimate.
    pred = (1.0 - float(blend)) * test["soft"].to_numpy(float) + float(blend) * pred
    return np.clip(pred, -0.5, 1.2)


def _evaluate_corrections(parts: list[pd.DataFrame], protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit corrections on one parity and score the other parity."""
    rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    specs = [
        ("loo_date_w10", ["date"], 3, 0.10),
        ("loo_date_w20", ["date"], 3, 0.20),
        ("loo_crop_doy_w10", ["crop_type", "doy_bin"], 20, 0.10),
        ("loo_crop_doy_w20", ["crop_type", "doy_bin"], 20, 0.20),
        ("loo_doy_w10", ["doy_bin"], 60, 0.10),
        ("loo_doy_w20", ["doy_bin"], 60, 0.20),
    ]
    for part in parts:
        part = part.reset_index(drop=True)
        cal = part[part.hash_parity == 0].copy(); test = part[part.hash_parity == 1].copy()
        if len(cal) < 30 or len(test) < 30:
            continue
        base = test["soft"].to_numpy(float)
        for name, keys, min_n, w in specs:
            pp = _apply_group_correction(cal, test, keys, w, min_n)
            rm, ma = _metric(test["_truth"].to_numpy(float), pp)
            rows.append({"protocol": protocol, "partition": part.partition.iloc[0], "method": name, "n": len(test), "rmse": rm, "mae": ma, "baseline_rmse": _metric(test["_truth"].to_numpy(float), base)[0]})
            q = test[["partition", "anon_polygon_id", "date", "_truth", "_true_src"]].copy(); q["method"] = name; q["pred"] = pp; pred_rows.append(q)
        for rank in (1, 2):
            for blend in (0.5, 1.0):
                name = f"pca_rank{rank}_b{blend:.1f}"
                pp = _pca_stack(cal, test, rank=rank, blend=blend)
                rm, ma = _metric(test["_truth"].to_numpy(float), pp)
                rows.append({"protocol": protocol, "partition": part.partition.iloc[0], "method": name, "n": len(test), "rmse": rm, "mae": ma, "baseline_rmse": _metric(test["_truth"].to_numpy(float), base)[0]})
                q = test[["partition", "anon_polygon_id", "date", "_truth", "_true_src"]].copy(); q["method"] = name; q["pred"] = pp; pred_rows.append(q)
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def _make_calibration_mask(frame: pd.DataFrame, eval_mask: np.ndarray, seed: int, frac: float = 0.10) -> tuple[pd.DataFrame, np.ndarray]:
    """Mask a separate observed calibration sample, never using eval truth.

    The returned frame retains the evaluation rows as they were (already
    hidden) and additionally hides a deterministic 10% sample of the rows
    whose targets are genuinely observed.  Predictions on this calibration
    sample therefore have a valid side-car truth while the outer evaluation
    rows remain completely out of the residual fit.
    """
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    existing = _flag(d.get("is_synthetic_gap", pd.Series(False, index=d.index)))
    known = np.isfinite(d["primary_ndvi"].to_numpy(float)) & ~existing
    rng = np.random.default_rng(int(seed))
    cal = np.zeros(len(d), dtype=bool)
    years = d["date"].dt.year
    for _, ix in d.loc[known].groupby(["anon_polygon_id", years], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        n = max(1, int(round(float(frac) * len(ii))))
        cal[rng.choice(ii, size=min(n, len(ii)), replace=False)] = True
    # Do not accidentally select an outer-evaluation row (it is already NaN,
    # but this explicit exclusion documents the leakage boundary).
    cal &= ~np.asarray(eval_mask, dtype=bool)
    for col in DYNAMIC:
        if col in d:
            d.loc[cal, col] = np.nan
    d.loc[cal, "is_synthetic_gap"] = True
    return d, cal


def _safe_correction_eval(
    items: list[dict[str, object]], protocol: str, train_default: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate residual/PCA corrections with *observed-only* calibration.

    ``items`` contains the outer query table, its masked frame, and the outer
    mask.  A fresh calibration mask is made from non-hidden rows and predicted
    with the same source-aware primitives.  No ``_truth`` value from outer
    query rows is used to fit a correction.
    """
    rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    specs = [
        ("safe_loo_date_w10", ["date"], 3, 0.10),
        ("safe_loo_date_w20", ["date"], 3, 0.20),
        ("safe_loo_crop_doy_w10", ["crop_type", "doy_bin"], 20, 0.10),
        ("safe_loo_crop_doy_w20", ["crop_type", "doy_bin"], 20, 0.20),
        ("safe_loo_doy_w10", ["doy_bin"], 60, 0.10),
        ("safe_loo_doy_w20", ["doy_bin"], 60, 0.20),
    ]
    for item_no, item in enumerate(items):
        q = item["q"].copy().reset_index(drop=True)
        frame = item["frame"].copy().reset_index(drop=True)
        eval_mask = np.asarray(item["mask"], dtype=bool)
        train_arg = item.get("train", train_default)
        # Different seeds ensure calibration draws do not line up with the
        # outer private-like mask.  The source/target sidecars are untouched.
        cal_frame, cal_mask = _make_calibration_mask(frame, eval_mask, seed=1701 + item_no)
        if not cal_mask.any():
            continue
        cp, _ = _predict_matrix(cal_frame, train_arg, family="base", k=8, degree=1, bin_days=30, date_weight=1.0)
        lp, _ = _predict_matrix(cal_frame, train_arg, family="lag", k=16, degree=3, bin_days=30, date_weight=1.0)
        cb = cp.rename(columns={c: f"base_{c}" for c in cp.columns if c != "row_index"})
        cl = lp.rename(columns={c: f"lag_{c}" for c in lp.columns if c != "row_index"})
        cz = cb.merge(cl, on="row_index", validate="one_to_one")
        ccal = _attach_query(cal_frame, cal_mask, cz.rename(columns={"base_soft": "soft", "base_hard": "hard", "base_route": "route", "base_p_s2": "p_s2", "base_p_landsat": "p_landsat", "base_p_modis": "p_modis", "base_pred_s2": "pred_s2", "base_pred_landsat": "pred_landsat", "base_pred_modis": "pred_modis"}), str(q["partition"].iloc[0]) + "_cal")
        # ``_attach_query`` aligns both families by row_index, so retain the
        # lag columns for PCA diversity without any positional assumptions.
        ccal["lag_soft"] = ccal["lag_soft"]
        ccal["lag_hard"] = ccal["lag_hard"]
        # q already carries the outer base/lag predictions and metadata.
        test = q.copy()
        base = test["soft"].to_numpy(float)
        for name, keys, min_n, w in specs:
            pp = _apply_group_correction(ccal, test, keys, w, min_n)
            rm, ma = _metric(test["_truth"].to_numpy(float), pp)
            rows.append({"protocol": protocol, "partition": str(test["partition"].iloc[0]), "method": name, "n": len(test), "cal_n": len(ccal), "rmse": rm, "mae": ma, "baseline_rmse": _metric(test["_truth"].to_numpy(float), base)[0], "fit_source": "observed_calibration_only"})
            z = test[["partition", "anon_polygon_id", "date", "_truth", "_true_src"]].copy(); z["method"] = name; z["pred"] = pp; pred_rows.append(z)
        for rank in (1, 2):
            for blend in (0.5, 1.0):
                name = f"safe_pca_rank{rank}_b{blend:.1f}"
                pp = _pca_stack(ccal, test, rank=rank, blend=blend)
                rm, ma = _metric(test["_truth"].to_numpy(float), pp)
                rows.append({"protocol": protocol, "partition": str(test["partition"].iloc[0]), "method": name, "n": len(test), "cal_n": len(ccal), "rmse": rm, "mae": ma, "baseline_rmse": _metric(test["_truth"].to_numpy(float), base)[0], "fit_source": "observed_calibration_only"})
                z = test[["partition", "anon_polygon_id", "date", "_truth", "_true_src"]].copy(); z["method"] = name; z["pred"] = pp; pred_rows.append(z)
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def _aggregate_route(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for (protocol, method, source), g in rows.groupby(["protocol", "method", "source"], dropna=False):
        if method == "route_accuracy":
            out.append({"protocol": protocol, "method": method, "source": source, "n": int(g.n.sum()), "metric": float(np.average(g.rmse, weights=g.n)), "mae": float(np.average(g.mae, weights=g.n))})
        else:
            out.append({"protocol": protocol, "method": method, "source": source, "n": int(g.n.sum()), "metric": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))), "mae": float(np.average(g.mae, weights=g.n))})
    return pd.DataFrame(out).sort_values(["protocol", "metric"])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="source-aware overnight evaluator")
    ap.add_argument("--years", nargs="*", type=int, default=[2019, 2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--skip-2025", action="store_true", help="skip the optional 2025 date-multiplicity proxy")
    args = ap.parse_args()
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)

    route_rows: list[dict[str, object]] = []
    correction_rows: list[pd.DataFrame] = []
    correction_pred_rows: list[pd.DataFrame] = []
    all_parts: dict[str, list[pd.DataFrame]] = {"exact_hidden_doy": [], "private_like": [], "private_2025_date": []}
    safe_parts: dict[str, list[dict[str, object]]] = {"exact_hidden_doy": [], "private_like": [], "private_2025_date": []}

    # Exact hidden-DOY projection onto train years.  Source-aware route and
    # correction predictions are generated with the fold itself as history.
    for year in args.years:
        fold, mask = _exact_fold(train, private, year)
        if not mask.any():
            continue
        pieces = []
        for family, k, degree in (("base", 8, 1), ("lag", 16, 3)):
            pred, _ = _predict_matrix(fold, None, family=family, k=k, degree=degree, bin_days=30, date_weight=1.0)
            name = "base" if family == "base" else "lag"
            z = pred.rename(columns={c: f"{name}_{c}" for c in pred.columns if c != "row_index"})
            pieces.append(z)
        zz = pieces[0].merge(pieces[1], on="row_index", validate="one_to_one")
        # Rename to a common table schema for the route scorer.
        q = _attach_query(fold, mask, zz.rename(columns={"base_soft": "soft", "base_hard": "hard", "base_route": "route", "base_p_s2": "p_s2", "base_p_landsat": "p_landsat", "base_p_modis": "p_modis", "base_pred_s2": "pred_s2", "base_pred_landsat": "pred_landsat", "base_pred_modis": "pred_modis"}), f"exact{year}")
        # Keep lag predictions as separate methods while preserving source
        # posterior from the base (same posterior by design).
        for c in ["lag_soft", "lag_hard", "lag_pred_s2", "lag_pred_landsat", "lag_pred_modis"]:
            q[c] = zz[c].to_numpy()
        q["protocol"] = "exact_hidden_doy"
        q["method_family"] = "base_lag"
        route_rows.extend(_route_rows(q, "exact_hidden_doy", "base_"))
        # Add lag rows under the same scorer names.
        qlag = q.copy(); qlag["soft"] = qlag["lag_soft"]; qlag["hard"] = qlag["lag_hard"]
        for s in SOURCES:
            qlag[f"pred_{s}"] = qlag[f"lag_pred_{s}"]
        route_rows.extend(_route_rows(qlag, "exact_hidden_doy", "lag_"))
        q["base_soft"] = q["soft"]; q["base_hard"] = q["hard"]
        q["lag_soft"] = qlag["soft"]; q["lag_hard"] = qlag["hard"]
        all_parts["exact_hidden_doy"].append(q)
        safe_parts["exact_hidden_doy"].append({"q": q, "frame": fold, "mask": mask, "train": None})

    # Random private-like masks and a 2025 date-multiplicity proxy.
    for seed in args.seeds:
        frame, mask = _mask_private_like(private, seed)
        pieces = []
        for family, k, degree in (("base", 8, 1), ("lag", 16, 3)):
            pred, _ = _predict_matrix(frame, train, family=family, k=k, degree=degree, bin_days=30, date_weight=1.0)
            name = "base" if family == "base" else "lag"
            pieces.append(pred.rename(columns={c: f"{name}_{c}" for c in pred.columns if c != "row_index"}))
        zz = pieces[0].merge(pieces[1], on="row_index", validate="one_to_one")
        q = _attach_query(frame, mask, zz.rename(columns={"base_soft": "soft", "base_hard": "hard", "base_route": "route", "base_p_s2": "p_s2", "base_p_landsat": "p_landsat", "base_p_modis": "p_modis", "base_pred_s2": "pred_s2", "base_pred_landsat": "pred_landsat", "base_pred_modis": "pred_modis"}), f"random{seed}")
        for c in ["lag_soft", "lag_hard", "lag_pred_s2", "lag_pred_landsat", "lag_pred_modis"]:
            q[c] = zz[c].to_numpy()
        q["protocol"] = "private_like"
        route_rows.extend(_route_rows(q, "private_like", "base_"))
        qlag = q.copy(); qlag["soft"] = qlag["lag_soft"]; qlag["hard"] = qlag["lag_hard"]
        for s in SOURCES:
            qlag[f"pred_{s}"] = qlag[f"lag_pred_{s}"]
        route_rows.extend(_route_rows(qlag, "private_like", "lag_"))
        all_parts["private_like"].append(q)
        safe_parts["private_like"].append({"q": q, "frame": frame, "mask": mask, "train": train})

        # Date-multiplicity 2025 proxy, useful as an additional stress slice.
        frame25, mask25 = _make_2025_date_proxy(private, seed)
        if mask25.any() and not args.skip_2025:
            pp = []
            for family, k, degree in (("base", 8, 1), ("lag", 16, 3)):
                pred, _ = _predict_matrix(frame25, train, family=family, k=k, degree=degree, bin_days=30, date_weight=1.0)
                name = "base" if family == "base" else "lag"
                pp.append(pred.rename(columns={c: f"{name}_{c}" for c in pred.columns if c != "row_index"}))
            z25 = pp[0].merge(pp[1], on="row_index", validate="one_to_one")
            q25 = _attach_query(frame25, mask25, z25.rename(columns={"base_soft": "soft", "base_hard": "hard", "base_route": "route", "base_p_s2": "p_s2", "base_p_landsat": "p_landsat", "base_p_modis": "p_modis", "base_pred_s2": "pred_s2", "base_pred_landsat": "pred_landsat", "base_pred_modis": "pred_modis"}), f"date25_{seed}")
            # ``_attach_query`` already aligned all lag columns by row_index;
            # no positional assignment is safe because real hidden rows are
            # also present in the prediction matrix.
            q25["protocol"] = "private_2025_date"
            route_rows.extend(_route_rows(q25, "private_2025_date", "base_"))
            q25l = q25.copy(); q25l["soft"] = q25l["lag_soft"]; q25l["hard"] = q25l["lag_hard"]
            for s in SOURCES:
                q25l[f"pred_{s}"] = q25l[f"lag_pred_{s}"]
            route_rows.extend(_route_rows(q25l, "private_2025_date", "lag_"))
            all_parts["private_2025_date"].append(q25)
            safe_parts["private_2025_date"].append({"q": q25, "frame": frame25, "mask": mask25, "train": train})

    route = pd.DataFrame(route_rows)
    route.to_csv(RESEARCH / "overnight_source_metrics.csv", index=False)
    route_agg = _aggregate_route(route)
    route_agg.to_csv(RESEARCH / "overnight_source_aggregate.csv", index=False)

    # Corrections/stacks are fit strictly from an *additional observed-only*
    # calibration mask.  The earlier parity split is retained below only as a
    # disposable diagnostic and is not used for candidate selection.
    safe_rows: list[pd.DataFrame] = []
    safe_pred_rows: list[pd.DataFrame] = []
    for protocol, items in safe_parts.items():
        if not items:
            continue
        cr, cp = _safe_correction_eval(items, protocol, train)
        if len(cr):
            safe_rows.append(cr)
        if len(cp):
            safe_pred_rows.append(cp)
    safe_corrections = pd.concat(safe_rows, ignore_index=True) if safe_rows else pd.DataFrame()
    safe_corrections.to_csv(RESEARCH / "overnight_correction_metrics.csv", index=False)
    if safe_pred_rows:
        pd.concat(safe_pred_rows, ignore_index=True).to_csv(RESEARCH / "overnight_correction_predictions.csv", index=False)
    # Keep parity diagnostics separate and explicitly non-deployable.
    diag_corrections = pd.concat(correction_rows, ignore_index=True) if correction_rows else pd.DataFrame()
    diag_corrections.to_csv(RESEARCH / "overnight_correction_diagnostic_metrics.csv", index=False)

    # Pick a candidate only if a correction/stack is consistently better than
    # the same-partition soft baseline on both primary protocols.  In the
    # expected negative case no candidate is materialized.
    candidate_info = "No overnight candidate materialized: no correction/stack passed the two-protocol stability gate."
    corrections = safe_corrections
    if len(corrections):
        agg = corrections.groupby(["protocol", "method"], as_index=False).apply(
            lambda g: pd.Series({"rmse": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))), "n": int(g.n.sum()), "baseline": float(np.sqrt(np.average(g.baseline_rmse ** 2, weights=g.n)))})
        )
        agg = agg.reset_index(drop=True)
        agg.to_csv(RESEARCH / "overnight_correction_aggregate.csv", index=False)
        # Require at least 0.0005 RMSE gain on both requested protocols.
        piv = agg.pivot(index="method", columns="protocol", values=["rmse", "baseline"])
        good = []
        for method in piv.index:
            try:
                ok1 = float(piv.loc[method, ("rmse", "exact_hidden_doy")]) <= float(piv.loc[method, ("baseline", "exact_hidden_doy")]) - 0.0005
                ok2 = float(piv.loc[method, ("rmse", "private_like")]) <= float(piv.loc[method, ("baseline", "private_like")]) - 0.0005
                if ok1 and ok2:
                    good.append(method)
            except Exception:
                pass
        if good:
            candidate_info = "Stable correction candidates: " + ", ".join(good)

    lines = [
        "# Overnight source-aware evaluator",
        "",
        "Research-only run; `outputs/model_dani_tuned*` and input CSVs were not modified.",
        "",
        "## Protocols",
        "",
        "- exact_hidden_doy: private synthetic DOYs projected onto train 2019--2024;",
        "- private_like: 15% random known private rows per AOI/year, seeds 0/1/2;",
        "- private_2025_date: optional 2025 date-multiplicity stress proxy.",
        "",
        "## Source routing",
        "",
        "`soft` is posterior-weighted source interpolation (production logic); `hard` selects the modal source; `oracle_source` is evaluation-only.",
        "",
        route_agg.head(40).to_string(index=False),
        "",
        "## Residual correction / low-rank diversity",
        "",
        "Corrections are fit only from an additional observed-only calibration mask (10% of genuinely observed rows) and scored on the outer hidden rows. Date/crop residuals use robust median, count shrinkage and a +/-0.03 cap. PCA stacks are rank 1/2 and cross-fitted. The superseded parity diagnostic is saved separately as `overnight_correction_diagnostic_metrics.csv` and is not used for selection.",
        "",
        corrections.head(80).to_string(index=False) if len(corrections) else "No correction rows.",
        "",
        candidate_info,
        "",
        "Artifacts: `overnight_source_metrics.csv`, `overnight_source_aggregate.csv`, `overnight_correction_metrics.csv`, `overnight_correction_aggregate.csv`, `overnight_source_eval.py`.",
    ]
    (RESEARCH / "overnight_source_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(route_agg.head(30).to_string(index=False))
    if len(corrections):
        print("CORRECTIONS")
        print(corrections.groupby("method").apply(lambda g: np.sqrt(np.average(g.rmse ** 2, weights=g.n))).sort_values().head(20).to_string())


if __name__ == "__main__":
    main()

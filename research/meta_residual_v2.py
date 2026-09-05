"""Leakage-safe residual/meta-model on the extwide40_v3_30 family.

The saved private-like holdout predictions are treated as fixed OOF features.
Context features are rebuilt from a frame in which both organiser gaps and the
holdout rows have all dynamic fields removed.  A residual learner is fit only
on other AOI (or AOI/year) groups in every outer split.  The script emits
diagnostics and, only when a rule survives all grouped/seed checks, a separate
research-only full-private candidate.  Existing production files are never
modified.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
OUT = ROOT / "outputs"

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
TRAIN_IDS: set[str] = set()

DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", TARGET, "doy", "ndvi_climatology_mean",
    "ndvi_climatology_std", "n_reference_years", "status", "ndvi_zscore",
]
SENSORS = ("s2", "landsat", "modis")


def _clear_dynamic(d: pd.DataFrame, exclude: np.ndarray) -> pd.DataFrame:
    z = d.copy().reset_index(drop=True)
    m = np.asarray(exclude, bool)
    for c in DYNAMIC:
        if c in z.columns:
            z.loc[m, c] = np.nan
    return z


def _source_arrays(d: pd.DataFrame, known: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source indicators; values from excluded rows are never used."""
    n = len(d)
    ind = []
    for s in SENSORS:
        col = f"{s}_ndvi"
        ind.append(d[col].notna().to_numpy(bool) if col in d else np.zeros(n, bool))
    # Primary source priority mirrors the competition construction.
    src = np.select(ind, [0, 1, 2], default=-1).astype(int)
    src[~known] = -1
    return np.asarray(ind[0] & known), np.asarray(ind[1] & known), np.asarray(ind[2] & known)


def _nearest_stats(ord_known: np.ndarray, y_known: np.ndarray, cm_known: np.ndarray,
                   q_ord: float) -> dict[str, float]:
    if len(ord_known) == 0:
        return {}
    order = np.argsort(ord_known)
    xx = np.asarray(ord_known, float)[order]
    yy = np.asarray(y_known, float)[order]
    cc = np.asarray(cm_known, float)[order]
    pos = int(np.searchsorted(xx, q_ord, side="left"))
    # There should be no exact query date among known rows, but handle it
    # defensively to avoid treating a duplicate as a future value.
    li = pos - 1
    ri = pos
    if ri < len(xx) and abs(xx[ri] - q_ord) < 1e-9:
        ri += 1
    prev_d = q_ord - xx[li] if li >= 0 else np.nan
    next_d = xx[ri] - q_ord if ri < len(xx) else np.nan
    prev_y = yy[li] if li >= 0 else np.nan
    next_y = yy[ri] if ri < len(xx) else np.nan
    if np.isfinite(prev_y) and np.isfinite(next_y) and (prev_d + next_d) > 0:
        interp = (prev_y * next_d + next_y * prev_d) / (prev_d + next_d)
        slope = (next_y - prev_y) / (prev_d + next_d)
    elif np.isfinite(prev_y):
        interp, slope = prev_y, 0.0
    elif np.isfinite(next_y):
        interp, slope = next_y, 0.0
    else:
        interp, slope = np.nan, np.nan
    out = {
        "prev_y": float(prev_y), "next_y": float(next_y),
        "prev_d": float(prev_d), "next_d": float(next_d),
        "span": float(prev_d + next_d) if np.isfinite(prev_d) and np.isfinite(next_d) else float(np.nanmin([prev_d, next_d])),
        "interp": float(interp), "slope": float(slope),
    }
    for window in (7, 14, 30, 60, 120):
        take = np.abs(xx - q_ord) <= window
        vals = yy[take & np.isfinite(yy)]
        cms = cc[take & np.isfinite(cc)]
        out[f"local_mean_{window}"] = float(np.mean(vals)) if len(vals) else np.nan
        out[f"local_median_{window}"] = float(np.median(vals)) if len(vals) else np.nan
        out[f"local_sd_{window}"] = float(np.std(vals)) if len(vals) > 1 else (0.0 if len(vals) else np.nan)
        out[f"local_n_{window}"] = float(len(vals))
        out[f"clim_mean_{window}"] = float(np.mean(cms)) if len(cms) else np.nan
    # A robust long-range same-AOI climatology summary helps rows whose local
    # bracket is one-sided.  It uses only visible rows.
    take = np.abs(xx - q_ord) <= 180
    cms = cc[take & np.isfinite(cc)]
    out["clim_local"] = float(np.median(cms)) if len(cms) else np.nan
    return out


def _peer_stats(known_frame: pd.DataFrame, qid: str, qdate: pd.Timestamp,
                qcrop: str) -> dict[str, float]:
    # Called only for query dates; exact-date peers are observable values from
    # other AOIs.  Excluding the target AOI prevents self-label leakage.
    same = known_frame[known_frame[DATE].eq(qdate) & known_frame[ID].astype(str).ne(str(qid))]
    vals = pd.to_numeric(same[TARGET], errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    crop_same = same[same.get("crop_type", pd.Series("", index=same.index)).fillna("").astype(str).eq(str(qcrop))]
    cvals = pd.to_numeric(crop_same[TARGET], errors="coerce").to_numpy(float)
    cvals = cvals[np.isfinite(cvals)]
    out = {
        "peer_median": float(np.median(vals)) if len(vals) else np.nan,
        "peer_mean": float(np.mean(vals)) if len(vals) else np.nan,
        "peer_sd": float(np.std(vals)) if len(vals) > 1 else (0.0 if len(vals) else np.nan),
        "peer_n": float(len(vals)),
        "crop_peer_median": float(np.median(cvals)) if len(cvals) else np.nan,
        "crop_peer_mean": float(np.mean(cvals)) if len(cvals) else np.nan,
        "crop_peer_n": float(len(cvals)),
    }
    # Date-level visible weather is target-independent context.  It can be
    # useful when the target AOI itself has no local bracket.
    for col, name in (("era5_temp_c", "date_temp"), ("era5_precip_mm", "date_precip")):
        if col in same:
            vv = pd.to_numeric(same[col], errors="coerce").to_numpy(float)
            vv = vv[np.isfinite(vv)]
            out[name] = float(np.median(vv)) if len(vv) else np.nan
    return out


def _source_posteriors(d: pd.DataFrame, known: np.ndarray, q: pd.DataFrame) -> pd.DataFrame:
    n = len(d)
    dates = pd.to_datetime(d[DATE])
    ids = d[ID].astype(str).to_numpy()
    doys = dates.dt.dayofyear.to_numpy(int)
    s2, ls, md = _source_arrays(d, known)
    # Source availability, rather than source labels from hidden rows, is the
    # only information available at inference time.
    src = np.select([s2, ls, md], [0, 1, 2], default=-1).astype(int)
    rows = []
    # Precompute global DOY counts as fallback.
    glob: dict[int, np.ndarray] = {}
    for doy, ix0 in pd.DataFrame({"doy": doys}).loc[known].groupby("doy", sort=False).groups.items():
        ix = np.asarray(ix0, int); v = src[ix]; v = v[v >= 0]
        glob[int(doy)] = np.bincount(v, minlength=3).astype(float)
    # AOI/DOY schedule often repeats over years; group only visible rows.
    loc: dict[tuple[str, int], np.ndarray] = {}
    tab = pd.DataFrame({"id": ids, "doy": doys})
    for key, ix0 in tab.loc[known].groupby(["id", "doy"], sort=False).groups.items():
        ix = np.asarray(ix0, int); v = src[ix]; v = v[v >= 0]
        loc[(str(key[0]), int(key[1]))] = np.bincount(v, minlength=3).astype(float)
    for _, row in q.iterrows():
        key = (str(row[ID]), int(pd.Timestamp(row[DATE]).dayofyear))
        cnt = loc.get(key)
        if cnt is None or cnt.sum() < 2:
            cnt = glob.get(key[1], np.zeros(3))
        den = float(cnt.sum())
        prob = cnt / den if den > 0 else np.full(3, 1 / 3)
        ent = float(-np.sum(np.where(prob > 0, prob * np.log(prob), 0.0)))
        rows.append({"source_p_s2": prob[0], "source_p_ls": prob[1], "source_p_md": prob[2],
                     "source_n": den, "source_entropy": ent, "source_mode": float(np.argmax(prob))})
    return pd.DataFrame(rows, index=q.index)


def context_features(frame: pd.DataFrame, exclude: np.ndarray, query: np.ndarray) -> pd.DataFrame:
    """Build visibility-safe context for query rows."""
    d = _clear_dynamic(frame, exclude)
    d[DATE] = pd.to_datetime(d[DATE])
    m = np.asarray(exclude, bool); qmask = np.asarray(query, bool)
    known = d[TARGET].notna().to_numpy(bool) & ~m
    qidx = np.flatnonzero(qmask)
    dates = d[DATE]
    ords = dates.map(pd.Timestamp.toordinal).to_numpy(float)
    years = dates.dt.year.to_numpy(int)
    doys = dates.dt.dayofyear.to_numpy(int)
    ids = d[ID].astype(str).to_numpy()
    crops = d.get("crop_type", pd.Series("", index=d.index)).fillna("").astype(str).to_numpy()
    known_frame = d.loc[known].copy()
    # Group known rows once, then query each group.  This is linear in query
    # count and avoids any target access from excluded rows.
    groups: dict[tuple[str, int], np.ndarray] = {}
    tab = pd.DataFrame({"id": ids, "year": years})
    for key, ix0 in tab.loc[known].groupby(["id", "year"], sort=False).groups.items():
        groups[(str(key[0]), int(key[1]))] = np.asarray(ix0, int)

    recs: list[dict[str, object]] = []
    cm = pd.to_numeric(d.get("ndvi_climatology_mean", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    y = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
    # Build per-query row.  The sidecar truth is never present in ``d``.
    for i in qidx:
        rec: dict[str, object] = {ID: ids[i], DATE: dates.iat[i]}
        rec["year"] = float(years[i]); rec["doy"] = float(doys[i]); rec["aoi_num"] = float(_aoi_num(ids[i]))
        rec["is_2025"] = float(years[i] == 2025); rec["is_shared"] = float(ids[i] in TRAIN_IDS)
        rec["sin1"] = np.sin(2 * np.pi * doys[i] / 365.25); rec["cos1"] = np.cos(2 * np.pi * doys[i] / 365.25)
        rec["sin2"] = np.sin(4 * np.pi * doys[i] / 365.25); rec["cos2"] = np.cos(4 * np.pi * doys[i] / 365.25)
        rec["crop_code"] = float(_crop_code(crops[i]))
        ix = groups.get((ids[i], int(years[i])), np.empty(0, int))
        good = ix[np.isfinite(y[ix])] if len(ix) else ix
        stats = _nearest_stats(ords[good], y[good], cm[good], ords[i]) if len(good) else {}
        rec.update(stats)
        rec.update(_peer_stats(known_frame, ids[i], dates.iat[i], crops[i]))
        # Visible sensor/date counts (all use known rows only).
        day_rows = known_frame[known_frame[DATE].eq(dates.iat[i])]
        rec["date_known_n"] = float(len(day_rows))
        for s in SENSORS:
            col = f"{s}_ndvi"
            rec[f"date_{s}_n"] = float(day_rows[col].notna().sum()) if col in day_rows else 0.0
        recs.append(rec)
    out = pd.DataFrame(recs)
    if len(out) != len(qidx):
        raise RuntimeError("context/query alignment failure")
    sp = _source_posteriors(d, known, out)
    out = pd.concat([out.reset_index(drop=True), sp.reset_index(drop=True)], axis=1)
    # Ensure all numeric context values are finite-or-NaN and no accidental
    # object columns enter the estimator.
    for c in out.columns:
        if c not in (ID, DATE):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _aoi_num(s: str) -> int:
    try:
        return int(str(s).split("-")[-1])
    except Exception:
        return abs(hash(str(s))) % 10000


_CROP_MAP: dict[str, int] = {}


def _crop_code(s: str) -> int:
    s = str(s)
    if s not in _CROP_MAP:
        _CROP_MAP[s] = len(_CROP_MAP) + 1
    return _CROP_MAP[s]


def _join_holdout_features(pr: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = _make_holdout(pr, 70404)
    hidden = pr[GAP].fillna(False).astype(bool).to_numpy()
    exclude = hidden | hold
    ctx = context_features(pr, exclude, hold)
    pred = pd.read_csv(R / "spectral_private_holdout_predictions.csv", parse_dates=[DATE], low_memory=False)
    # Keep candidate columns that are available in both holdout and full
    # private inference.  ``base`` is the requested extwide40_v3_30 anchor.
    keep = [ID, DATE, "truth", "cohort", "year", "ext40_v3_30", "spectral", "ext40_v3_40",
            "ext40", "v3", "blend_30", "joint_blend_30"]
    keep = [c for c in keep if c in pred.columns]
    q = pred[keep].copy(); q[DATE] = pd.to_datetime(q[DATE])
    # ``year``/``doy`` are already present in the saved prediction table;
    # retain that observable metadata and avoid pandas' ``_x/_y`` suffixes.
    ctx_join = ctx.drop(columns=[c for c in ("year", "doy") if c in ctx.columns])
    q = q.merge(ctx_join, on=[ID, DATE], how="left", validate="one_to_one")
    q["base"] = q["ext40_v3_30"].astype(float)
    # Candidate disagreement features are observable model outputs, not labels.
    for c in ["spectral", "ext40_v3_40", "ext40", "v3", "blend_30", "joint_blend_30"]:
        if c in q:
            q[f"diff_{c}"] = q[c].astype(float) - q["base"]
    q["resid_target"] = q["truth"].astype(float) - q["base"]
    q["group_aoi"] = q[ID].astype(str)
    q["group_aoiyear"] = q[ID].astype(str) + "_" + q["year"].astype(str)
    # Defensive check: query-side context was generated with target masked.
    if len(q) != int(hold.sum()) or q["truth"].isna().any():
        raise RuntimeError("holdout feature alignment failure")
    return q, pd.DataFrame({"hold": hold, "exclude": exclude})


def _make_holdout(pr: pd.DataFrame, seed: int = 70404) -> np.ndarray:
    known = pr[TARGET].notna().to_numpy(bool) & ~pr[GAP].fillna(False).to_numpy(bool)
    out = np.zeros(len(pr), bool); rng = np.random.default_rng(int(seed))
    yy = pd.to_datetime(pr[DATE]).dt.year
    for _, ix0 in pr.loc[known].groupby([ID, yy], sort=False).groups.items():
        ix = np.asarray(ix0, int); n = max(1, int(round(.15 * len(ix))))
        out[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
    return out


def _model(kind: str):
    if kind.startswith("ridge"):
        alpha = float(kind.replace("ridge", ""))
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))
    if kind.startswith("hgb"):
        leaf = int(kind.replace("hgb", ""))
        return HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=.025, max_iter=90,
            max_leaf_nodes=leaf, min_samples_leaf=70, l2_regularization=20.0,
            random_state=42,
        )
    raise ValueError(kind)


def _metric(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def _evaluate(q: pd.DataFrame, features: list[str], group_col: str, seed: int, model_kind: str,
              caps: tuple[float, ...] = (.02, .04, .06, .08)) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = q[features].to_numpy(float); y = q["resid_target"].to_numpy(float)
    base = q["base"].to_numpy(float); truth = q["truth"].to_numpy(float)
    groups = q[group_col].astype(str).to_numpy()
    # GroupShuffleSplit with a fixed seed gives independent outer partitions;
    # no AOI occurs in both fit and score for any split.
    splitter = GroupShuffleSplit(n_splits=3, test_size=.20, random_state=int(seed))
    rows: list[dict[str, object]] = []; pred_parts: list[pd.DataFrame] = []
    for split_no, (tri, tei) in enumerate(splitter.split(X, y, groups), 1):
        m = _model(model_kind)
        m.fit(X[tri], y[tri])
        raw_corr = np.asarray(m.predict(X[tei]), float)
        b_rm = _metric(truth[tei], base[tei])
        for cap in caps:
            corr = np.clip(raw_corr, -float(cap), float(cap))
            pred = np.clip(base[tei] + corr, -.5, 1.2)
            rm = _metric(truth[tei], pred)
            rows.append({"group": group_col, "seed": seed, "split": split_no, "model": model_kind, "cap": cap,
                         "n": len(tei), "rmse": rm, "baseline_rmse": b_rm, "delta_rmse": rm - b_rm,
                         "improved": int(rm < b_rm), "train_n": len(tri)})
            pp = q.iloc[tei][[ID, DATE, "truth", "base", "cohort", "year"]].copy()
            pp["pred"] = pred; pp["correction"] = corr; pp["group"] = group_col; pp["seed"] = seed; pp["split"] = split_no; pp["model"] = model_kind; pp["cap"] = cap
            pred_parts.append(pp)
    return pd.DataFrame(rows), pd.concat(pred_parts, ignore_index=True)


def _full_private_features(pr: pd.DataFrame, hold_meta: pd.DataFrame) -> pd.DataFrame:
    hidden = pr[GAP].fillna(False).astype(bool).to_numpy()
    ctx = context_features(pr, hidden, hidden)
    # Build full-private candidate model outputs from stable artifacts.
    q = pr.loc[hidden, [ID, DATE]].copy().reset_index(drop=True)
    q[DATE] = pd.to_datetime(q[DATE])
    def add(path: Path, name: str):
        p = pd.read_csv(path, parse_dates=[DATE]); p[DATE] = pd.to_datetime(p[DATE])
        z = q.merge(p[[ID, DATE, "primary_ndvi_pred"]], on=[ID, DATE], how="left", validate="one_to_one")
        q[name] = z["primary_ndvi_pred"].to_numpy(float)
    add(OUT / "model_dani_lag40_peer10_extwide40_v3_30_submission.csv", "ext40_v3_30")
    add(OUT / "model_dani_lag40_peer10_extwide40_submission.csv", "ext40")
    add(OUT / "model_dani_extended_hgb_v3_wide.csv", "v3")
    sp = pd.read_csv(R / "spectral_full_predictions_checkpoint.csv", parse_dates=[DATE]); sp[DATE] = pd.to_datetime(sp[DATE])
    q = q.merge(sp[[ID, DATE, "spectral_pred"]].rename(columns={"spectral_pred": "spectral"}), on=[ID, DATE], how="left", validate="one_to_one")
    q["ext40_v3_40"] = .6 * q.ext40 + .4 * q.v3
    q["blend_30"] = .7 * q.ext40_v3_30 + .3 * q.spectral
    q["joint_blend_30"] = q["blend_30"]
    q["base"] = q.ext40_v3_30
    for c in ["spectral", "ext40_v3_40", "ext40", "v3", "blend_30", "joint_blend_30"]:
        q[f"diff_{c}"] = q[c] - q.base
    q = q.merge(ctx, on=[ID, DATE], how="left", validate="one_to_one")
    q["group_aoi"] = q[ID].astype(str); q["group_aoiyear"] = q[ID].astype(str) + "_" + q[DATE].dt.year.astype(str)
    return q


def main() -> None:
    global TRAIN_IDS
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    TRAIN_IDS = set(train[ID].astype(str))
    q, _ = _join_holdout_features(private, train)
    candidate_cols = ["base", "spectral", "ext40_v3_40", "ext40", "v3", "blend_30", "joint_blend_30"]
    context_cols = [c for c in q.columns if c not in {ID, DATE, "truth", "cohort", "resid_target", "group_aoi", "group_aoiyear"} and c not in candidate_cols]
    features = [c for c in candidate_cols + context_cols if c in q.columns]
    # Remove constant/all-missing columns before fitting.
    features = [c for c in features if q[c].notna().any() and q[c].nunique(dropna=True) > 1]
    print("holdout", len(q), "features", len(features), flush=True)
    all_rows: list[pd.DataFrame] = []; all_preds: list[pd.DataFrame] = []
    for group_col in ("group_aoi", "group_aoiyear"):
        for seed in (0, 1, 2):
            for kind in ("ridge30", "ridge100", "hgb8", "hgb16"):
                met, pred = _evaluate(q, features, group_col, seed, kind)
                all_rows.append(met); all_preds.append(pred)
    metrics = pd.concat(all_rows, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    metrics.to_csv(R / "meta_residual_v2_metrics.csv", index=False)
    preds.to_csv(R / "meta_residual_v2_predictions.csv", index=False)
    # Baseline and cohort summaries, pooled over all outer splits/seeds.
    agg = metrics.groupby(["group", "model", "cap"], as_index=False).apply(
        lambda g: pd.Series({"runs": len(g), "n": int(g.n.sum()),
                             "rmse": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))),
                             "baseline_rmse": float(np.sqrt(np.average(g.baseline_rmse ** 2, weights=g.n))),
                             "delta_rmse": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n)) - np.sqrt(np.average(g.baseline_rmse ** 2, weights=g.n))),
                             "improved_runs": int(g.improved.sum())}), include_groups=False
    ).reset_index(drop=True)
    agg.to_csv(R / "meta_residual_v2_aggregate.csv", index=False)
    # Cohort metrics for the strongest candidate rows.  This uses predictions
    # generated in the outer splits, never in-sample corrections.
    cohort_rows = []
    for (grp, kind, cap), g in preds.groupby(["group", "model", "cap"], sort=False):
        for cohort, h in [("all", g), ("history", g[g.year < 2025]), ("2025", g[g.year == 2025]),
                          ("new", g[g.cohort == "new"]), ("shared", g[g.cohort == "shared"]),
                          ("new_history", g[(g.cohort == "new") & (g.year < 2025)]),
                          ("new_2025", g[(g.cohort == "new") & (g.year == 2025)]),
                          ("shared_2025", g[(g.cohort == "shared") & (g.year == 2025)])]:
            if len(h) == 0: continue
            cohort_rows.append({"group": grp, "model": kind, "cap": cap, "cohort": cohort,
                                "n": len(h), "rmse": _metric(h.truth.to_numpy(float), h.pred.to_numpy(float)),
                                "baseline_rmse": _metric(h.truth.to_numpy(float), h.base.to_numpy(float))})
    cohorts = pd.DataFrame(cohort_rows); cohorts["delta_rmse"] = cohorts.rmse - cohorts.baseline_rmse
    cohorts.to_csv(R / "meta_residual_v2_cohorts.csv", index=False)

    # Select a conservative rule only if it improves every group/seed pooled
    # audit and has no catastrophic cohort regression.  Prefer ridge for a
    # stable extrapolation to unseen AOIs; HGB is retained for diagnostics.
    eligible = agg[(agg.group == "group_aoi") & (agg.model.str.startswith("ridge")) & (agg.improved_runs >= 14)].copy()
    if not eligible.empty:
        eligible = eligible.sort_values(["delta_rmse", "cap"])
        best = eligible.iloc[0]
    else:
        best = None
    decision = {"promoted": False, "reason": "no model improved all grouped seeds", "best": None}
    if best is not None:
        bc = cohorts[(cohorts.group == best.group) & (cohorts.model == best.model) & (cohorts.cap == best.cap)]
        # Require non-positive delta in broad all/history/2025 aggregates.
        broad = bc[bc.cohort.isin(["all", "history", "2025"])]
        if len(broad) and (broad.delta_rmse <= 0.0001).all():
            decision = {"promoted": True, "reason": "stable grouped/seed gain", "best": best.to_dict()}
            # Fit on all holdout rows and apply to actual private query rows.
            full = _full_private_features(private, q)
            full_features = [c for c in features if c in full.columns]
            model = _model(str(best.model)); model.fit(q[full_features].to_numpy(float), q.resid_target.to_numpy(float))
            corr = np.clip(np.asarray(model.predict(full[full_features]), float), -float(best.cap), float(best.cap))
            final = np.clip(full.base.to_numpy(float) + corr, -.5, 1.2)
            out = full[[ID, DATE]].copy(); out["primary_ndvi_pred"] = final
            out.to_csv(R / "meta_residual_v2_private_submission.csv", index=False, float_format="%.8f")
            decision["private_rows"] = len(out); decision["private_min"] = float(final.min()); decision["private_max"] = float(final.max())
    (R / "meta_residual_v2_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    lines = ["# Meta residual v2", "", "Research-only residual learner over `ext40_v3_30`; context is rebuilt after masking organiser + holdout gaps.", "", f"Features ({len(features)}): {', '.join(features)}", "", "## Pooled grouped/seed metrics", "", agg.sort_values("delta_rmse").head(30).to_string(index=False), "", "## Decision", "", json.dumps(decision, ensure_ascii=False, indent=2, default=float), "", "No production baseline was overwritten."]
    (R / "meta_residual_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(agg.sort_values("delta_rmse").head(30).to_string(index=False), flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

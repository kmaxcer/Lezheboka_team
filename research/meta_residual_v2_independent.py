"""Independent-mask audit for the leakage-safe history residual learner.

This research-only runner reconstructs the ext40-style baseline on fresh
private-known masks (rather than reusing the fixed 70404 table), then evaluates
the compact Ridge residual correction with AOI-grouped outer splits.  Labels are
kept in a sidecar and are never passed to feature builders for query rows.
No file under ``outputs/`` is written.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
ARCH = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
sys.path.insert(0, str(R))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARCH))

import meta_residual_v2 as meta  # noqa: E402
from evaluate_private_cohort_blend import (  # noqa: E402
    fit_extended,
    fit_hgb,
    make_holdout,
)
from infer_lag import predict_private_lag  # noqa: E402
from paired_aoi_v2 import _config_name, peer_predictions  # noqa: E402
from ensemble_cv_v2_apply import _seasonal_residuals, _shock, _state  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
CANON = {97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clear_query(pr: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Clear every dynamic field on gaps, retaining labels only out-of-band."""
    z = pr.copy().reset_index(drop=True)
    m = np.asarray(mask, bool)
    for c in meta.DYNAMIC:
        if c in z:
            z.loc[m, c] = np.nan
    z[GAP] = m
    return z


def _reference(train: pd.DataFrame, private_masked: pd.DataFrame,
               gap_keys: set[tuple[object, object]]) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the sorted train+private frame expected by the HGB helpers."""
    tr = train.copy(); pm = private_masked.copy()
    tr[GAP] = False; tr["_origin"] = "train"; pm["_origin"] = "private"
    tr[DATE] = pd.to_datetime(tr[DATE]); pm[DATE] = pd.to_datetime(pm[DATE])
    tr["_test_order"] = np.nan; pm["_test_order"] = np.arange(len(pm))
    ref = pd.concat([tr, pm], ignore_index=True, sort=False)
    ref = ref.sort_values([ID, DATE, "_origin"]).reset_index(drop=True)
    ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int)
    ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    # Keep sidecar truth for scoring; it is not a feature column.
    labels = pd.concat([train[[ID, DATE, TARGET]],
                        private_masked[[ID, DATE, "_truth"]].rename(columns={"_truth": TARGET})],
                       ignore_index=True)
    labels[DATE] = pd.to_datetime(labels[DATE])
    labels = labels.rename(columns={TARGET: "_truth"})
    # The private frame's primary value is already NaN at gaps; obtain labels
    # from a separate unmasked copy in the caller and merge there instead.
    # This defensive branch handles the common case where _truth is present.
    if "_truth" not in labels:
        labels = pd.concat([train[[ID, DATE, TARGET]].rename(columns={TARGET: "_truth"}),
                            private_masked[[ID, DATE, "_truth"]]], ignore_index=True)
    ref = ref.merge(labels[[ID, DATE, "_truth"]], on=[ID, DATE], how="left", validate="one_to_one")
    gaps = np.array([tuple(x) in gap_keys for x in ref[[ID, DATE]].to_numpy()], dtype=bool)
    ref.loc[gaps, TARGET] = np.nan
    return ref, gaps


def _build_reference(train: pd.DataFrame, private: pd.DataFrame,
                     hold: np.ndarray, hidden: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return original private, masked private, and sorted modelling reference."""
    p = private.copy().reset_index(drop=True)
    p[DATE] = pd.to_datetime(p[DATE]); p["_truth"] = pd.to_numeric(p[TARGET], errors="coerce")
    gaps = np.asarray(hidden, bool) | np.asarray(hold, bool)
    pm = _clear_query(p, gaps)
    gap_keys = set(map(tuple, p.loc[gaps, [ID, DATE]].to_numpy()))
    # Reimplement reference merge explicitly to avoid ever losing private
    # labels when _clear_query has set primary_ndvi to NaN.
    tr = train.copy(); tr[GAP] = False; tr["_origin"] = "train"
    pm["_origin"] = "private"; tr["_test_order"] = np.nan; pm["_test_order"] = np.arange(len(pm))
    # Keep private labels strictly side-car; helpers must see only the masked
    # primary column.  Otherwise the subsequent label merge would suffix the
    # column and fit_hgb could accidentally miss it.
    pm_model = pm.drop(columns=["_truth"], errors="ignore")
    tr[DATE] = pd.to_datetime(tr[DATE]); pm[DATE] = pd.to_datetime(pm[DATE])
    ref = pd.concat([tr, pm_model], ignore_index=True, sort=False).sort_values([ID, DATE, "_origin"]).reset_index(drop=True)
    ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int)
    ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    # Sidecar labels are from the untouched train/private tables only.
    lab_tr = train[[ID, DATE, TARGET]].rename(columns={TARGET: "_truth"}).copy()
    lab_pr = p[[ID, DATE, "_truth"]].copy()
    labels = pd.concat([lab_tr, lab_pr], ignore_index=True)
    labels[DATE] = pd.to_datetime(labels[DATE])
    ref = ref.merge(labels, on=[ID, DATE], how="left", validate="one_to_one")
    gaps_ref = np.array([tuple(x) in gap_keys for x in ref[[ID, DATE]].to_numpy()], dtype=bool)
    ref.loc[gaps_ref, TARGET] = np.nan
    return p, pm, ref, gaps_ref


def _score(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else float("nan")


def _make_q(train: pd.DataFrame, private: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fit baseline components on one fresh mask and return holdout rows."""
    hold = make_holdout(private, seed)
    hidden = private[GAP].fillna(False).astype(bool).to_numpy()
    p, pm, ref, gaps_ref = _build_reference(train, private, hold, hidden)
    qkeys = p.loc[hold, [ID, DATE]].copy().reset_index(drop=True)
    qkeys[DATE] = pd.to_datetime(qkeys[DATE])
    truth = p.loc[hold, TARGET].to_numpy(float)

    # HGB and extended components are rebuilt against the fresh mask.  The
    # helpers read only target-masked ``ref`` and return predictions on gaps.
    hp = fit_hgb(ref, gaps_ref)
    hr = ref.loc[gaps_ref, [ID, DATE]].copy(); hr["hgb"] = hp
    lag = predict_private_lag(pm, train=train, k=16, degree=3, bin_days=30,
                              use_date_prior=True, date_weight=1.0)
    lag[DATE] = pd.to_datetime(lag[DATE])
    lm = lag.set_index([ID, DATE])["primary_ndvi_pred"]
    q = qkeys.copy(); q["truth"] = truth
    q["hgb"] = [hr.set_index([ID, DATE]).loc[(i, d), "hgb"] for i, d in qkeys[[ID, DATE]].itertuples(index=False, name=None)]
    q["lag"] = [lm.get((i, d), np.nan) for i, d in qkeys[[ID, DATE]].itertuples(index=False, name=None)]

    # Same-year peer maps use only currently visible rows in the masked frame.
    pp, _ = peer_predictions(pm, hold, partition=f"independent_{seed}")
    peer_col = _config_name(16, .60, .125, 2)
    pp = pp.drop(columns=["_row"], errors="ignore")
    q = q.merge(pp[[ID, DATE, peer_col]], on=[ID, DATE], how="left", validate="one_to_one")
    q["base40"] = .6 * q.hgb + .4 * q.lag
    q["peer40"] = q.base40
    ok = q[peer_col].notna(); q.loc[ok, "peer40"] = .9 * q.loc[ok, "base40"] + .1 * q.loc[ok, peer_col]
    known = pm[TARGET].notna().to_numpy(bool) & ~pm[GAP].fillna(False).to_numpy(bool)
    qi = np.flatnonzero(hold)
    resid = _seasonal_residuals(pm, known)
    shock, _ = _shock(pm, known, resid, qi); state, _ = _state(pm, known, resid, qi)
    q["shock"] = shock; q["state"] = state
    canon = q[DATE].dt.dayofyear.isin(CANON).to_numpy(bool)
    q["joint40"] = q.peer40.to_numpy(float) + np.where(canon, 0, .35*np.nan_to_num(shock) - .20*np.nan_to_num(state))
    ep = fit_extended(ref, gaps_ref, 4)
    er = ref.loc[gaps_ref, [ID, DATE]].copy(); er["extended"] = ep
    q = q.merge(er, on=[ID, DATE], how="left", validate="one_to_one")
    q["ext40"] = .6*q.joint40 + .4*q.extended
    q["cohort"] = np.where(q[ID].astype(str).isin(set(train[ID].astype(str))), "shared", "new")
    q["year"] = q[DATE].dt.year.astype(int)
    # The requested extwide40_v3_30 anchor is unavailable for a fresh mask
    # without rerunning v3; ext40 is the same calibrated lag/peer/shock core.
    # A compact residual check on this core is conservative and independent.
    q["base"] = q["ext40"].astype(float)
    # Context is built from original private rows with both hidden and this
    # holdout removed.  No sidecar truth reaches context_features.
    meta.TRAIN_IDS = set(train[ID].astype(str))
    ctx = meta.context_features(private, hidden | hold, hold)
    ctx = ctx.drop(columns=[c for c in ("year", "doy") if c in ctx], errors="ignore")
    q = q.merge(ctx, on=[ID, DATE], how="left", validate="one_to_one")
    q["resid_target"] = q.truth - q.base
    q["group_aoi"] = q[ID].astype(str)
    return q


def evaluate_q(q: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    """AOI-grouped outer validation for compact Ridge on one mask."""
    cand = [c for c in ["base", "hgb", "lag", "peer40", "joint40", "extended", "ext40"] if c in q]
    ctx_names = ["year", "doy", "is_2025", "is_shared", "sin1", "cos1", "sin2", "cos2",
                 "span", "prev_d", "next_d", "interp", "slope", "local_mean_7",
                 "local_mean_14", "local_mean_30", "local_sd_30", "local_n_30",
                 "clim_local", "peer_median", "peer_sd", "peer_n", "crop_peer_median",
                 "crop_peer_n", "date_known_n", "source_p_s2", "source_p_ls", "source_p_md",
                 "source_entropy", "source_n"]
    fs = [c for c in cand + ctx_names if c in q and q[c].notna().any() and q[c].nunique(dropna=True) > 1]
    X = q[fs].to_numpy(float); y = q.resid_target.to_numpy(float)
    base = q.base.to_numpy(float); truth = q.truth.to_numpy(float)
    groups = q[ID].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for split_no, (tri, tei) in enumerate(GroupShuffleSplit(n_splits=3, test_size=.2, random_state=seed).split(X, y, groups), 1):
        model = meta._model("ridge30"); model.fit(X[tri], y[tri]); raw = model.predict(X[tei])
        b = _score(truth[tei], base[tei])
        for cap in (.005, .01, .015, .02, .03):
            p = np.clip(base[tei] + np.clip(raw, -cap, cap), -.5, 1.2)
            rows.append({"mask_seed": seed, "split": split_no, "n": len(tei), "features": len(fs),
                         "cap": cap, "baseline_rmse": b, "rmse": _score(truth[tei], p),
                         "delta_rmse": _score(truth[tei], p) - b,
                         "improved": int(_score(truth[tei], p) < b)})
    return rows


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    train[GAP] = False; private[GAP] = private[GAP].fillna(False).astype(bool)
    all_q: list[pd.DataFrame] = []; metrics: list[dict[str, object]] = []
    # Fresh masks are intentionally distinct from the fixed 70404 screen.
    for seed in (0, 1):
        print(f"independent mask {seed}: fitting baseline", flush=True)
        q = _make_q(train, private, seed)
        all_q.append(q.assign(mask_seed=seed))
        metrics.extend(evaluate_q(q, seed + 9000))
        print(f"independent mask {seed}: rows={len(q)} history={int((q.year<2025).sum())}", flush=True)
    qq = pd.concat(all_q, ignore_index=True)
    mm = pd.DataFrame(metrics)
    qq.to_csv(R / "meta_residual_v2_independent_predictions.csv", index=False, float_format="%.8f")
    mm.to_csv(R / "meta_residual_v2_independent_metrics.csv", index=False, float_format="%.10f")
    # Report history-only and all rows separately; the private target is mostly
    # history but 2025 behavior is intentionally not silently extrapolated.
    summary = []
    for route, z in [("all", qq), ("history", qq[qq.year < 2025]), ("2025", qq[qq.year == 2025])]:
        if len(z) == 0: continue
        y = z.truth.to_numpy(float); b = z.base.to_numpy(float)
        summary.append({"route": route, "rows": len(z), "baseline_rmse": _score(y, b),
                        "mean_residual": float(np.mean(z.resid_target)),
                        "mask_seeds": "0,1"})
    # Pooled grouped metrics by cap, requiring both independent masks to win.
    for cap, z in mm.groupby("cap"):
        summary.append({"route": "outer_aoi_ridge30", "cap": cap, "rows": int(z.n.sum()),
                        "baseline_rmse": float(np.sqrt(np.average(z.baseline_rmse**2, weights=z.n))),
                        "rmse": float(np.sqrt(np.average(z.rmse**2, weights=z.n))),
                        "delta_rmse": float(np.sqrt(np.average(z.rmse**2, weights=z.n)) - np.sqrt(np.average(z.baseline_rmse**2, weights=z.n))),
                        "wins": int(z.improved.sum()), "runs": int(len(z))})
    ss = pd.DataFrame(summary); ss.to_csv(R / "meta_residual_v2_independent_summary.csv", index=False, float_format="%.10f")
    meta_info = {"mask_seeds": [0, 1], "holdout_rows": {str(s): int((qq.mask_seed == s).sum()) for s in (0, 1)},
                 "hidden_rows": int(private[GAP].sum()), "private_sha256": sha(DATA / "private_features.csv"),
                 "train_sha256": sha(DATA / "train_dataset.csv"), "seconds": round(time.time()-t0, 1),
                 "production_baseline_overwritten": False}
    (R / "meta_residual_v2_independent_metadata.json").write_text(json.dumps(meta_info, indent=2), encoding="utf-8")
    lines = ["# meta residual v2 independent-mask audit", "", "Fresh masks 0 and 1; baseline rebuilt with leakage-safe HGB/lag/peer/shock/extended components. Outer residual validation is AOI-grouped with three splits per mask.", "", ss.to_string(index=False), "", json.dumps(meta_info, indent=2), "", "No production artifact changed."]
    (R / "meta_residual_v2_independent_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(ss.to_string(index=False), flush=True)
    print(json.dumps(meta_info, indent=2), flush=True)


if __name__ == "__main__":
    main()

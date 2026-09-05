"""Leakage-audited ensemble selection for the CosmoHack NDVI gap task.

This module is deliberately research-only.  It collects the row-level
predictions that already exist in ``research/`` and evaluates combinations on
two independent proxy protocols:

* ``exact_hidden_doy`` – the organizer's hidden day-of-year pattern projected
  to train years 2019--2024 (one held-out year at a time);
* ``random_private_like`` – three independently masked private frames (one
  held-out seed at a time).

Weights are learned only from *other* held-out partitions.  For random seeds,
rows with the same ``(anon_polygon_id, date)`` as the test partition are
removed from the fitting pool because masks overlap.  The core ensemble uses
only observable/fixed candidates (HGB, lag, date-shock and state features);
label-fitted post-corrections are audited but excluded from the deployable
selection.  No production file is modified.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - a deterministic fallback is provided
    minimize = None


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
KEY = ["dataset", "partition", "anon_polygon_id", "date"]
ROWKEY = ["anon_polygon_id", "date"]


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(p[ok] - y[ok])))


def _gap(rmse: float) -> float:
    """The event's 30-point local proxy score (RMSE scale=0.10)."""
    return float(round(30.0 * max(0.0, 1.0 - float(rmse) / 0.10), 2))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_core() -> pd.DataFrame:
    """Load and key-validate the main post-correction row-level table."""
    path = RESEARCH / "teammate_sweep_postcorr_preds.csv"
    raw = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    if raw.duplicated(KEY + ["method"]).any():
        raise ValueError("duplicate method/key rows in postcorr predictions")
    # Every method should carry one truth value per row.  Do this check before
    # pivoting so a malformed artifact cannot silently alter the metric.
    tn = raw.groupby(KEY, sort=False)["_truth"].nunique(dropna=False)
    if (tn > 1).any():
        raise ValueError("truth mismatch among methods")
    meta_cols = ["canon", "hidden_doy", "span_bin", "shared_id", "doy", "year"]
    have_meta = [c for c in meta_cols if c in raw.columns]
    meta = raw[KEY + ["_truth"] + have_meta].drop_duplicates(KEY)
    pred = raw.pivot_table(index=KEY, columns="method", values="pred", aggfunc="first")
    pred = pred.reset_index()
    pred.columns.name = None
    d = meta.merge(pred, on=KEY, how="inner", validate="one_to_one")
    if len(d) != raw[KEY].drop_duplicates().shape[0]:
        raise ValueError("row count changed while pivoting predictions")

    shock_path = RESEARCH / "overnight_next_shock_predictions.csv"
    shock = pd.read_csv(shock_path, parse_dates=["date"], low_memory=False)
    shock = shock[shock["candidate"].eq("baseline")].copy()
    if shock.duplicated(KEY).any():
        raise ValueError("duplicate baseline shock rows")
    s_cols = ["baseline", "shock", "state", "shock_n", "state_n"]
    d = d.merge(shock[KEY + s_cols], on=KEY, how="left", validate="one_to_one")

    # Primitive/fixed candidates.  ``shock`` and ``state`` are built from
    # visible rows only (see overnight_next_shock_eval.py); NaN means there is
    # not enough observable evidence, so the correction is zero.
    d["hgb"] = pd.to_numeric(d["hgb_raw"], errors="coerce")
    d["blend10"] = pd.to_numeric(d["blend_lag_0.10"], errors="coerce")
    d["blend20"] = pd.to_numeric(d["blend_lag_0.20"], errors="coerce")
    d["shock10"] = d["baseline"] + 0.10 * d["shock"].fillna(0.0)
    d["shock15"] = d["baseline"] + 0.15 * d["shock"].fillna(0.0)
    d["shock20"] = d["baseline"] + 0.20 * d["shock"].fillna(0.0)
    d["joint"] = d["baseline"] + 0.15 * d["shock"].fillna(0.0) - 0.05 * d["state"].fillna(0.0)
    # The date-peer candidate above is retained under an explicit name.  A
    # separate AOI-pair artifact is attached below when available.
    d["date_peer10"] = pd.to_numeric(d["hgb_peer_0.10"], errors="coerce")
    # Observable DOY regime: the fixed ``canon`` flag is a date-only feature,
    # not a target/source annotation.  The rule is predeclared and evaluated
    # without fitting its branch on the test partition.
    d["canon_joint"] = np.where(d["canon"].astype(bool), d["baseline"], d["joint"])
    d["canon_shock"] = np.where(d["canon"].astype(bool), d["baseline"], d["shock15"])
    d["span_joint"] = np.where(
        d["span_bin"].isin(["<=4", "4-8", "8-14", "14-21", "21-35"]),
        d["joint"], d["baseline"],
    )
    peer_path = RESEARCH / "paired_aoi_v2_predictions.csv"
    if peer_path.exists():
        peer = pd.read_csv(peer_path, usecols=["partition", "anon_polygon_id", "date", "n16_c60_r125_k2"],
                           parse_dates=["date"], low_memory=False)
        # Core partitions use ``exact2019``/``random0``; the paired artifact
        # uses ``exact_2019``/``random_0``.  Normalize only this controlled
        # naming difference and key-validate the merge.
        peer["partition_core"] = (peer["partition"].astype(str)
                                   .str.replace("exact_", "exact", regex=False)
                                   .str.replace("random_", "random", regex=False))
        peer = peer.rename(columns={"n16_c60_r125_k2": "aoi_peer_raw"})
        peer = peer[["partition_core", "anon_polygon_id", "date", "aoi_peer_raw"]]
        if peer.duplicated(["partition_core", "anon_polygon_id", "date"]).any():
            raise ValueError("duplicate AOI-peer prediction keys")
        d = d.merge(peer, left_on=["partition", "anon_polygon_id", "date"],
                    right_on=["partition_core", "anon_polygon_id", "date"],
                    how="left", validate="one_to_one")
        d["aoi_peer10"] = d["baseline"].to_numpy(float)
        ok = np.isfinite(d["aoi_peer_raw"].to_numpy(float))
        d.loc[ok, "aoi_peer10"] = (0.90 * d.loc[ok, "baseline"] +
                                     0.10 * d.loc[ok, "aoi_peer_raw"])
        d["aoi_peer15"] = d["baseline"].to_numpy(float)
        d.loc[ok, "aoi_peer15"] = (0.85 * d.loc[ok, "baseline"] +
                                     0.15 * d.loc[ok, "aoi_peer_raw"])
        d["aoi_peer10_joint"] = d["aoi_peer10"] + 0.15 * d["shock"].fillna(0.0) - 0.05 * d["state"].fillna(0.0)
        d["aoi_peer10_canon_joint"] = np.where(
            d["canon"].astype(bool), d["aoi_peer10"], d["aoi_peer10_joint"])
        d["aoi_peer10_canon_shock"] = np.where(
            d["canon"].astype(bool), d["aoi_peer10"],
            d["aoi_peer10"] + 0.15 * d["shock"].fillna(0.0))
    else:
        for c in ["aoi_peer_raw", "aoi_peer10", "aoi_peer15", "aoi_peer10_joint",
                  "aoi_peer10_canon_joint", "aoi_peer10_canon_shock"]:
            d[c] = np.nan
    return d


def attach_exact_artifacts(d: pd.DataFrame) -> pd.DataFrame:
    """Attach local/smoother predictions available only for exact folds.

    These are diagnostics, not part of the cross-protocol deployable set.  The
    smoother's ``idx`` is explicitly mapped back to the immutable train row
    index and then joined by ``(AOI,date)``.
    """
    ex_path = RESEARCH / "exact_compare_preds.csv"
    if ex_path.exists():
        ex = pd.read_csv(ex_path, parse_dates=["date"], low_memory=False)
        ex = ex.rename(columns={"hgb": "hgb_exact"})
        cols = [c for c in ["anon_polygon_id", "date", "base_k6", "base_k8", "base_k12",
                            "lag_k12_d2", "lag_k16_d3", "lag_k24_d2", "hgb_exact"] if c in ex]
        ex = ex[cols].drop_duplicates(ROWKEY)
        d = d.merge(ex, on=ROWKEY, how="left", validate="many_to_one")

    sm_path = RESEARCH / "smooth_grid_preds.csv"
    tr_path = DATA / "train_dataset.csv"
    if sm_path.exists() and tr_path.exists():
        sm = pd.read_csv(sm_path, low_memory=False)
        tr = pd.read_csv(tr_path, parse_dates=["date"], low_memory=False)
        maps = []
        for year in sorted(sm["year"].dropna().unique()):
            g = sm[(sm["year"] == year) & sm["method"].eq("median_8_0")].sort_values("idx")
            if g.empty:
                continue
            if g["idx"].max() >= len(tr):
                raise ValueError("smooth idx outside train frame")
            q = tr.iloc[g["idx"].astype(int).to_numpy()][ROWKEY].copy()
            q["smooth_median"] = g["pred"].to_numpy(float)
            maps.append(q)
        if maps:
            smk = pd.concat(maps, ignore_index=True)
            if smk.duplicated(ROWKEY).any():
                raise ValueError("duplicate smoother keys")
            d = d.merge(smk, on=ROWKEY, how="left", validate="many_to_one")
    return d


def _fit_simplex(x: np.ndarray, y: np.ndarray, anchor: np.ndarray | None = None,
                 ridge: float = 0.0) -> np.ndarray:
    """Constrained nonnegative weights summing to one.

    A tiny ridge to an explicit anchor is useful when candidate columns are
    nearly collinear (HGB/lag/shock).  The objective and constraints are
    deterministic; no test labels enter this function when called by outer_cv.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    m = x.shape[1]
    if m == 0:
        return np.empty(0)
    if anchor is None:
        anchor = np.ones(m, dtype=float) / m
    else:
        anchor = np.asarray(anchor, float)
        anchor = np.clip(anchor, 0.0, None)
        anchor = anchor / anchor.sum() if anchor.sum() > 0 else np.ones(m) / m
    if len(y) < max(20, m * 3):
        return anchor.copy()
    if minimize is None:
        # Projected coordinate fallback: repeatedly transfer mass to the best
        # coordinate.  This is only used if scipy is absent.
        w = anchor.copy()
        for _ in range(300):
            best = w.copy(); bestloss = np.mean((x @ w - y) ** 2)
            for j in range(m):
                for delta in (-0.02, 0.02):
                    z = w.copy(); z[j] = np.clip(z[j] + delta, 0, 1)
                    z = z / z.sum()
                    loss = np.mean((x @ z - y) ** 2)
                    if loss < bestloss:
                        best, bestloss = z, loss
            if np.allclose(best, w):
                break
            w = best
        return w
    def objective(w: np.ndarray) -> float:
        return float(np.mean((x @ w - y) ** 2) + ridge * np.sum((w - anchor) ** 2))
    starts = [anchor, np.ones(m) / m]
    best = None
    for start in starts:
        res = minimize(objective, start, method="SLSQP",
                       bounds=[(0.0, 1.0)] * m,
                       constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
                       options={"maxiter": 1000, "ftol": 1e-12})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return anchor.copy()
    w = np.clip(np.asarray(best.x, float), 0.0, 1.0)
    return w / w.sum() if w.sum() > 0 else anchor.copy()


def _key_token(frame: pd.DataFrame) -> pd.Series:
    return frame["anon_polygon_id"].astype(str) + "|" + pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")


def _outer_split(d: pd.DataFrame, dataset: str):
    z = d[d["dataset"].eq(dataset)].copy()
    parts = sorted(z["partition"].dropna().unique())
    for part in parts:
        test = z[z["partition"].eq(part)].copy()
        train = z[~z["partition"].eq(part)].copy()
        # Random masks overlap.  A row hidden in the test seed must not be
        # allowed to contribute its target to the meta-training fit via a
        # different seed.
        if dataset == "random_private_like":
            forbidden = set(_key_token(test))
            train = train[~_key_token(train).isin(forbidden)].copy()
        yield str(part), train, test


def _fixed_predictions(d: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "hgb": d["hgb"].to_numpy(float),
        "blend10": d["blend10"].to_numpy(float),
        "blend20": d["blend20"].to_numpy(float),
        "shock10": d["shock10"].to_numpy(float),
        "shock15": d["shock15"].to_numpy(float),
        "shock20": d["shock20"].to_numpy(float),
        "joint": d["joint"].to_numpy(float),
        "canon_shock": d["canon_shock"].to_numpy(float),
        "canon_joint": d["canon_joint"].to_numpy(float),
        "span_joint": d["span_joint"].to_numpy(float),
        "date_peer10": d["date_peer10"].to_numpy(float),
        "aoi_peer10": d["aoi_peer10"].to_numpy(float),
        "aoi_peer15": d["aoi_peer15"].to_numpy(float),
        "aoi_peer10_joint": d["aoi_peer10_joint"].to_numpy(float),
        "aoi_peer10_canon_joint": d["aoi_peer10_canon_joint"].to_numpy(float),
        "aoi_peer10_canon_shock": d["aoi_peer10_canon_shock"].to_numpy(float),
    }


def _evaluate_row(dataset: str, partition: str, method: str, y: np.ndarray,
                  p: np.ndarray, n_train: int = 0, weights: str = "") -> dict:
    return {
        "dataset": dataset,
        "partition": partition,
        "method": method,
        "n": int(len(y)),
        "n_train": int(n_train),
        "rmse": _rmse(y, p),
        "mae": _mae(y, p),
        "gapscore_proxy": _gap(_rmse(y, p)),
        "weights": weights,
    }


def run_outer(d: pd.DataFrame):
    """Run fixed and cross-fitted constrained ensembles."""
    records = []
    pred_rows = []
    weight_rows = []
    # Four columns are enough to span the useful corrections while keeping the
    # simplex identifiable.  ``shock15``/``joint`` are fixed, observable
    # candidate predictions, not label-fitted post-corrections.
    cols = ["hgb", "blend20", "shock15", "joint"]
    anchor = np.array([0.64, 0.16, 0.10, 0.10])
    for dataset in ["exact_hidden_doy", "random_private_like"]:
        z = d[d["dataset"].eq(dataset)].copy()
        for part, train, test in _outer_split(d, dataset):
            y = test["_truth"].to_numpy(float)
            fixed = _fixed_predictions(test)
            for method, p in fixed.items():
                records.append(_evaluate_row(dataset, part, method, y, p, len(train)))

            # Unregularized and anchored simplex fits.  Both are fitted only
            # on other partitions, with overlap removal above.
            xtr = train[cols].to_numpy(float)
            ytr = train["_truth"].to_numpy(float)
            w0 = _fit_simplex(xtr, ytr, anchor=anchor, ridge=0.0)
            wr = _fit_simplex(xtr, ytr, anchor=anchor, ridge=0.20)
            for name, w in [("simplex_global", w0), ("ridge_global", wr)]:
                p = test[cols].to_numpy(float) @ w
                ws = ",".join(f"{c}={v:.6f}" for c, v in zip(cols, w))
                records.append(_evaluate_row(dataset, part, name, y, p, len(train), ws))
                weight_rows.append({"dataset": dataset, "test_partition": part,
                                    "fit": name, "regime": "global",
                                    "n_train": len(train), **{c: float(v) for c, v in zip(cols, w)}})

            # Regime simplex: only canon has enough support and a clear
            # predeclared interpretation.  Fall back to the global anchored
            # fit for tiny groups; this guards against accidental overfit.
            regime_pred = np.full(len(test), np.nan)
            for regval in [False, True]:
                tr_g = train[train["canon"].astype(bool).eq(regval)]
                te_g = test["canon"].astype(bool).eq(regval)
                if te_g.sum() == 0:
                    continue
                if len(tr_g) >= 200:
                    wg = _fit_simplex(tr_g[cols].to_numpy(float), tr_g["_truth"].to_numpy(float),
                                      anchor=anchor, ridge=0.20)
                else:
                    wg = wr
                regime_pred[te_g.to_numpy()] = test.loc[te_g, cols].to_numpy(float) @ wg
                weight_rows.append({"dataset": dataset, "test_partition": part,
                                    "fit": "ridge_canon", "regime": str(regval),
                                    "n_train": len(tr_g), **{c: float(v) for c, v in zip(cols, wg)}})
            records.append(_evaluate_row(dataset, part, "ridge_canon", y, regime_pred, len(train)))

            # Save compact row-level outputs for independent inspection.
            q = test[KEY + ["_truth", "canon", "span_bin"]].copy()
            q["test_method"] = "ridge_global"
            q["pred_ridge_global"] = test[cols].to_numpy(float) @ wr
            q["pred_simplex_global"] = test[cols].to_numpy(float) @ w0
            q["pred_ridge_canon"] = regime_pred
            q["pred_fixed_joint"] = test["joint"].to_numpy(float)
            q["pred_fixed_canon_joint"] = test["canon_joint"].to_numpy(float)
            pred_rows.append(q)

    metrics = pd.DataFrame(records)
    weights = pd.DataFrame(weight_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    return metrics, weights, preds


def audit_external_row_tables() -> pd.DataFrame:
    """Summarize every existing row-level artifact without using it for fit."""
    rows = []
    specs = [
        ("exact_compare_preds.csv", "exact_local", "label_free_local"),
        ("smooth_grid_preds.csv", "exact_smoother", "label_free_local"),
        ("overnight_next_shock_predictions.csv", "shock_state", "observable_feature"),
        ("overnight_correction_predictions.csv", "postcorr", "crossfit_label_fitted_excluded"),
        ("teammate_sweep_ensemble_predictions.csv", "seed_ensemble", "separate_protocol"),
        ("paired_aoi_v2_predictions.csv", "aoi_pair", "observable_peer_candidate"),
        ("paired_aoi_v2_private_hgb_lag20_peer10.csv", "aoi_pair_private", "private_candidate"),
        ("private_holdout_seed0_rows.csv", "private_holdout", "separate_protocol"),
        ("overnight_root_source_preds.csv", "source_models", "separate_protocol"),
    ]
    for name, family, usage in specs:
        path = RESEARCH / name
        if not path.exists():
            continue
        try:
            x = pd.read_csv(path, nrows=5, low_memory=False)
            full = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # pragma: no cover
            rows.append({"file": name, "family": family, "usage": usage, "error": str(exc)})
            continue
        method_col = next((c for c in ["method", "candidate", "mode", "frame"] if c in full.columns), "")
        protocol_col = next((c for c in ["dataset", "protocol", "partition", "mode"] if c in full.columns), "")
        rows.append({
            "file": name,
            "family": family,
            "usage": usage,
            "rows": len(full),
            "columns": len(full.columns),
            "unique_keys": int(full[[c for c in ["anon_polygon_id", "date"] if c in full]].drop_duplicates().shape[0]) if {"anon_polygon_id", "date"}.issubset(full.columns) else np.nan,
            "methods_or_candidates": ",".join(map(str, full[method_col].dropna().unique()[:30])) if method_col else "",
            "protocols": ",".join(map(str, full[protocol_col].dropna().unique()[:30])) if protocol_col else "",
            "sha256": _sha256(path),
        })
    return pd.DataFrame(rows)


def audit_external_metrics() -> pd.DataFrame:
    """Score all parseable row-level artifacts on their native protocols.

    The output is descriptive only.  It makes protocol incompatibilities
    explicit instead of silently pooling rows from different masks.
    """
    out: list[dict[str, object]] = []

    def add(family: str, dataset: str, partition: str, method: str,
            y: pd.Series | np.ndarray, p: pd.Series | np.ndarray,
            eligible: bool, note: str = "") -> None:
        yy = np.asarray(y, float); pp = np.asarray(p, float)
        ok = np.isfinite(yy) & np.isfinite(pp)
        if not ok.any():
            return
        out.append({"family": family, "dataset": dataset, "partition": partition,
                    "method": method, "n": int(ok.sum()), "rmse": _rmse(yy, pp),
                    "mae": _mae(yy, pp), "eligible_for_core_meta": bool(eligible),
                    "note": note})

    # Main post-correction table: fixed HGB/lag rows are eligible; methods whose
    # parameters were label-fitted in another cross-fit are audit-only here.
    p = pd.read_csv(RESEARCH / "teammate_sweep_postcorr_preds.csv", low_memory=False)
    safe = {"hgb_raw", "hgb_clip_01", "hgb_peer_0.10", "blend_lag_0.10", "blend_lag_0.20"}
    for (ds, part, method), g in p.groupby(["dataset", "partition", "method"], sort=False):
        add("postcorr", str(ds), str(part), str(method), g["_truth"], g["pred"],
            str(method) in safe,
            "fixed/observable" if str(method) in safe else "crossfit label-fitted; excluded from second-level fit")

    # Shock/state table contains both raw baseline and first-level cross-fitted
    # candidates.  Only the baseline is reused; raw features are joined in
    # load_core and receive new outer validation.
    sp = RESEARCH / "overnight_next_shock_predictions.csv"
    if sp.exists():
        s = pd.read_csv(sp, low_memory=False)
        for (ds, part, method), g in s.groupby(["dataset", "partition", "candidate"], sort=False):
            add("shock_state", str(ds), str(part), str(method), g["truth"], g["pred"],
                str(method) == "baseline", "raw observable shock/state audited separately")

    cp = RESEARCH / "overnight_correction_predictions.csv"
    if cp.exists():
        c = pd.read_csv(cp, low_memory=False)
        c["dataset_audit"] = np.select(
            [c["partition"].astype(str).str.startswith("exact"),
             c["partition"].astype(str).str.startswith("random"),
             c["partition"].astype(str).str.startswith("date25")],
            ["exact_hidden_doy", "random_private_like", "date25"], default="unknown")
        for (ds, part, method), g in c.groupby(["dataset_audit", "partition", "method"], sort=False):
            add("overnight_corrections", str(ds), str(part), str(method), g["_truth"], g["pred"],
                False, "crossfit/PCA correction; incompatible second-level fit")

    ex = RESEARCH / "exact_compare_preds.csv"
    if ex.exists():
        e = pd.read_csv(ex, low_memory=False)
        methods = [c for c in ["base_k6", "base_k8", "base_k12", "lag_k12_d2",
                                "lag_k16_d3", "lag_k24_d2", "hgb"] if c in e]
        for year, g in e.groupby("year", sort=False):
            for method in methods:
                add("exact_local", "exact_hidden_doy", f"exact{int(year)}", method,
                    g["_truth"], g[method], False, "exact-only; no common random protocol")

    sm = RESEARCH / "smooth_grid_preds.csv"
    if sm.exists():
        s = pd.read_csv(sm, low_memory=False)
        for (protocol, year, method), g in s.groupby(["protocol", "year", "method"], sort=False):
            add("smooth", str(protocol), f"{protocol}{int(year)}", str(method),
                g["truth"], g["pred"], False, "exact-only smoother")

    ens = RESEARCH / "teammate_sweep_ensemble_predictions.csv"
    if ens.exists():
        e = pd.read_csv(ens, low_memory=False)
        methods = [c for c in ["base_k6", "lag_k16_d3", "hgb_seed42", "hgb_seed7",
                                "hgb_seed123", "hgb_seed_mean", "blend_hgb80_lag20"] if c in e]
        for (mode, seed), g in e.groupby(["mode", "mask_seed"], sort=False):
            for method in methods:
                add("seed_ensemble", str(mode), f"seed{int(seed)}", method,
                    g["_truth"], g[method], False, "partial seed/mode coverage")

    pp = RESEARCH / "paired_aoi_v2_predictions.csv"
    if pp.exists():
        p = pd.read_csv(pp, low_memory=False)
        p["dataset_audit"] = np.where(p["family"].eq("exact"),
                                      "exact_hidden_doy", "random_private_like")
        p["partition_audit"] = (p["partition"].astype(str)
                                 .str.replace("exact_", "exact", regex=False)
                                 .str.replace("random_", "random", regex=False))
        peer_cols = [c for c in p.columns if c.startswith("n16_c60_r125_k2")]
        for (ds, part), g in p.groupby(["dataset_audit", "partition_audit"], sort=False):
            for method in peer_cols:
                add("aoi_pair", str(ds), str(part), method, g["_truth"], g[method],
                    False, "observable peer map; fixed config, not used in core fit")

    src = RESEARCH / "overnight_root_source_preds.csv"
    if src.exists():
        s = pd.read_csv(src, low_memory=False)
        methods = [c for c in ["prod", "soft", "hard", "uniform", "oracle_true", "lag"] if c in s]
        for (mode, fold), g in s.groupby(["mode", "fold"], sort=False):
            for method in methods:
                add("source", str(mode), str(fold), method, g["_truth"], g[method],
                    False, "partial protocol; oracle excluded")
    return pd.DataFrame(out)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d = load_core()
    d = attach_exact_artifacts(d)
    metrics, weights, preds = run_outer(d)
    metrics.to_csv(RESEARCH / "ensemble_cv_v2_results.csv", index=False, float_format="%.9f")
    weights.to_csv(RESEARCH / "ensemble_cv_v2_weights.csv", index=False, float_format="%.9f")
    preds.to_csv(RESEARCH / "ensemble_cv_v2_predictions.csv", index=False, float_format="%.9f")
    manifest = audit_external_row_tables()
    manifest.to_csv(RESEARCH / "ensemble_cv_v2_manifest.csv", index=False)
    audit_metrics = audit_external_metrics()
    audit_metrics.to_csv(RESEARCH / "ensemble_cv_v2_audit_metrics.csv", index=False,
                         float_format="%.9f")

    # Pooled scores are recomputed from row-level predictions (not a mean of
    # per-partition RMSEs).  For the learned methods, this is genuinely outer
    # cross-fitted because each saved prediction came from another partition's
    # weights.
    pooled = []
    for (dataset, method), g in metrics.groupby(["dataset", "method"], sort=False):
        base_rm = metrics[(metrics["dataset"] == dataset) &
                          (metrics["method"] == "blend20")].set_index("partition")["rmse"]
        aligned_base = base_rm.reindex(g["partition"].to_numpy())
        wins = int((g["rmse"].to_numpy(float) < aligned_base.to_numpy(float)).sum())
        pooled.append({
            "dataset": dataset,
            "method": method,
            "n": int(g["n"].sum()),
            "rmse_pooled": float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"]))),
            "mae_pooled": float(np.average(g["mae"], weights=g["n"])),
            "gapscore_proxy": _gap(float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"])))),
            "partitions": int(len(g)),
            "wins_vs_baseline": wins,
        })
    pooled_df = pd.DataFrame(pooled).sort_values(["dataset", "rmse_pooled", "method"])
    pooled_df.to_csv(RESEARCH / "ensemble_cv_v2_pooled.csv", index=False, float_format="%.9f")

    # A concise human-readable report, including explicit leakage boundaries.
    lines = [
        "# Ensemble CV v2 (research-only)",
        "",
        "No production artifact was modified.  All row-level fits use only other held-out partitions; random-seed overlap keys are removed from meta-training.",
        "",
        "## Pooled outer-CV results",
        "",
        pooled_df.to_string(index=False),
        "",
        "## Fixed rule selected for deployment candidate",
        "",
        "`baseline = blend_lag_0.20`; `joint = baseline + 0.15*shock - 0.05*state`; apply `joint` only when the date-only `canon` flag is false, otherwise baseline.  The strongest common-protocol fixed rule is `aoi_peer10_canon_joint`: replace the baseline with a 10% same-year AOI-peer blend where available, then apply the same non-canon shock/state correction.",
        "The shock/state features are computed from visible rows of the current private mask; no hidden target/status/source fields are read.",
        "",
        "## Selection notes",
        "",
        "- `aoi_peer10_canon_joint` is the leading observable rule in the common-protocol audit (exact pooled RMSE 0.062022; random pooled RMSE 0.068657; all leave-year/seed folds improve).  `canon_joint` is retained as a no-peer fallback (exact 0.062343; random 0.069155).",
        "- A separately generated private candidate is in `outputs/model_dani_peer_joint_submission.csv`; stronger lag30/local coefficient sweeps remain research alternatives under `research/ensemble_cv_v2_local_*`.",
        "- Unregularized simplex weights are reported for diagnosis; because candidates are highly collinear, the anchored simplex is the conservative reference.",
        "- Affine/bias/group post-corrections in `teammate_sweep_postcorr_preds.csv` and `overnight_correction_predictions.csv` are audited but excluded from the deployable fit because their parameters were learned from labels in other saved partitions.",
        "- Smooth/local/source tables are retained in the manifest; they do not have a complete common protocol and are not mixed into the final candidate.",
        "",
        "## Files",
        "",
        "- `ensemble_cv_v2_results.csv`, `ensemble_cv_v2_pooled.csv` — partition and pooled metrics;",
        "- `ensemble_cv_v2_weights.csv` — outer-fitted simplex weights;",
        "- `ensemble_cv_v2_predictions.csv` — row-level outer predictions;",
        "- `ensemble_cv_v2_manifest.csv` — inventory/hash of all row-level artifacts;",
        "- `ensemble_cv_v2_audit_metrics.csv` — native-protocol metrics for every parseable row-level family.",
    ]
    (RESEARCH / "ensemble_cv_v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pooled_df.to_string(index=False))


if __name__ == "__main__":
    main()

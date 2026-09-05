"""Leakage-safe common-date shock / latent-state evaluator.

This is an overnight research experiment for the Dani HGB + lag ensemble.  It
reuses the row-level cross-fitted baseline predictions already produced in
``teammate_sweep_postcorr_preds.csv`` and constructs two correction features
from *observed rows of the current mask only*:

``common_shock``
    robust same-date residual median over other AOIs;
``latent_state``
    robust, distance-weighted recent residual state for the same AOI/year.

Coefficients are fitted cross-partition (other years or other random seeds)
and then applied to the held-out partition.  Thus hidden labels from a test
partition are never used to build its correction.  This script is diagnostic
only and never writes ``outputs/model_dani_tuned*``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402


KEY = ["anon_polygon_id", "date"]
BASE_METHOD = "blend_lag_0.20"
CLIP_LO, CLIP_HI = -0.5, 1.2


def _median_or_nan(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _seasonal_baseline(frame: pd.DataFrame, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a robust seasonal baseline and observed residuals.

    The hierarchy is AOI/year/16-day bin -> AOI/16-day bin -> year/bin ->
    global bin -> global median.  Every table is fitted using rows marked
    ``known`` in the current fold, so hidden query values cannot enter it.
    """
    d = frame.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["_yr"] = d["date"].dt.year.astype(int)
    d["_doy"] = d["date"].dt.dayofyear.astype(int)
    d["_bin"] = ((d["_doy"] - 1) // 16).astype(int)
    d["_known_eval"] = np.asarray(known, dtype=bool)
    d["_y_eval"] = pd.to_numeric(d.get("_truth", d["primary_ndvi"]), errors="coerce")
    d.loc[~d["_known_eval"], "_y_eval"] = np.nan

    obs = d.loc[d["_known_eval"] & d["_y_eval"].notna(),
               ["anon_polygon_id", "_yr", "_bin", "_doy", "_y_eval"]].copy()
    if obs.empty:
        return np.full(len(d), np.nan), np.full(len(d), np.nan)

    # Keep extreme malformed values out of the robust profile, while retaining
    # the normal NDVI range of the source data.
    obs = obs[obs["_y_eval"].between(-0.5, 1.2)].copy()
    p1 = (obs.groupby(["anon_polygon_id", "_yr", "_bin"], observed=True)["_y_eval"]
            .median().rename("_p1").reset_index())
    p2 = (obs.groupby(["anon_polygon_id", "_bin"], observed=True)["_y_eval"]
            .median().rename("_p2").reset_index())
    p3 = (obs.groupby(["_yr", "_bin"], observed=True)["_y_eval"]
            .median().rename("_p3").reset_index())
    p4 = (obs.groupby(["_bin"], observed=True)["_y_eval"]
            .median().rename("_p4").reset_index())
    gmed = float(obs["_y_eval"].median())

    z = d[["anon_polygon_id", "_yr", "_bin"]].copy()
    z = z.merge(p1, on=["anon_polygon_id", "_yr", "_bin"], how="left", sort=False)
    z = z.merge(p2, on=["anon_polygon_id", "_bin"], how="left", sort=False)
    z = z.merge(p3, on=["_yr", "_bin"], how="left", sort=False)
    z = z.merge(p4, on=["_bin"], how="left", sort=False)
    prof = z["_p1"].combine_first(z["_p2"]).combine_first(z["_p3"]).combine_first(z["_p4"]).fillna(gmed).to_numpy(float)
    y = d["_y_eval"].to_numpy(float)
    residual = y - prof
    residual[~known] = np.nan
    residual = np.clip(residual, -0.5, 0.5)
    return prof, residual


def _shock_feature(frame: pd.DataFrame, known: np.ndarray, residual: np.ndarray,
                   query_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same-date, leave-AOI-out robust residual median for query rows."""
    d = frame.copy()
    d["date"] = pd.to_datetime(d["date"])
    dates = d["date"].to_numpy()
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    by_date: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    for dt, g in d.loc[known & np.isfinite(residual)].groupby("date", sort=False):
        ii = g.index.to_numpy(dtype=int)
        vals = residual[ii]
        good = np.isfinite(vals)
        by_date[pd.Timestamp(dt)] = (ids[ii][good], vals[good])

    out = np.full(len(query_idx), np.nan, dtype=float)
    counts = np.zeros(len(query_idx), dtype=int)
    for j, i in enumerate(query_idx):
        pair = by_date.get(pd.Timestamp(dates[i]))
        if pair is None:
            continue
        peer_ids, vals = pair
        vals = vals[peer_ids != ids[i]]
        vals = vals[np.isfinite(vals)]
        counts[j] = len(vals)
        # Three peers is the minimum for a meaningful date shock; the median
        # is deliberately conservative against one anomalous AOI.
        if len(vals) >= 3:
            out[j] = float(np.median(np.clip(vals, -0.3, 0.3)))
    return out, counts


def _latent_feature(frame: pd.DataFrame, known: np.ndarray, residual: np.ndarray,
                    query_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recent same-AOI/year residual state, computed from visible rows."""
    d = frame.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["_yr"] = d["date"].dt.year.astype(int)
    ordinal = d["date"].map(pd.Timestamp.toordinal).to_numpy(float)
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    out = np.full(len(query_idx), np.nan, dtype=float)
    counts = np.zeros(len(query_idx), dtype=int)
    groups: dict[tuple[str, int], np.ndarray] = {}
    for k, ix in d.loc[known & np.isfinite(residual)].groupby(["anon_polygon_id", "_yr"], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        ii = ii[np.isfinite(residual[ii])]
        if len(ii):
            groups[(str(k[0]), int(k[1]))] = ii
    for j, i in enumerate(query_idx):
        ii = groups.get((ids[i], int(d["_yr"].iat[i])))
        if ii is None or len(ii) < 2:
            continue
        dist = np.abs(ordinal[ii] - ordinal[i])
        take = np.argsort(dist)[: min(8, len(ii))]
        take = take[dist[take] <= 120]
        if len(take) < 2:
            continue
        vals = np.clip(residual[ii[take]], -0.3, 0.3)
        w = np.exp(-dist[take] / 45.0)
        if not np.isfinite(w).all() or w.sum() <= 0:
            continue
        out[j] = float(np.average(vals, weights=w))
        counts[j] = len(vals)
    return out, counts


def _prediction_map(preds: pd.DataFrame, dataset: str, partition: str) -> pd.DataFrame:
    z = preds[(preds["dataset"] == dataset) &
              (preds["partition"] == partition) &
              (preds["method"] == BASE_METHOD)].copy()
    z["date"] = pd.to_datetime(z["date"])
    z = z[KEY + ["_truth", "pred"]].rename(columns={"pred": "baseline"})
    if z.duplicated(KEY).any():
        raise ValueError(f"duplicate baseline keys in {dataset}/{partition}")
    return z


def _make_partition(frame: pd.DataFrame, mask: np.ndarray, dataset: str,
                    partition: str, pred_map: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    mask = np.asarray(mask, dtype=bool)
    # _mask_private stores truth before masking; make_fold stores it in _truth.
    if "_truth" not in d:
        d["_truth"] = pd.to_numeric(d["primary_ndvi"], errors="coerce")
    known = (~mask) & pd.to_numeric(d["primary_ndvi"], errors="coerce").notna().to_numpy()
    _, residual = _seasonal_baseline(d, known)
    qi = np.flatnonzero(mask)
    shock, shock_n = _shock_feature(d, known, residual, qi)
    state, state_n = _latent_feature(d, known, residual, qi)
    q = d.loc[qi, KEY + ["_truth"]].copy().reset_index(drop=True)
    q["date"] = pd.to_datetime(q["date"])
    q["dataset"] = dataset
    q["partition"] = partition
    q["shock"] = shock
    q["state"] = state
    q["shock_n"] = shock_n
    q["state_n"] = state_n
    q = q.merge(pred_map, on=KEY, how="left", validate="one_to_one")
    if q["baseline"].isna().any():
        raise ValueError(f"missing baseline rows in {dataset}/{partition}")
    # The saved row-level table and the fold truth must agree; this guards
    # against accidental mask/order drift without using labels for features.
    if not np.allclose(q["_truth_x"].to_numpy(float), q["_truth_y"].to_numpy(float), equal_nan=True):
        raise ValueError(f"truth mismatch in {dataset}/{partition}")
    q["truth"] = q["_truth_x"].to_numpy(float)
    q = q.drop(columns=[c for c in ["_truth_x", "_truth_y", "_truth"] if c in q.columns], errors="ignore")
    # Restore a single explicit truth column after the defensive merge.
    q["truth"] = pd.to_numeric(d.loc[qi, "_truth"], errors="coerce").to_numpy(float)
    return q


def _fit_linear(parts: list[pd.DataFrame], features: list[str]) -> np.ndarray:
    """Fit a conservative correction on non-test partitions."""
    z = pd.concat(parts, ignore_index=True)
    cols = ["baseline"] + features
    ok = np.isfinite(z["truth"].to_numpy(float)) & np.isfinite(z["baseline"].to_numpy(float))
    for c in features:
        ok &= np.isfinite(z[c].to_numpy(float))
    if ok.sum() < 30:
        return np.zeros(1 + len(features), dtype=float)
    X = z.loc[ok, features].to_numpy(float)
    y = z.loc[ok, "truth"].to_numpy(float) - z.loc[ok, "baseline"].to_numpy(float)
    # Ridge on slopes only, with a tiny penalty relative to the scale of the
    # residual features.  This prevents a three-seed fit from exploding.
    A = np.c_[np.ones(len(X)), X]
    reg = np.eye(A.shape[1]) * 0.5
    reg[0, 0] = 0.05
    coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
    coef[0] = float(np.clip(coef[0], -0.03, 0.03))
    coef[1:] = np.clip(coef[1:], -0.75, 0.75)
    return coef


def _apply(z: pd.DataFrame, coef: np.ndarray, features: list[str]) -> np.ndarray:
    p = z["baseline"].to_numpy(float).copy()
    ok = np.ones(len(z), dtype=bool)
    X = []
    for c in features:
        a = z[c].to_numpy(float)
        ok &= np.isfinite(a)
        X.append(np.nan_to_num(a, nan=0.0))
    if X:
        p += coef[0] + np.column_stack(X) @ coef[1:]
    else:
        p += coef[0]
    # If a feature is missing, use the intercept-only correction.  This is
    # safer than treating missing peer/state evidence as a zero shock.
    if X:
        p[~ok] = z.loc[~ok, "baseline"].to_numpy(float) + coef[0]
    return np.clip(p, CLIP_LO, CLIP_HI)


def _metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    e = p[ok] - y[ok]
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def main() -> None:
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    preds = pd.read_csv(RESEARCH / "teammate_sweep_postcorr_preds.csv", parse_dates=["date"], low_memory=False)

    exact_parts: list[pd.DataFrame] = []
    for year in range(2019, 2025):
        fold, _ = make_fold(train.copy(), private.copy(), year)
        fold["date"] = pd.to_datetime(fold["date"])
        mask = fold["is_synthetic_gap"].fillna(False).to_numpy(bool)
        part = _make_partition(fold, mask, "exact_hidden_doy", f"exact{year}",
                               _prediction_map(preds, "exact_hidden_doy", f"exact{year}"))
        exact_parts.append(part)

    random_parts: list[pd.DataFrame] = []
    for seed in (0, 1, 2):
        frame, mask = _mask_private(private.copy(), seed)
        part = _make_partition(frame, mask, "random_private_like", f"random{seed}",
                               _prediction_map(preds, "random_private_like", f"random{seed}"))
        random_parts.append(part)

    protocols = [("exact_hidden_doy", exact_parts), ("random_private_like", random_parts)]
    rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for dataset, parts in protocols:
        for ti, test in enumerate(parts):
            train_parts = [p for j, p in enumerate(parts) if j != ti]
            candidates: dict[str, np.ndarray] = {"baseline": test["baseline"].to_numpy(float)}
            base_rm, base_ma = _metric(test["truth"].to_numpy(float), candidates["baseline"])
            rows.append({"dataset": dataset, "partition": str(test["partition"].iat[0]),
                         "candidate": "baseline", "n": len(test),
                         "coef_intercept": np.nan, "coef_shock": np.nan,
                         "coef_state": np.nan,
                         "feature_finite_shock": int(test["shock"].notna().sum()),
                         "feature_finite_state": int(test["state"].notna().sum()),
                         "rmse": base_rm, "mae": base_ma})
            z = test[KEY + ["truth", "baseline", "shock", "state", "shock_n", "state_n"]].copy()
            z["dataset"] = dataset; z["partition"] = str(test["partition"].iat[0]); z["candidate"] = "baseline"
            z["pred"] = candidates["baseline"]
            pred_rows.append(z)
            for name, fs in [("shock_cf", ["shock"]),
                             ("state_cf", ["state"]),
                             ("joint_cf", ["shock", "state"])]:
                coef = _fit_linear(train_parts, fs)
                candidates[name] = _apply(test, coef, fs)
                c_shock = np.nan
                c_state = np.nan
                if fs == ["shock"] and len(coef) > 1:
                    c_shock = coef[1]
                elif fs == ["state"] and len(coef) > 1:
                    c_state = coef[1]
                elif fs == ["shock", "state"]:
                    if len(coef) > 1:
                        c_shock = coef[1]
                    if len(coef) > 2:
                        c_state = coef[2]
                rows.append({"dataset": dataset, "partition": str(test["partition"].iat[0]),
                             "candidate": name, "n": len(test), "coef_intercept": coef[0],
                             "coef_shock": c_shock, "coef_state": c_state,
                             "feature_finite_shock": int(test["shock"].notna().sum()),
                             "feature_finite_state": int(test["state"].notna().sum()),
                             "rmse": _metric(test["truth"].to_numpy(float), candidates[name])[0],
                             "mae": _metric(test["truth"].to_numpy(float), candidates[name])[1]})
                z = test[KEY + ["truth", "baseline", "shock", "state", "shock_n", "state_n"]].copy()
                z["dataset"] = dataset; z["partition"] = str(test["partition"].iat[0]); z["candidate"] = name
                z["pred"] = candidates[name]
                pred_rows.append(z)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESEARCH / "overnight_next_shock_metrics.csv", index=False)
    pp = pd.concat(pred_rows, ignore_index=True)
    pp.to_csv(RESEARCH / "overnight_next_shock_predictions.csv", index=False)

    agg_rows: list[dict[str, object]] = []
    for (dataset, candidate), g in metrics.groupby(["dataset", "candidate"], sort=False):
        agg_rows.append({"dataset": dataset, "candidate": candidate,
                         "n": int(g["n"].sum()),
                         "rmse_pooled": float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"]))),
                         "mae_pooled": float(np.average(g["mae"], weights=g["n"])),
                         "partitions": int(len(g))})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(RESEARCH / "overnight_next_shock_aggregate.csv", index=False)

    # A candidate is only interesting if it wins pooled and on most held-out
    # partitions in both protocols.  This criterion intentionally errs on the
    # side of retaining the known production blend.
    base = metrics[metrics.candidate == "baseline"].set_index(["dataset", "partition"])["rmse"]
    lines = [
        "# Overnight common-date shock / latent-state experiment",
        "",
        "Дата: 2026-09-05. Эксперимент leakage-safe и research-only: `outputs/model_dani_tuned*` не изменялись.",
        "",
        "## Протокол",
        "",
        "- `exact_hidden_doy`: private synthetic DOY projected onto train years 2019--2024 (6 held-out partitions).",
        "- `random_private_like`: 15% random hidden rows per AOI/year, seeds 0--2 (3 held-out partitions).",
        "- Baseline is the saved production-like `0.80*HGB + 0.20*lag` row-level prediction.",
        "- `shock` uses only visible same-date peers from the current fold; `state` uses only visible nearby rows of the same AOI/year.",
        "- Correction coefficients are fit on the other partitions and applied to the held-out partition.",
        "",
        "## Pooled cross-fitted RMSE",
        "",
        agg.sort_values(["dataset", "rmse_pooled"]).to_string(index=False),
        "",
        "## Decision",
        "",
    ]
    for dataset, parts in protocols:
        sub = metrics[metrics.dataset == dataset]
        b = sub[sub.candidate == "baseline"].set_index("partition")["rmse"]
        lines.append(f"### {dataset}")
        for cand in ["shock_cf", "state_cf", "joint_cf"]:
            c = sub[sub.candidate == cand].set_index("partition")["rmse"]
            common = b.index.intersection(c.index)
            wins = int((c.loc[common] < b.loc[common]).sum())
            cg = sub[sub.candidate == cand].set_index("partition").loc[common]
            bg = sub[sub.candidate == "baseline"].set_index("partition").loc[common]
            delta = float(np.sqrt(np.average(c.loc[common] ** 2, weights=cg["n"])) -
                          np.sqrt(np.average(b.loc[common] ** 2, weights=bg["n"])))
            lines.append(f"- `{cand}`: wins {wins}/{len(common)} partitions; pooled RMSE delta {delta:+.6f} vs baseline.")
        lines.append("")
    lines.extend([
        "Вывод: common-date shock и latent-state проверены на realistic private-mask CV. Поправка не считается production-кандидатом, если выигрыш не устойчив одновременно на exact и random протоколах; production `model_dani_tuned_submission.csv` оставлен без изменений.",
        "",
        "## Файлы",
        "",
        "- `overnight_next_shock_eval.py` — воспроизводимый evaluator;",
        "- `overnight_next_shock_metrics.csv` — метрики по partition и cross-fit коэффициенты;",
        "- `overnight_next_shock_aggregate.csv` — pooled summary;",
        "- `overnight_next_shock_predictions.csv` — row-level diagnostic predictions.",
    ])
    (RESEARCH / "overnight_next_shock_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(agg.sort_values(["dataset", "rmse_pooled"]).to_string(index=False))


if __name__ == "__main__":
    main()

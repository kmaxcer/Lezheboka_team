"""Leakage-safe AOI residual-bias calibration for the HGB+lag20 baseline.

Research-only evaluator.  For each scored partition, residuals are taken from
other held-out partitions, where the calibration row was masked when its
prediction was made (visible OOF).  Keys in the current outer mask are removed
from the calibration pool.  No production files are changed.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
BASELINE = RESEARCH / "teammate_sweep_postcorr_preds.csv"
METHOD = "blend_lag_0.20"
ID = "anon_polygon_id"
DATE = "date"


def _stat(values: np.ndarray, kind: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if kind == "median":
        return float(np.median(values))
    if kind == "mean":
        return float(np.mean(values))
    if kind == "trim10":
        if len(values) < 10:
            return float(np.mean(values))
        lo, hi = np.quantile(values, [0.10, 0.90])
        return float(np.mean(np.clip(values, lo, hi)))
    raise ValueError(kind)


def _load() -> pd.DataFrame:
    d = pd.read_csv(BASELINE, parse_dates=[DATE], low_memory=False)
    d = d.loc[d["method"].eq(METHOD)].copy()
    d[DATE] = pd.to_datetime(d[DATE]).dt.normalize()
    d[ID] = d[ID].astype(str)
    d["year"] = d[DATE].dt.year.astype(int)
    d["doy"] = d[DATE].dt.dayofyear.astype(int)
    d["resid"] = d["_truth"].astype(float) - d["pred"].astype(float)
    d["aoi"] = d[ID]
    d["aoi_year"] = d[ID] + "|" + d["year"].astype(str)
    d["aoi_bin32"] = d[ID] + "|" + (d["doy"] // 32).astype(str)
    if d.duplicated(["dataset", "partition", ID, DATE]).any():
        raise ValueError("duplicate baseline key within partition")
    return d.reset_index(drop=True)


def _keyset(d: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return set(zip(d[ID].astype(str), pd.to_datetime(d[DATE]).dt.normalize()))


def _partitions(d: pd.DataFrame):
    """Yield dataset, test rows, and leakage-safe OOF calibration rows."""
    for dataset, g in d.groupby("dataset", sort=False):
        for part in g["partition"].drop_duplicates():
            test = g.loc[g["partition"].eq(part)].copy()
            cal = g.loc[g["partition"].ne(part)].copy()
            blocked = _keyset(test)
            keep = [k not in blocked for k in zip(cal[ID], cal[DATE])]
            cal = cal.loc[np.asarray(keep, bool)].copy()
            yield str(dataset), str(part), test, cal


def _fit(
    cal: pd.DataFrame,
    group_col: str,
    min_n: int,
    shrink: float,
    clip: float,
    stat: str,
    prior_name: str,
) -> tuple[dict[object, float], float, int]:
    vals = cal["resid"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    prior = 0.0 if prior_name == "zero" else _stat(vals, stat)
    if not np.isfinite(prior):
        prior = 0.0
    prior = float(np.clip(prior, -clip, clip))
    out: dict[object, float] = {}
    for key, g in cal.groupby(group_col, sort=False, dropna=False):
        r = g["resid"].to_numpy(float)
        r = r[np.isfinite(r)]
        if len(r) < int(min_n):
            continue
        raw = _stat(r, stat)
        if np.isfinite(raw):
            out[key] = float(np.clip(shrink * raw + (1.0 - shrink) * prior, -clip, clip))
    return out, prior, len(out)


def _apply(
    test: pd.DataFrame,
    cal: pd.DataFrame,
    group_col: str,
    min_n: int,
    shrink: float,
    clip: float,
    stat: str,
    prior_name: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    bias, prior, ng = _fit(cal, group_col, min_n, shrink, clip, stat, prior_name)
    p = test["pred"].to_numpy(float).copy()
    covered = np.zeros(len(test), dtype=bool)
    for i, key in enumerate(test[group_col].to_numpy(object)):
        if key in bias:
            p[i] += bias[key]
            covered[i] = True
    return np.clip(p, -0.5, 1.2), covered, prior, ng


def _raw_map(cal: pd.DataFrame, group_col: str, min_n: int, stat: str) -> dict[object, float]:
    """Compute group statistics once; the hyperparameter grid reuses them."""
    out: dict[object, float] = {}
    for key, g in cal.groupby(group_col, sort=False, dropna=False):
        r = g["resid"].to_numpy(float)
        r = r[np.isfinite(r)]
        if len(r) < int(min_n):
            continue
        value = _stat(r, stat)
        if np.isfinite(value):
            out[key] = float(value)
    return out


def _apply_raw(
    test: pd.DataFrame,
    raw_map: dict[object, float],
    prior: float,
    shrink: float,
    clip: float,
    group_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized application of one raw AOI map."""
    mapped = test[group_col].map(raw_map).to_numpy(float)
    covered = np.isfinite(mapped)
    bias = np.where(covered, shrink * mapped + (1.0 - shrink) * prior, 0.0)
    bias = np.clip(bias, -clip, clip)
    p = np.clip(test["pred"].to_numpy(float) + bias, -0.5, 1.2)
    return p, covered


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(p[ok] - y[ok]))) if ok.any() else np.nan


def main() -> None:
    d = _load()
    metrics: list[dict[str, object]] = []
    groups = ("aoi", "aoi_year", "aoi_bin32")
    mins = (3, 5, 8, 12, 20, 30)
    shrinks = (0.10, 0.25, 0.50, 0.75, 1.00)
    clips = (0.01, 0.02, 0.03, 0.04)
    stats = ("median", "mean", "trim10")
    priors = ("zero", "global")
    for dataset, part, test, cal in _partitions(d):
        y = test["_truth"].to_numpy(float)
        base = test["pred"].to_numpy(float)
        cohorts = [("all", np.ones(len(test), bool))]
        if dataset == "random_private_like":
            cohorts.append(("2025", test["year"].eq(2025).to_numpy(bool)))
        for group_col in groups:
            for min_n in mins:
                for stat in stats:
                    # Groupby/robust statistic is the expensive part; compute
                    # it once and reuse for all shrinkage/clip/prior settings.
                    raw_map = _raw_map(cal, group_col, min_n, stat)
                    all_resid = cal["resid"].to_numpy(float)
                    all_resid = all_resid[np.isfinite(all_resid)]
                    prior_values = {
                        "zero": 0.0,
                        "global": float(_stat(all_resid, stat)) if len(all_resid) else 0.0,
                    }
                    for shrink in shrinks:
                        for clip in clips:
                            for prior_name in priors:
                                prior = float(np.clip(prior_values[prior_name], -clip, clip))
                                pred, covered = _apply_raw(
                                    test, raw_map, prior, shrink, clip, group_col
                                )
                                for cohort, take in cohorts:
                                    if not take.any():
                                        continue
                                    br = _rmse(y[take], base[take])
                                    rr = _rmse(y[take], pred[take])
                                    metrics.append({
                                        "dataset": dataset, "partition": part,
                                        "cohort": cohort, "group": group_col,
                                        "min_n": min_n, "shrink": shrink,
                                        "clip": clip, "stat": stat, "prior": prior_name,
                                        "n": int(take.sum()), "n_cal": int(len(cal)),
                                        "n_groups": int(len(raw_map)), "n_applied": int((covered & take).sum()),
                                        "coverage": float((covered & take).sum() / take.sum()),
                                        "prior_value": prior,
                                        "baseline_rmse": br,
                                        "baseline_mae": _mae(y[take], base[take]),
                                        "rmse": rr, "mae": _mae(y[take], pred[take]),
                                        "delta_rmse": rr - br,
                                    })
        print(f"{dataset}/{part}: test={len(test)} cal={len(cal)} baseline={_rmse(y, base):.6f}", flush=True)

    m = pd.DataFrame(metrics)
    m.to_csv(RESEARCH / "tail_anomaly_v2_aoi_bias_metrics.csv", index=False)
    hcols = ["dataset", "cohort", "group", "min_n", "shrink", "clip", "stat", "prior"]
    agg_rows = []
    for keys, g in m.groupby(hcols, sort=False, dropna=False):
        n = g["n"].to_numpy(float)
        b = g["baseline_rmse"].to_numpy(float)
        r = g["rmse"].to_numpy(float)
        row = dict(zip(hcols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "runs": int(len(g)), "n": int(n.sum()),
            "baseline_rmse_pooled": float(np.sqrt(np.average(b * b, weights=n))),
            "rmse_pooled": float(np.sqrt(np.average(r * r, weights=n))),
            "delta_rmse_pooled": float(np.sqrt(np.average(r * r, weights=n)) - np.sqrt(np.average(b * b, weights=n))),
            "mae_weighted": float(np.average(g["mae"], weights=n)),
            "coverage_weighted": float(np.average(g["coverage"], weights=n)),
            "improved_runs": int((g["delta_rmse"] < 0).sum()),
        })
        agg_rows.append(row)
    a = pd.DataFrame(agg_rows)
    views = []
    for dataset, cohort, label in (
        ("exact_hidden_doy", "all", "exact"),
        ("random_private_like", "all", "random"),
        ("random_private_like", "2025", "random2025"),
    ):
        v = a[(a["dataset"] == dataset) & (a["cohort"] == cohort)].copy()
        keys = ["group", "min_n", "shrink", "clip", "stat", "prior"]
        v = v[keys + ["rmse_pooled", "baseline_rmse_pooled", "delta_rmse_pooled", "coverage_weighted", "improved_runs", "runs"]]
        v = v.rename(columns={c: f"{label}_{c}" for c in v.columns if c not in keys})
        views.append(v)
    s = views[0]
    for v in views[1:]:
        s = s.merge(v, on=["group", "min_n", "shrink", "clip", "stat", "prior"], how="inner", validate="one_to_one")
    if len(s):
        s["worst_delta"] = s[["exact_delta_rmse_pooled", "random_delta_rmse_pooled", "random2025_delta_rmse_pooled"]].max(axis=1)
        s["mean_delta"] = s[["exact_delta_rmse_pooled", "random_delta_rmse_pooled", "random2025_delta_rmse_pooled"]].mean(axis=1)
        s["all_three_improve"] = s["worst_delta"] < 0
        s = s.sort_values(["all_three_improve", "worst_delta", "mean_delta"], ascending=[False, True, True])
    s.to_csv(RESEARCH / "tail_anomaly_v2_aoi_bias_aggregate.csv", index=False)
    if len(s):
        best = s.iloc[0]
        decision = "retain as a research candidate" if bool(best["all_three_improve"]) else "DISCARD; no stable AOI bias gain over baseline"
        lines = [
            "# AOI residual-bias calibration v2",
            "",
            "Baseline: HGB + 20% lag-aware local prediction (blend_lag_0.20).",
            "Calibration uses only other held-out partitions; each residual is visible-row OOF and current outer keys are excluded.",
            "",
            f"Best robust row: group={best['group']}, min_n={int(best['min_n'])}, shrink={best['shrink']:.2f}, clip={best['clip']:.2f}, stat={best['stat']}, prior={best['prior']}.",
            "",
            f"- exact hidden-DOY: {best['exact_baseline_rmse_pooled']:.6f} -> {best['exact_rmse_pooled']:.6f} (delta {best['exact_delta_rmse_pooled']:+.6f}; coverage {best['exact_coverage_weighted']:.1%}; improved {int(best['exact_improved_runs'])}/{int(best['exact_runs'])})",
            f"- random private-like: {best['random_baseline_rmse_pooled']:.6f} -> {best['random_rmse_pooled']:.6f} (delta {best['random_delta_rmse_pooled']:+.6f}; coverage {best['random_coverage_weighted']:.1%}; improved {int(best['random_improved_runs'])}/{int(best['random_runs'])})",
            f"- random 2025: {best['random2025_baseline_rmse_pooled']:.6f} -> {best['random2025_rmse_pooled']:.6f} (delta {best['random2025_delta_rmse_pooled']:+.6f}; coverage {best['random2025_coverage_weighted']:.1%}; improved {int(best['random2025_improved_runs'])}/{int(best['random2025_runs'])})",
            "",
            f"All-three improvement: {bool(best['all_three_improve'])}.",
            f"Decision: {decision}.",
            "Production files were not modified.",
        ]
    else:
        lines = ["# AOI residual-bias calibration v2", "", "No evaluable candidates; discard."]
    (RESEARCH / "tail_anomaly_v2_aoi_bias_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

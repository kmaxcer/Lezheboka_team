"""Leakage-safe source-posterior residual regressions.

This is a research-only screen.  It uses only observable year/DOY sensor
presence posteriors as features; the unmasked ``true_src`` column is retained
solely for diagnostics and is never fed to a fitted model.  Every regression
is fitted on non-canonical rows from the other outer partitions, then scored
on the held-out partition (and the random-2025 cohort).
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
sys.path.insert(0, str(R))
import ensemble_cv_v2_source_correction as src  # noqa: E402


def features(d: pd.DataFrame, kind: str) -> np.ndarray:
    """Construct deterministic observable features for source correction."""
    p = d[["p_s2", "p_landsat", "p_modis"]].to_numpy(float)
    # Center year and use smooth seasonal coordinates.  No target/source
    # columns are referenced here.
    yr = d["year"].to_numpy(float)
    doy = d["doy"].to_numpy(float)
    ys = (yr - 2020.0) / 5.0
    ang = 2.0 * np.pi * (doy - 1.0) / 365.25
    z = [p[:, 0], p[:, 1], p[:, 2], ys, np.sin(ang), np.cos(ang)]
    if kind in ("source_year", "source_year_doy"):
        z += [p[:, 0] * ys, p[:, 1] * ys, p[:, 2] * ys]
    if kind == "source_year_doy":
        z += [p[:, 0] * np.sin(ang), p[:, 1] * np.sin(ang), p[:, 2] * np.sin(ang),
              p[:, 0] * np.cos(ang), p[:, 1] * np.cos(ang), p[:, 2] * np.cos(ang)]
    return np.column_stack(z)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Small ridge solve with an unpenalized intercept."""
    xx = np.column_stack([np.ones(len(x)), x])
    # Standardize non-intercept columns using training statistics upstream;
    # all features are already O(1), so a direct stable solve suffices.
    reg = np.eye(xx.shape[1]); reg[0, 0] = 0.0
    return np.linalg.solve(xx.T @ xx + alpha * reg, xx.T @ y)


def predict(beta: np.ndarray, x: np.ndarray, cap: float) -> np.ndarray:
    q = np.column_stack([np.ones(len(x)), x]) @ beta
    return np.clip(q, -cap, cap)


def main() -> None:
    z, tr, pr = src._read_cv()
    z = src._attach_posteriors(z, tr, pr)
    y = z["truth"].to_numpy(float)
    methods = ["source", "source_year", "source_year_doy"]
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    caps = [0.01, 0.02, 0.04]
    rows = []
    row_records = []
    for ds, part, train_ix, test_ix in src._folds(z):
        tr0 = z.loc[train_ix].copy(); te = z.loc[test_ix].copy()
        for lw in (0.30, 0.325, 0.35, 0.40):
            pred, _ = src._base_pred(z, np.arange(len(z)), lw)
            # The local formula has no source correction on canonical DOYs.
            fit_mask = ~tr0["canon"].to_numpy(bool)
            score_mask = ~te["canon"].to_numpy(bool)
            xtr_all = features(tr0, "source_year_doy")
            xte_all = features(te, "source_year_doy")
            yy = y[train_ix] - pred[train_ix]
            for kind in methods:
                xtr = features(tr0, kind); xte = features(te, kind)
                for alpha in alphas:
                    # Use only non-canonical outer-train rows.  Query labels
                    # and true source sidecars are not part of X.
                    beta = fit_ridge(xtr[fit_mask], yy[fit_mask], alpha)
                    for cap in caps:
                        corr = np.zeros(len(te), float)
                        corr[score_mask] = predict(beta, xte[score_mask], cap)
                        pp = pred[test_ix] + corr
                        qy = y[test_ix]
                        row_records.append(pd.DataFrame({
                            "dataset": ds, "partition": part, "lag_weight": lw,
                            "method": kind, "alpha": alpha, "cap": cap,
                            "year": te["year"].to_numpy(int), "truth": qy,
                            "pred": pp, "base": pred[test_ix],
                        }))
    rr = pd.concat(row_records, ignore_index=True)
    rr["cohort"] = np.where(rr["dataset"].eq("exact"), "exact", np.where(rr["year"].eq(2025), "random2025", "random"))
    out = []
    for (lw, method, alpha, cap, cohort), g in rr.groupby(["lag_weight", "method", "alpha", "cap", "cohort"], sort=False):
        e = g["pred"].to_numpy(float) - g["truth"].to_numpy(float)
        eb = g["base"].to_numpy(float) - g["truth"].to_numpy(float)
        wins = 0; folds = 0
        for _, q in g.groupby("partition", sort=True):
            a = np.sqrt(np.mean((q["pred"].to_numpy(float)-q["truth"].to_numpy(float))**2))
            b = np.sqrt(np.mean((q["base"].to_numpy(float)-q["truth"].to_numpy(float))**2))
            wins += int(a < b); folds += 1
        out.append({"lag_weight": lw, "method": method, "alpha": alpha, "cap": cap,
                    "cohort": cohort, "n": len(g), "rmse": np.sqrt(np.mean(e*e)),
                    "baseline_rmse": np.sqrt(np.mean(eb*eb)), "delta_rmse": np.sqrt(np.mean(e*e))-np.sqrt(np.mean(eb*eb)),
                    "wins": wins, "folds": folds})
    o = pd.DataFrame(out)
    # Pivot compactly for ranking configurations across all three cohorts.
    piv = o.pivot_table(index=["lag_weight", "method", "alpha", "cap"], columns="cohort", values=["rmse", "baseline_rmse", "delta_rmse", "wins", "folds"], aggfunc="first")
    piv.columns = ["_".join(str(x) for x in c) for c in piv.columns]
    piv = piv.reset_index()
    for c in ("exact", "random", "random2025"):
        if f"delta_rmse_{c}" not in piv: piv[f"delta_rmse_{c}"] = np.nan
    piv["worst_delta"] = piv[[f"delta_rmse_{c}" for c in ("exact", "random", "random2025")]].max(axis=1)
    piv["mean_delta"] = piv[[f"delta_rmse_{c}" for c in ("exact", "random", "random2025")]].mean(axis=1)
    piv = piv.sort_values(["worst_delta", "mean_delta"])
    o.to_csv(R / "ensemble_cv_v2_source_regression_cohorts.csv", index=False, float_format="%.9f")
    piv.to_csv(R / "ensemble_cv_v2_source_regression_summary.csv", index=False, float_format="%.9f")
    # Save only a compact top table; full row predictions are intentionally
    # omitted to keep the research directory manageable.
    report = ["# Source-posterior regression audit", "", piv.head(40).to_string(index=False), "", "No candidate is deployable unless all exact/random/random2025 cohorts improve under outer folds."]
    (R / "ensemble_cv_v2_source_regression_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

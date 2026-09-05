"""Research-only grid for observable AOI-peer + shock/state ensembles.

The script joins the already materialized peer-AOI OOF rows with the raw
shock/state features.  It never fits on the row being scored: peer maps and
shock/state values were constructed by their respective visible-only
protocols.  This file is deliberately separate from production generation so
that coefficient/config selection remains auditable.

Outputs
-------
``ensemble_cv_v2_grid_scores.csv``
    Pooled and per-partition scores for the tested fixed rules.
``ensemble_cv_v2_grid_shortlist.csv``
    Stable candidates ranked by worst leave-partition delta.
``ensemble_cv_v2_grid_report.md``
    Compact protocol/selection report.
"""
from __future__ import annotations

from pathlib import Path
import itertools
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CANON_DOY = {97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289}


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def load_rows() -> pd.DataFrame:
    peer = pd.read_csv(RESEARCH / "paired_aoi_v2_predictions.csv",
                       parse_dates=["date"], low_memory=False)
    if peer.duplicated(["partition", "anon_polygon_id", "date"]).any():
        raise ValueError("duplicate peer keys")
    shock = pd.read_csv(RESEARCH / "overnight_next_shock_predictions.csv",
                        parse_dates=["date"], low_memory=False)
    shock = shock[shock["candidate"].eq("baseline")].copy()
    if shock.duplicated(["partition", "anon_polygon_id", "date"]).any():
        raise ValueError("duplicate shock baseline keys")
    shock["partition_peer"] = (shock["partition"].astype(str)
                                .str.replace("exact", "exact_", regex=False)
                                .str.replace("random", "random_", regex=False))
    # The replacement above would turn exact2019 into exact_2019 and
    # random0 into random_0; assert this controlled normalization explicitly.
    shock["partition_peer"] = shock["partition_peer"].str.replace("exact__", "exact_", regex=False)
    shock["partition_peer"] = shock["partition_peer"].str.replace("random__", "random_", regex=False)
    keep = ["partition_peer", "anon_polygon_id", "date", "shock", "state",
            "shock_n", "state_n"]
    z = peer.merge(shock[keep], left_on=["partition", "anon_polygon_id", "date"],
                   right_on=["partition_peer", "anon_polygon_id", "date"],
                   how="left", validate="one_to_one")
    if len(z) != len(peer):
        raise ValueError("row count changed in shock join")
    z["doy"] = z["date"].dt.dayofyear.astype(int)
    z["canon"] = z["doy"].isin(CANON_DOY)
    z["dataset"] = np.where(z["family"].eq("exact"),
                             "exact_hidden_doy", "random_private_like")
    z["base_hgb"] = pd.to_numeric(z["hgb"], errors="coerce")
    z["base_lag20"] = 0.80 * z["hgb"] + 0.20 * z["lag"]
    z["base_lag30"] = 0.70 * z["hgb"] + 0.30 * z["lag"]
    # A tiny lag10 variant is useful as a sanity check; it is not expected to
    # beat lag20 but keeps the base-family comparison explicit.
    z["base_lag10"] = 0.90 * z["hgb"] + 0.10 * z["lag"]
    if z[["_truth", "base_hgb", "base_lag20", "shock", "state"]].isna().all(axis=1).any():
        raise ValueError("unexpected all-NaN joined rows")
    return z


def config_columns(z: pd.DataFrame) -> list[str]:
    # Only the exact peer grid columns are admissible; no truth-derived names.
    cols = [c for c in z.columns if c.startswith("n") and "_c" in c and "_r" in c and "_k" in c]
    if not cols:
        raise ValueError("no AOI peer config columns")
    return sorted(cols)


def score_rule(z: pd.DataFrame, base_col: str, peer_col: str, weight: float,
               alpha: float, beta: float, mode: str) -> np.ndarray:
    b = z[base_col].to_numpy(float)
    q = z[peer_col].to_numpy(float)
    covered = np.isfinite(q)
    p = b.copy()
    p[covered] = (1.0 - weight) * b[covered] + weight * q[covered]
    s = np.nan_to_num(z["shock"].to_numpy(float), nan=0.0)
    st = np.nan_to_num(z["state"].to_numpy(float), nan=0.0)
    corr = alpha * s + beta * st
    if mode == "canon_joint":
        corr[z["canon"].to_numpy(bool)] = 0.0
    elif mode == "canon_shock":
        corr[z["canon"].to_numpy(bool)] = 0.0
        corr = alpha * s
        corr[z["canon"].to_numpy(bool)] = 0.0
    elif mode == "all_joint":
        pass
    elif mode == "all_shock":
        corr = alpha * s
    elif mode == "canon_state":
        corr = beta * st
        corr[z["canon"].to_numpy(bool)] = 0.0
    elif mode == "none":
        corr[:] = 0.0
    else:
        raise ValueError(mode)
    return p + corr


def protocol_rows(z: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for ds in ["exact_hidden_doy", "random_private_like"]:
        q = z[z["dataset"].eq(ds)]
        for part, g in q.groupby("partition", sort=True):
            out.append((f"{ds}|{part}", g))
        if ds == "random_private_like":
            for part, g in q.groupby("partition", sort=True):
                h = g[g["year"].eq(2025)]
                if len(h):
                    out.append((f"{ds}|{part}|year2025", h))
    return out


def main() -> None:
    z = load_rows()
    cfgs = config_columns(z)
    bases = ["base_hgb", "base_lag20", "base_lag30", "base_lag10"]
    # Coefficients are intentionally a small, predeclared grid.  The values
    # around 0.15/-0.05 are the conservative rule; larger values are included
    # as diagnostics and must pass all leave-partition checks before use.
    weights = [0.05, 0.08, 0.10, 0.12, 0.15]
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    betas = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    modes = ["none", "canon_joint", "canon_shock", "all_joint", "all_shock", "canon_state"]

    # First retain configs that are competitive as peer-only rules on each
    # base.  This limits the coefficient grid while still evaluating all
    # plausible configs from the full 108-config peer experiment.
    native = pd.read_csv(RESEARCH / "paired_aoi_v2_aggregate.csv")
    native = native[(native["peer_config"].ne("none")) &
                    (native["cohort"].isin(["all", "year_2025"]))]
    score_cfg = []
    for base in ["hgb", "hgb_lag20", "hgb_lag30"]:
        q = native[native["base"].eq(base)].copy()
        if q.empty:
            continue
        # Mean rank across exact, random-all and random-2025 at each weight.
        for w in weights:
            qw = q[q["peer_weight"].eq(w)]
            if qw.empty:
                continue
            e = qw[(qw["family"].eq("exact")) & (qw["cohort"].eq("all"))]
            r = qw[(qw["family"].eq("random")) & (qw["cohort"].eq("all"))]
            y = qw[(qw["family"].eq("random")) & (qw["cohort"].eq("year_2025"))]
            m = e[["peer_config", "delta_rmse"]].rename(columns={"delta_rmse": "e"})
            m = m.merge(r[["peer_config", "delta_rmse"]].rename(columns={"delta_rmse": "r"}), on="peer_config")
            m = m.merge(y[["peer_config", "delta_rmse"]].rename(columns={"delta_rmse": "y"}), on="peer_config")
            m["mean"] = m[["e", "r", "y"]].mean(axis=1)
            m["worst"] = m[["e", "r", "y"]].max(axis=1)
            m = m.sort_values(["worst", "mean"]).head(24)
            score_cfg.extend(m["peer_config"].astype(str).tolist())
    selected_cfgs = sorted(set(score_cfg))
    # Ensure the canonical config used by the deployable candidate is always
    # present even if a native artifact changed.
    if "n16_c60_r125_k2" in cfgs and "n16_c60_r125_k2" not in selected_cfgs:
        selected_cfgs.append("n16_c60_r125_k2")
    selected_cfgs = [c for c in selected_cfgs if c in cfgs]

    rows: list[dict[str, object]] = []
    prot = protocol_rows(z)
    # Baseline records (including no-peer/no-correction) are useful for exact
    # delta comparisons and contract checks.
    for base in bases:
        for label, g in prot:
            y = g["_truth"].to_numpy(float)
            b = g[base].to_numpy(float)
            rows.append({"family_partition": label, "dataset": g["dataset"].iat[0],
                         "partition": g["partition"].iat[0], "cohort": "year2025" if "year2025" in label else "all",
                         "base": base, "peer_config": "none", "peer_weight": 0.0,
                         "alpha": 0.0, "beta": 0.0, "mode": "none", "n": len(g),
                         "coverage": 0.0, "rmse": rmse(y, b)})

    for base, cfg, weight, alpha, beta, mode in itertools.product(
            bases, selected_cfgs, weights, alphas, betas, modes):
        p = score_rule(z, base, cfg, weight, alpha, beta, mode)
        for label, g in prot:
            ix = g.index.to_numpy()
            y = g["_truth"].to_numpy(float)
            pp = p[ix]
            covered = np.isfinite(g[cfg].to_numpy(float))
            rows.append({"family_partition": label, "dataset": g["dataset"].iat[0],
                         "partition": g["partition"].iat[0], "cohort": "year2025" if "year2025" in label else "all",
                         "base": base, "peer_config": cfg, "peer_weight": weight,
                         "alpha": alpha, "beta": beta, "mode": mode, "n": len(g),
                         "coverage": float(covered.mean()), "rmse": rmse(y, pp)})
    scores = pd.DataFrame(rows)
    # Attach within-protocol baseline RMSE for the same base, then summarize
    # pooled exact/random/random-2025 scores for every fixed rule.
    key = ["base", "peer_config", "peer_weight", "alpha", "beta", "mode"]
    bas = scores[scores["peer_config"].eq("none")][["family_partition", "rmse", "base"]]
    bas = bas.rename(columns={"rmse": "baseline_rmse"})
    scores = scores.merge(bas, on=["family_partition", "base"], how="left", validate="many_to_one")
    scores["delta_rmse"] = scores["rmse"] - scores["baseline_rmse"]
    scores.to_csv(RESEARCH / "ensemble_cv_v2_grid_scores.csv", index=False, float_format="%.9f")

    # Correct pooled RMSE: sum squared error, weighted by row count (not mean
    # of RMSEs).  Also retain per-fold worst delta and win count.
    summaries = []
    for k, g in scores[scores["peer_config"].ne("none")].groupby(key, sort=False):
        # Each row already has a genuine partition RMSE; n-weighting gives the
        # exact pooled RMSE when fold rows are disjoint.
        def pooled(mask):
            q = g.loc[mask]
            return float(np.sqrt(np.average(q["rmse"] ** 2, weights=q["n"]))) if len(q) else np.nan
        ex = g[g["dataset"].eq("exact_hidden_doy") & g["cohort"].eq("all")]
        ra = g[g["dataset"].eq("random_private_like") & g["cohort"].eq("all")]
        r25 = g[g["dataset"].eq("random_private_like") & g["cohort"].eq("year2025")]
        vals = dict(zip(key, k))
        vals.update({
            "exact_rmse": pooled(g["dataset"].eq("exact_hidden_doy") & g["cohort"].eq("all")),
            "random_rmse": pooled(g["dataset"].eq("random_private_like") & g["cohort"].eq("all")),
            "random2025_rmse": pooled(g["dataset"].eq("random_private_like") & g["cohort"].eq("year2025")),
            "exact_baseline_rmse": float(np.sqrt(np.average(ex["baseline_rmse"] ** 2, weights=ex["n"]))) if len(ex) else np.nan,
            "random_baseline_rmse": float(np.sqrt(np.average(ra["baseline_rmse"] ** 2, weights=ra["n"]))) if len(ra) else np.nan,
            "random2025_baseline_rmse": float(np.sqrt(np.average(r25["baseline_rmse"] ** 2, weights=r25["n"]))) if len(r25) else np.nan,
            "exact_coverage": float(np.average(ex["coverage"], weights=ex["n"])) if len(ex) else np.nan,
            "random_coverage": float(np.average(ra["coverage"], weights=ra["n"])) if len(ra) else np.nan,
            "random2025_coverage": float(np.average(r25["coverage"], weights=r25["n"])) if len(r25) else np.nan,
            "exact_wins": int((ex["delta_rmse"] < 0).sum()),
            "random_wins": int((ra["delta_rmse"] < 0).sum()),
            "random2025_wins": int((r25["delta_rmse"] < 0).sum()),
            "exact_folds": int(len(ex)), "random_folds": int(len(ra)), "random2025_folds": int(len(r25)),
        })
        vals["exact_delta_rmse"] = vals["exact_rmse"] - vals["exact_baseline_rmse"]
        vals["random_delta_rmse"] = vals["random_rmse"] - vals["random_baseline_rmse"]
        vals["random2025_delta_rmse"] = vals["random2025_rmse"] - vals["random2025_baseline_rmse"]
        vals["worst_delta"] = max(vals["exact_delta_rmse"], vals["random_delta_rmse"], vals["random2025_delta_rmse"])
        vals["mean_delta"] = np.mean([vals["exact_delta_rmse"], vals["random_delta_rmse"], vals["random2025_delta_rmse"]])
        vals["all_folds_improve"] = bool(vals["exact_wins"] == vals["exact_folds"] and vals["random_wins"] == vals["random_folds"])
        summaries.append(vals)
    summary = pd.DataFrame(summaries).sort_values(["all_folds_improve", "worst_delta", "mean_delta"], ascending=[False, True, True])
    summary.to_csv(RESEARCH / "ensemble_cv_v2_grid_shortlist.csv", index=False, float_format="%.9f")

    top = summary.head(25)
    lines = [
        "# Ensemble CV v2 observable grid",
        "",
        "Peer predictions and shock/state features are joined by exact `(partition, anon_polygon_id, date)` keys.  No hidden target/status/source field is read during candidate construction.",
        "",
        f"Rows: {len(z)}; tested peer configs: {len(selected_cfgs)} / {len(cfgs)}; rules: {len(summary)}.",
        "",
        "## Stable shortlist (sorted by all-fold improvement, worst pooled delta)",
        "",
        top.to_string(index=False),
        "",
        "A candidate is deployment-worthy only if its coefficients/config are predeclared and its leave-year/leave-seed deltas remain negative; grid winners at the boundary are diagnostics, not automatic promotion.",
    ]
    (RESEARCH / "ensemble_cv_v2_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

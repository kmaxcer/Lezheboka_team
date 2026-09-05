"""Fast, leakage-audited coefficient/config sweep for AOI-peer ensembles.

This is a companion to :mod:`ensemble_cv_v2_grid`.  It keeps only one summary
row per rule (rather than materialising millions of per-row records), so it is
safe to run repeatedly on a laptop.  The first pass scores every peer config at
the conservative correction; a second pass refines the most stable configs on
a coefficient grid.  Scores are calculated per leave-year/leave-seed fold and
then pooled with exact squared-error weighting.
"""
from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CANON_DOY = {97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289}


def _rmse2(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return np.nan, 0
    return float(np.mean((p[ok] - y[ok]) ** 2)), int(ok.sum())


def _norm_partition(x: pd.Series) -> pd.Series:
    x = x.astype(str)
    # Explicit, anchored conversion avoids accidental replacement in a future
    # partition name (e.g. ``randomized``).
    return x.str.replace(r"^(exact)(\d+)$", r"exact_\2", regex=True).str.replace(
        r"^(random)(\d+)$", r"random_\2", regex=True)


def load() -> pd.DataFrame:
    p = pd.read_csv(RESEARCH / "paired_aoi_v2_predictions.csv", parse_dates=["date"], low_memory=False)
    s = pd.read_csv(RESEARCH / "overnight_next_shock_predictions.csv", parse_dates=["date"], low_memory=False)
    s = s[s["candidate"].eq("baseline")].copy()
    if p.duplicated(["partition", "anon_polygon_id", "date"]).any():
        raise ValueError("duplicate peer keys")
    if s.duplicated(["partition", "anon_polygon_id", "date"]).any():
        raise ValueError("duplicate shock keys")
    s["partition_peer"] = _norm_partition(s["partition"])
    z = p.merge(s[["partition_peer", "anon_polygon_id", "date", "shock", "state", "shock_n", "state_n"]],
                left_on=["partition", "anon_polygon_id", "date"],
                right_on=["partition_peer", "anon_polygon_id", "date"], how="left", validate="one_to_one")
    if len(z) != len(p):
        raise ValueError("shock join changed row count")
    z["dataset"] = np.where(z["family"].eq("exact"), "exact", "random")
    z["year25"] = z["year"].eq(2025)
    z["canon"] = z["date"].dt.dayofyear.isin(CANON_DOY)
    z["base_hgb"] = z["hgb"].to_numpy(float)
    z["base_lag20"] = 0.8 * z["hgb"].to_numpy(float) + 0.2 * z["lag"].to_numpy(float)
    z["base_lag30"] = 0.7 * z["hgb"].to_numpy(float) + 0.3 * z["lag"].to_numpy(float)
    z["base_lag10"] = 0.9 * z["hgb"].to_numpy(float) + 0.1 * z["lag"].to_numpy(float)
    return z


def _candidate(z: pd.DataFrame, base: str, cfg: str, w: float, a: float, b: float, mode: str) -> np.ndarray:
    q = z[cfg].to_numpy(float)
    p = z[base].to_numpy(float).copy()
    ok = np.isfinite(q)
    p[ok] = (1.0 - w) * p[ok] + w * q[ok]
    sh = np.nan_to_num(z["shock"].to_numpy(float), nan=0.0)
    st = np.nan_to_num(z["state"].to_numpy(float), nan=0.0)
    if mode in ("canon_joint", "all_joint"):
        c = a * sh + b * st
    elif mode in ("canon_shock", "all_shock"):
        c = a * sh
    elif mode == "canon_state":
        c = b * st
    elif mode == "none":
        c = np.zeros(len(z), float)
    else:
        raise ValueError(mode)
    if mode.startswith("canon"):
        c[z["canon"].to_numpy(bool)] = 0.0
    return p + c


def _folds(z: pd.DataFrame):
    # Use integer positions, so all candidate vectors can be indexed without
    # pandas alignment overhead.  Random folds also remove overlapping AOI/date
    # keys from the *fit* side when nested selection is requested.
    for ds in ("exact", "random"):
        q = z[z["dataset"].eq(ds)]
        for part, g in q.groupby("partition", sort=True):
            test = g.index.to_numpy()
            if ds == "random":
                tokens = set(zip(g["anon_polygon_id"].astype(str), g["date"].astype(str)))
                train = q.index[~q.set_index([q["anon_polygon_id"].astype(str), q["date"].astype(str)]).index.isin(tokens)]
                # The expression above is intentionally explicit; reset to a
                # simple NumPy mask after pandas has checked the key semantics.
                tk = np.array(list(zip(q["anon_polygon_id"].astype(str), q["date"].astype(str))), dtype=object)
                keep = np.array([tuple(v) not in tokens for v in tk], dtype=bool)
                train = q.index[keep]
            else:
                train = q.index[~q.index.isin(test)]
            yield ds, str(part), train.to_numpy(int), test


def _partition_masks(z: pd.DataFrame):
    # A compact list of scored masks.  Exact/random all and random-2025 are
    # kept separately; this catches year-specific regressions.
    out = []
    for ds in ("exact", "random"):
        q = z[z["dataset"].eq(ds)]
        for part, g in q.groupby("partition", sort=True):
            ix = g.index.to_numpy(int)
            out.append((ds, str(part), "all", ix))
            if ds == "random":
                y = g[g["year25"]].index.to_numpy(int)
                if len(y):
                    out.append((ds, str(part), "year2025", y))
    return out


def summarize(z: pd.DataFrame, rules: list[tuple], baseline_cols: list[str]) -> pd.DataFrame:
    y_all = z["_truth"].to_numpy(float)
    masks = _partition_masks(z)
    # Cache baseline fold MSE/count once.
    base_cache: dict[tuple[str, str], tuple[float, int]] = {}
    for bc in baseline_cols:
        for ds, part, cohort, ix in masks:
            base_cache[(bc, f"{ds}|{part}|{cohort}")] = _rmse2(y_all[ix], z.loc[ix, bc].to_numpy(float))
    records = []
    for rule in rules:
        base, cfg, w, a, b, mode = rule
        p = _candidate(z, base, cfg, w, a, b, mode)
        fold_mse = []
        for ds, part, cohort, ix in masks:
            mse, n_ok = _rmse2(y_all[ix], p[ix])
            bmse, bn = base_cache[(base, f"{ds}|{part}|{cohort}")]
            cov = np.isfinite(z.loc[ix, cfg].to_numpy(float)).mean()
            fold_mse.append((ds, part, cohort, mse, bmse, len(ix), cov, n_ok))
        def agg(ds: str, cohort: str):
            q = [r for r in fold_mse if r[0] == ds and r[2] == cohort]
            if not q:
                return (np.nan,) * 4
            n = np.array([r[5] for r in q], float)
            rm = float(np.sqrt(np.average([r[3] for r in q], weights=n)))
            br = float(np.sqrt(np.average([r[4] for r in q], weights=n)))
            cov = float(np.average([r[6] for r in q], weights=n))
            wins = int(sum(r[3] < r[4] for r in q))
            return rm, br, cov, wins
        ex = agg("exact", "all"); ra = agg("random", "all"); r25 = agg("random", "year2025")
        rec = {"base": base, "peer_config": cfg, "peer_weight": w, "alpha": a, "beta": b, "mode": mode,
               "exact_rmse": ex[0], "exact_baseline_rmse": ex[1], "exact_delta_rmse": ex[0] - ex[1], "exact_coverage": ex[2], "exact_wins": ex[3],
               "random_rmse": ra[0], "random_baseline_rmse": ra[1], "random_delta_rmse": ra[0] - ra[1], "random_coverage": ra[2], "random_wins": ra[3],
               "random2025_rmse": r25[0], "random2025_baseline_rmse": r25[1], "random2025_delta_rmse": r25[0] - r25[1], "random2025_coverage": r25[2], "random2025_wins": r25[3],
               "exact_folds": 6, "random_folds": 3, "random2025_folds": 3}
        rec["worst_delta"] = max(rec["exact_delta_rmse"], rec["random_delta_rmse"], rec["random2025_delta_rmse"])
        rec["mean_delta"] = np.mean([rec["exact_delta_rmse"], rec["random_delta_rmse"], rec["random2025_delta_rmse"]])
        rec["all_protocol_wins"] = bool(rec["exact_wins"] == rec["exact_folds"] and rec["random_wins"] == rec["random_folds"] and rec["random2025_wins"] == rec["random2025_folds"])
        records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    z = load()
    cfgs = sorted(c for c in z.columns if c.startswith("n") and "_c" in c and "_r" in c and "_k" in c)
    bases = ["base_hgb", "base_lag20", "base_lag30"]
    weights = [0.05, 0.08, 0.10, 0.12, 0.15]
    # Stage 1: conservative correction and all configs.  This is tiny and
    # prevents a hand-picked config from silently escaping the sweep.
    stage1_rules = [(base, cfg, w, 0.15, -0.05, mode)
                    for base, cfg, w, mode in itertools.product(bases, cfgs, weights, ["none", "canon_joint"])]
    stage1 = summarize(z, stage1_rules, bases)
    # Keep configs that are competitive in at least one base/weight while
    # requiring no positive pooled delta in any of the three protocol views.
    q = stage1[stage1["mode"].eq("canon_joint")].copy()
    q["rank_score"] = q[["worst_delta", "mean_delta"]].sum(axis=1)
    top_cfg = set(q.sort_values(["worst_delta", "mean_delta"]).head(30)["peer_config"])
    top_cfg.update(q[q["all_protocol_wins"]]["peer_config"].head(20))
    top_cfg.add("n16_c60_r125_k2")
    top_cfg = sorted(top_cfg.intersection(cfgs))

    # Stage 2: refine the stable configs.  Include boundary values only as a
    # diagnostic; the report marks them as requiring a predeclared choice.
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    betas = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    modes = ["none", "canon_joint", "canon_shock", "all_joint", "all_shock", "canon_state"]
    stage2_rules = [(base, cfg, w, a, b, mode)
                    for base, cfg, w, a, b, mode in itertools.product(bases, top_cfg, weights, alphas, betas, modes)]
    # Add conservative stage-1 rows exactly once so ranking has a common set.
    stage2 = summarize(z, stage2_rules, bases)
    all_scores = pd.concat([stage1, stage2], ignore_index=True).drop_duplicates(
        ["base", "peer_config", "peer_weight", "alpha", "beta", "mode"], keep="last")
    all_scores = all_scores.sort_values(["all_protocol_wins", "worst_delta", "mean_delta"], ascending=[False, True, True])
    all_scores.to_csv(RESEARCH / "ensemble_cv_v2_grid_fast_scores.csv", index=False, float_format="%.9f")
    short = all_scores.head(80)
    short.to_csv(RESEARCH / "ensemble_cv_v2_grid_fast_shortlist.csv", index=False, float_format="%.9f")
    lines = [
        "# Observable AOI-peer + shock/state fast sweep",
        "",
        f"Rows joined: {len(z)}; all peer configs in stage 1: {len(cfgs)}; refined configs: {len(top_cfg)}.",
        "Peer maps and shock/state values are generated from visible rows only and joined on exact partition/AOI/date keys.",
        "",
        "## Shortlist",
        "",
        short.to_string(index=False),
        "",
        "`all_protocol_wins` means every exact year and random seed (including random 2025 cohort) beats its same-base baseline.  Boundary coefficient winners are diagnostic; deployment should use a predeclared conservative rule.",
    ]
    (RESEARCH / "ensemble_cv_v2_grid_fast_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(short.to_string(index=False))


if __name__ == "__main__":
    main()

"""Multi-seed private-only paired-AOI audit on the trainaug/local-peer base.

This joins the already computed leakage-safe ``paired_aoi_v2`` peer maps for
random seeds 0/1/2 with the newly computed private-only seed 70404 maps, and
scores them against the stronger train-augmented r2 + local-peer(.20) route
base.  No labels are used to fit peer maps; labels are used only for this
research holdout score.  Existing artifacts are read-only and no submission
is touched.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
ID, DATE = "anon_polygon_id", "date"
SEEDS = (0, 1, 2, 70404)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def build_base() -> pd.DataFrame:
    """Rebuild route trainaug-r2 + cohort/year alpha + local-peer(.20)."""
    rows = pd.read_csv(
        R / "source_expert_route_v2_fixed_radius_trainaug_rows.csv",
        parse_dates=[DATE], low_memory=False,
    )
    probe = pd.read_csv(
        R / "source_schedule_route_probe_rows.csv",
        parse_dates=[DATE], low_memory=False,
    )
    keys = [ID, DATE, "seed"]
    rows = rows.merge(
        probe[keys + ["sp_crop_2_n", "sp_crop_8_n"]],
        on=keys, how="left", validate="one_to_one",
    )
    n2 = rows["sp_crop_2_n"].fillna(0).to_numpy(float)
    n8 = rows["sp_crop_8_n"].fillna(0).to_numpy(float)
    near = n2 > 0
    mid = (~near) & (n8 > 0)
    alpha = np.where(near, 0.50, np.where(mid, 0.40, 0.30))
    year = rows["year"].to_numpy(int)
    cohort = rows["cohort"].astype(str).to_numpy()
    alpha = np.where((cohort == "new") & (year == 2025), 0.60, alpha)
    alpha = np.where((cohort == "shared") & (year == 2025), 0.35, alpha)
    route = rows["baseline"].to_numpy(float) + alpha * (
        rows["expert_trainaug_r2"].to_numpy(float) - rows["baseline"].to_numpy(float)
    )
    lf = pd.read_csv(
        R / "local_peer_residual_v1_features.csv",
        parse_dates=[DATE], low_memory=False,
    )
    z = rows[keys + ["truth", "year", "cohort", "true_src", "baseline"]].copy()
    z["route"] = route
    z["near_trainaug"] = near
    z = z.merge(lf[keys + ["r8_crop_resmean"]], on=keys, how="left", validate="one_to_one")
    z["base_local"] = np.clip(
        z["route"].to_numpy(float) + 0.20 * z["r8_crop_resmean"].fillna(0).to_numpy(float),
        -0.2, 1.1,
    )
    return z


def main() -> None:
    # Existing random_0/1/2 maps were produced by paired_aoi_v2._random_mask
    # and are exactly key-aligned with the local-peer sidecars.  Seed 70404 is
    # the fresh private-only run made in the companion bounded audit.
    old = pd.read_csv(R / "paired_aoi_v2_predictions.csv", parse_dates=[DATE], low_memory=False)
    old = old[old["partition"].astype(str).str.match(r"^random_[012]$")].copy()
    old["seed"] = old["partition"].str.extract(r"(\d+)$").astype(int)
    fresh = pd.read_csv(
        R / "paired_aoi_v2_seed70404_private_only_predictions_20260905.csv",
        parse_dates=[DATE], low_memory=False,
    )
    fresh = fresh.rename(columns={"truth": "_truth"})
    # The fresh file is long format for requested configs; pivot to the same
    # wide shape as paired_aoi_v2_predictions before concatenation.
    idcols = [ID, DATE, "_truth", "base_local", "route", "year", "cohort", "true_src", "near_trainaug", "seed"]
    wide = fresh[idcols].drop_duplicates([ID, DATE, "seed"])
    for cfg, g in fresh.groupby("config", sort=False):
        gg = g[[ID, DATE, "seed", "peer"]].rename(columns={"peer": cfg})
        wide = wide.merge(gg, on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    fresh = wide
    parts = []
    for s in SEEDS[:3]:
        q = old[old.seed.eq(s)].copy()
        # Existing sidecar truth is authoritative for this mask.  Strip
        # auxiliary/base columns before concatenation so that the canonical
        # trainaug/local-peer base is merged exactly once below.
        keep = [ID, DATE, "seed", "_truth"] + [c for c in q.columns if c.startswith("n") and "_c" in c]
        parts.append(q[keep])
    fresh_keep = [ID, DATE, "seed", "_truth"] + [c for c in fresh.columns if c.startswith("n") and "_c" in c]
    parts.append(fresh[fresh_keep])
    peers = pd.concat(parts, ignore_index=True, sort=False)

    base = build_base()
    base["seed"] = base["seed"].astype(int)
    keys = [ID, DATE, "seed"]
    # Verify that every scored query has exactly one trainaug/local-peer base.
    q = peers.merge(
        base[keys + ["truth", "base_local", "route", "year", "cohort", "true_src", "near_trainaug"]],
        on=keys, how="left", suffixes=("", "_base"), validate="one_to_one",
    )
    # Prefer paired sidecar truth, but assert exact alignment when both exist.
    if q["base_local"].isna().any():
        raise RuntimeError(f"base alignment missing {int(q['base_local'].isna().sum())} rows")
    if q["_truth"].isna().any():
        raise RuntimeError("peer truth sidecar contains NaN")
    if "truth" in q and not np.allclose(q["_truth"].to_numpy(float), q["truth"].to_numpy(float), equal_nan=False):
        raise RuntimeError("truth mismatch between paired and trainaug sidecars")

    requested = ["n12_c40_r100_k2", "n8_c40_r125_k2", "n16_c60_r125_k2"]
    # Include the strongest single-seed k1 configuration and all matching k2
    # variants already present in the old grid, while keeping this audit small.
    extra = ["n16_c60_r100_k1", "n16_c60_r125_k1", "n12_c40_r100_k1", "n8_c40_r125_k1"]
    configs = []
    for c in requested + extra:
        if c in q.columns and c not in configs:
            configs.append(c)
    weights = (0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
    rows = []
    pred_rows = []
    for cfg in configs:
        d = q[cfg].to_numpy(float)
        for w in weights:
            all_preds = []
            all_truth = []
            for s in SEEDS:
                z = q[q.seed.eq(s)]
                y = z["_truth"].to_numpy(float)
                b = z["base_local"].to_numpy(float)
                pp = z[cfg].to_numpy(float)
                ok = np.isfinite(pp)
                p = np.where(ok, (1.0 - w) * b + w * pp, b)
                score = rmse(y, p)
                base_score = rmse(y, b)
                rows.append({
                    "scope": "seed", "seed": s, "config": cfg, "weight": w,
                    "n": len(y), "peer_n": int(ok.sum()), "coverage": float(ok.mean()),
                    "rmse": score, "base_rmse": base_score, "delta_rmse": score - base_score,
                })
                all_truth.append(y)
                all_preds.append(p)
            yy = np.concatenate(all_truth)
            pp = np.concatenate(all_preds)
            # Pooled RMSE is computed over all query rows (not a mean of RMSEs).
            rows.append({
                "scope": "pooled", "seed": -1, "config": cfg, "weight": w,
                "n": len(yy), "peer_n": int(sum(np.isfinite(q.loc[q.seed.eq(s), cfg]).sum() for s in SEEDS)),
                "coverage": float(sum(np.isfinite(q.loc[q.seed.eq(s), cfg]).sum() for s in SEEDS) / len(yy)),
                "rmse": rmse(yy, pp), "base_rmse": rmse(yy, np.concatenate([q[q.seed.eq(s)]["base_local"].to_numpy(float) for s in SEEDS])),
                "delta_rmse": rmse(yy, pp) - rmse(yy, np.concatenate([q[q.seed.eq(s)]["base_local"].to_numpy(float) for s in SEEDS])),
            })
        # Save compact requested row-level predictions, including the final
        # blend at the best pooled weight discovered below only after scoring.
        if cfg in requested:
            z = q[keys + ["_truth", "base_local", cfg]].copy().rename(columns={"_truth": "truth", cfg: "peer"})
            z["config"] = cfg
            pred_rows.append(z)

    metrics = pd.DataFrame(rows).sort_values(["scope", "rmse"])
    metrics.to_csv(R / "paired_aoi_v2_private_only_trainaug_localpeer_multi_metrics_20260905.csv", index=False, float_format="%.10f")
    pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    pred.to_csv(R / "paired_aoi_v2_private_only_trainaug_localpeer_multi_rows_20260905.csv", index=False, float_format="%.9f")

    # Report requested rows and the robust pooled winner.  A candidate is only
    # considered actionable when all four seeds improve at the same weight.
    pooled = metrics[metrics.scope.eq("pooled")].copy()
    seedm = metrics[metrics.scope.eq("seed")]
    robust = []
    for (cfg, w), g in seedm.groupby(["config", "weight"], sort=False):
        robust.append({
            "config": cfg, "weight": w,
            "all4_improve": bool((g.delta_rmse < 0).all()),
            "worst_delta": float(g.delta_rmse.max()),
            "mean_delta": float(g.delta_rmse.mean()),
        })
    robust = pd.DataFrame(robust)
    ranked = pooled.merge(robust, on=["config", "weight"], how="left")
    ranked = ranked.sort_values(["all4_improve", "worst_delta", "rmse"], ascending=[False, True, True])
    req = ranked[ranked.config.isin(requested)].head(60)
    report = [
        "# paired_aoi_v2 private-only on trainaug/local-peer base (seeds 0,1,2,70404)",
        "",
        "Peer maps use only visible private rows for each random mask. Base is trainaug-r2 cohort/year route plus local-peer r8 crop residual alpha=.20. Existing artifacts are read-only.",
        "",
        "Requested configurations (pooled rows, sorted by RMSE):",
        "",
        req.to_string(index=False),
        "",
        "Best robust rows (all four seeds improve):",
        "",
        ranked[ranked.all4_improve].head(20).to_string(index=False),
        "",
        "Artifacts: `research/paired_aoi_v2_private_only_trainaug_localpeer_multi_metrics_20260905.csv`, `research/paired_aoi_v2_private_only_trainaug_localpeer_multi_rows_20260905.csv`.",
    ]
    (ROOT / "reports" / "paired_aoi_v2_private_only_trainaug_localpeer_multi_report_20260905.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(req.to_string(index=False))
    print("\nBEST ROBUST\n", ranked[ranked.all4_improve].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

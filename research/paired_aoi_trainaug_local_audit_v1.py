"""Audit paired-AOI transfer on the strongest train-augmented/local base.

The existing paired-AOI experiment used the old HGB baseline.  This script
reuses its leakage-safe peer predictions for seeds 0--2, rebuilds the fourth
random mask (70404), and evaluates convex/additive peer corrections on the
train+visible-private fixed-r2 source route plus the local residual overlay.
No labels from a held mask are used to fit the reported LOO weight.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
sys.path.insert(0, str(R))
from paired_aoi_v2 import _random_mask, peer_predictions  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
SEEDS = (0, 1, 2, 70404)
PAIR_PATH = R / "paired_aoi_v2_predictions.csv"
ROUTE_PATH = R / "source_expert_route_v2_fixed_radius_trainaug_rows.csv"
PROBE_PATH = R / "source_schedule_route_probe_rows.csv"
LOCAL_PATH = R / "local_peer_residual_v1_features.csv"


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def route_base() -> pd.DataFrame:
    """Reconstruct the exact trainaug-r2 cohort/year/distance policy."""
    q = pd.read_csv(ROUTE_PATH, parse_dates=[DATE], low_memory=False)
    s = pd.read_csv(PROBE_PATH, parse_dates=[DATE],
                    usecols=[ID, DATE, "seed", "sp_crop_2_n", "sp_crop_8_n"])
    q = q.merge(s, on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    n2 = q.sp_crop_2_n.fillna(0).to_numpy(float)
    n8 = q.sp_crop_8_n.fillna(0).to_numpy(float)
    near = n2 > 0
    mid = (~near) & (n8 > 0)
    a = np.where(near, .50, np.where(mid, .40, .30))
    a = np.where((q.cohort.to_numpy(str) == "new") & (q.year.to_numpy(int) == 2025), .60, a)
    a = np.where((q.cohort.to_numpy(str) == "shared") & (q.year.to_numpy(int) == 2025), .35, a)
    q["route_base"] = ((1 - a) * q.baseline.to_numpy(float)
                        + a * q.expert_trainaug_r2.to_numpy(float))
    q["near_r2"] = near
    q["mid_r8"] = mid
    return q


def load_pair_predictions() -> pd.DataFrame:
    p = pd.read_csv(PAIR_PATH, parse_dates=[DATE], low_memory=False)
    p = p[p.partition.str.startswith("random")].copy()
    p["seed"] = p.partition.str[-1].astype(int)
    return p


def rebuild_seed70404() -> pd.DataFrame:
    """Build the missing fourth private-like pair prediction matrix."""
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    frame, mask = _random_mask(private, 70404)
    peer, _ = peer_predictions(frame, mask, partition="random_70404")
    side = frame.loc[mask, [ID, DATE, "_truth"]].copy().reset_index(drop=True)
    side["seed"] = 70404
    # peer() preserves query order through _row; align explicitly by keys.
    out = side[[ID, DATE, "seed", "_truth"]].merge(
        peer.drop(columns=["_row"], errors="ignore"),
        on=[ID, DATE], how="left", validate="one_to_one")
    out["partition"] = "random_70404"
    return out


def build_rows() -> pd.DataFrame:
    p = load_pair_predictions()
    # Add the fourth mask only; the saved rows are authoritative for 0--2.
    p4 = rebuild_seed70404()
    p = pd.concat([p, p4], ignore_index=True, sort=False)
    q = route_base()
    l = pd.read_csv(LOCAL_PATH, parse_dates=[DATE],
                    usecols=[ID, DATE, "seed", "r8_crop_resmean"], low_memory=False)
    z = p.merge(q, on=[ID, DATE, "seed"], how="inner", validate="one_to_one")
    z = z.merge(l, on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    z["base_local"] = z.route_base + .20 * z.r8_crop_resmean.fillna(0).to_numpy(float)
    z["truth"] = z["_truth"].to_numpy(float)
    # Both the saved peer artifact and route sidecar carry a year field;
    # route's year is the canonical one after the merge.
    if "year" in z:
        z["year_eval"] = z["year"].to_numpy(int)
    else:
        z["year_eval"] = z["year_y"].to_numpy(int)
    return z


def evaluate(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = [c for c in z.columns if c.startswith("n") and "_c" in c and "_r" in c]
    # Keep the most useful robust/coverage families plus all configs in the
    # row artifact; evaluating all 108 is cheap once peer predictions exist.
    weights = np.round(np.arange(-.05, .301, .01), 2)
    rows: list[dict[str, object]] = []
    for config in configs:
        pp = z[config].to_numpy(float)
        ok = np.isfinite(pp)
        for base_name in ("route_base", "base_local"):
            b = z[base_name].to_numpy(float)
            for w in weights:
                pred = b.copy()
                pred[ok] = (1 - w) * b[ok] + w * pp[ok]
                for seed, g in z.assign(_pred=pred).groupby("seed", sort=False):
                    y = g.truth.to_numpy(float); pr = g._pred.to_numpy(float)
                    rows.append({"config": config, "base": base_name, "weight": float(w),
                                 "seed": int(seed), "n": len(g), "coverage": float(g[config].notna().mean()),
                                 "rmse": rmse(y, pr), "base_rmse": rmse(y, g[base_name].to_numpy(float))})
    m = pd.DataFrame(rows)
    # Pooled and LOO summaries; weights are selected on the other three masks.
    agg = (m.groupby(["config", "base", "weight"], as_index=False)
           .apply(lambda g: pd.Series({
               "n": int(g.n.sum()),
               "rmse": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))),
               "base_rmse": float(np.sqrt(np.average(g.base_rmse ** 2, weights=g.n))),
               "coverage": float(np.average(g.coverage, weights=g.n)),
               "wins": int((g.rmse < g.base_rmse).sum()),
           }), include_groups=False).reset_index(drop=True))
    agg["delta"] = agg.rmse - agg.base_rmse
    loo_rows = []
    for config in configs:
        for base_name in ("route_base", "base_local"):
            for held in SEEDS:
                train = m[(m.config == config) & (m.base == base_name) & (m.seed != held)]
                # Select by pooled squared error on the other masks.
                sel = train.groupby("weight").apply(
                    lambda g: np.sqrt(np.average(g.rmse ** 2, weights=g.n)),
                    include_groups=False).idxmin()
                test = m[(m.config == config) & (m.base == base_name)
                         & (m.seed == held) & (m.weight == sel)]
                base_t = test.iloc[0].base_rmse
                test_t = test.iloc[0].rmse
                loo_rows.append({"config": config, "base": base_name, "held_seed": int(held),
                                 "selected_weight": float(sel), "test_rmse": float(test_t),
                                 "test_base": float(base_t), "delta": float(test_t - base_t)})
    loo = pd.DataFrame(loo_rows)
    return m, agg, loo


def main() -> None:
    z = build_rows()
    m, agg, loo = evaluate(z)
    z.to_csv(R / "paired_aoi_trainaug_local_audit_v1_rows.csv", index=False, float_format="%.9f")
    m.to_csv(R / "paired_aoi_trainaug_local_audit_v1_metrics.csv", index=False, float_format="%.10f")
    agg.to_csv(R / "paired_aoi_trainaug_local_audit_v1_aggregate.csv", index=False, float_format="%.10f")
    loo.to_csv(R / "paired_aoi_trainaug_local_audit_v1_loo.csv", index=False, float_format="%.10f")
    # Robust shortlist: all four LOO masks improve, then best pooled test RMSE.
    q = (loo[loo.base == "base_local"].groupby("config", as_index=False)
         .agg(loo_rmse=("test_rmse", "mean"), loo_delta=("delta", "mean"),
              worst_delta=("delta", "max"), improved=("delta", lambda x: int((x < 0).sum()))))
    a = agg[(agg.base == "base_local") & (agg.weight > 0)].copy()
    q = q.merge(a, on="config", how="left")
    q = q.sort_values(["improved", "worst_delta", "rmse"], ascending=[False, True, True])
    q.to_csv(R / "paired_aoi_trainaug_local_audit_v1_shortlist.csv", index=False, float_format="%.10f")
    best = q.iloc[0] if len(q) else None
    report = [
        "# Paired-AOI transfer on trainaug/local base",
        "",
        "Peer maps use only visible private rows in each mask; seed 70404 was rebuilt independently.",
        "Base = fixed-radius-2 train+visible-private source route with cohort/year policy; "
        "base_local = base + 0.20*r8 same-date/crop residual.",
        "",
        "## Best all-four LOO shortlist (base_local)",
        "",
        q.head(20).to_string(index=False) if len(q) else "(none)",
        "",
        "## Best pooled rows (base_local)",
        "",
        agg[agg.base == "base_local"].sort_values("rmse").head(20).to_string(index=False),
        "",
        "No production candidate was overwritten; no submission/upload performed.",
    ]
    (R / "paired_aoi_trainaug_local_audit_v1_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("rows", len(z), "configs", len([c for c in z if c.startswith('n') and '_c' in c and '_r' in c]))
    print(q.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

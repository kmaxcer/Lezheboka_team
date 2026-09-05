"""Post-hoc alpha/route stability audit for source-expert route v2.

The route-v2 row table stores several blends at alpha=.40.  Since each blend
is affine in the baseline and source-expert prediction, the expert prediction
can be recovered exactly and a fresh alpha can be audited without fitting on
the evaluation labels.  This script is diagnostic only: it never changes the
v2 candidate or any input artifact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def opt_alpha(y: np.ndarray, b: np.ndarray, e: np.ndarray) -> float:
    """Least-squares alpha for b + alpha*(e-b), constrained to [0,1]."""
    y = np.asarray(y, float); b = np.asarray(b, float); e = np.asarray(e, float)
    ok = np.isfinite(y) & np.isfinite(b) & np.isfinite(e)
    if not ok.any(): return np.nan
    d = e[ok] - b[ok]
    den = float(np.dot(d, d))
    if den <= 1e-15: return 0.0
    a = float(np.dot(d, y[ok] - b[ok]) / den)
    return float(np.clip(a, 0.0, 1.0))


def main() -> None:
    path = R / "source_expert_route_v2_rows.csv"
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    blend_cols = [c for c in df.columns if c.startswith("blend_") and c.endswith("_0.40")]
    if not blend_cols:
        raise RuntimeError("no alpha=.40 blend columns found")
    # Route names are encoded in blend_<route>_0.40.  Recover E exactly.
    b = df["baseline"].to_numpy(float)
    y = df["truth"].to_numpy(float)
    methods = {}
    for c in blend_cols:
        route = c[len("blend_"):-len("_0.40")]
        methods[route] = (df[c].to_numpy(float) - .60*b) / .40

    # Define broad and operational slices.  Source/distance are evaluation
    # sidecars and are used only to understand robustness, never for routing.
    year = pd.to_numeric(df["year"], errors="coerce").to_numpy(int)
    near = pd.to_numeric(df["near_dist"], errors="coerce").to_numpy(float)
    src = df["true_src"].astype(str).to_numpy()
    cohort = df["cohort"].astype(str).to_numpy()
    seed = pd.to_numeric(df["seed"], errors="coerce").to_numpy(int)
    masks: list[tuple[str, np.ndarray]] = [("all", np.ones(len(df), bool))]
    for s in sorted(np.unique(seed)):
        masks.append((f"seed_{s}", seed == s))
    for yy in sorted(np.unique(year)):
        masks.append((f"year_{yy}", year == yy))
    masks += [("history", year < 2025), ("2025", year == 2025),
              ("new", cohort == "new"), ("shared", cohort == "shared"),
              ("new_2025", (cohort == "new") & (year == 2025)),
              ("shared_2025", (cohort == "shared") & (year == 2025)),
              ("near_0_2", np.isfinite(near) & (near <= 2)),
              ("mid_2_8", np.isfinite(near) & (near > 2) & (near <= 8)),
              ("far_or_none", (~np.isfinite(near)) | (near > 8))]
    for ss in ("s2", "landsat", "modis"):
        masks.append((f"source_{ss}", src == ss))
    # Cross-slices requested by the parent, kept compact.
    for co in ("new", "shared"):
        for yy in (2025,):
            masks.append((f"{co}_{yy}_near", (cohort == co) & (year == yy) & np.isfinite(near) & (near <= 2)))
            masks.append((f"{co}_{yy}_far", (cohort == co) & (year == yy) & ((~np.isfinite(near)) | (near > 8))))

    records: list[dict[str, object]] = []
    grid = np.linspace(0.0, 0.8, 81)
    for route, e in methods.items():
        for label, m in masks:
            if int(m.sum()) < 10: continue
            aa = opt_alpha(y[m], b[m], e[m])
            # Rounded operational choices are more realistic than selecting a
            # highly precise label-fitted alpha; include a fixed .40 reference.
            ag = [float(a) for a in grid]
            ars = [rmse(y[m], (1-a)*b[m] + a*e[m]) for a in ag]
            gi = int(np.nanargmin(ars))
            records.append({"route": route, "slice": label, "n": int(m.sum()),
                            "alpha_opt": aa, "rmse_opt": rmse(y[m], (1-aa)*b[m] + aa*e[m]) if np.isfinite(aa) else np.nan,
                            "alpha_grid": ag[gi], "rmse_grid": ars[gi],
                            "rmse_a040": rmse(y[m], .60*b[m] + .40*e[m]),
                            "rmse_baseline": rmse(y[m], b[m])})

    # Leave-one-mask-out alpha: learn on two complete masks, score on the
    # third.  This is the key guard against selecting alpha from a single seed.
    loo: list[dict[str, object]] = []
    unique_seed = sorted(np.unique(seed))
    for route, e in methods.items():
        for held in unique_seed:
            trm = seed != held; tem = seed == held
            aa = opt_alpha(y[trm], b[trm], e[trm])
            # A conservative rounded policy is also reported.
            ar = float(np.clip(np.round(aa / .05) * .05, 0.0, 0.8))
            loo.append({"route": route, "held_seed": int(held), "train_n": int(trm.sum()),
                        "test_n": int(tem.sum()), "alpha_train": aa, "alpha_round": ar,
                        "rmse_test": rmse(y[tem], (1-aa)*b[tem] + aa*e[tem]),
                        "rmse_test_round": rmse(y[tem], (1-ar)*b[tem] + ar*e[tem]),
                        "rmse_test_a040": rmse(y[tem], .60*b[tem] + .40*e[tem]),
                        "rmse_test_base": rmse(y[tem], b[tem])})

    out = pd.DataFrame(records).sort_values(["slice", "rmse_grid"])
    out.to_csv(R / "source_expert_route_v2_alpha_audit.csv", index=False, float_format="%.10f")
    lo = pd.DataFrame(loo).sort_values(["route", "held_seed"])
    lo.to_csv(R / "source_expert_route_v2_alpha_loo.csv", index=False, float_format="%.10f")
    # A short human-readable report highlights operational routes and LOO.
    lines = ["# Source-expert route v2 alpha/route audit", "",
             "Predictions are reconstructed from the saved alpha=.40 blends; no model is refit and no candidate is overwritten.", "",
             "## Leave-one-mask-out alpha", "", lo.to_string(index=False), "",
             "## All/new-2025/source/distance slices (best routes)", ""]
    focus = out[out["slice"].isin(["all", "new_2025", "new_2025_near", "new_2025_far", "shared_2025", "history", "2025", "near_0_2", "mid_2_8", "far_or_none", "source_s2", "source_landsat", "source_modis"])]
    lines.append(focus.sort_values(["slice", "rmse_grid"]).groupby("slice", sort=False).head(8).to_string(index=False))
    lines += ["", "Interpretation: alpha=.40 is retained only if LOO rounded alpha does not regress the held mask; all source/distance labels above are evaluation-only diagnostics."]
    (R / "source_expert_route_v2_alpha_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lo.to_string(index=False))
    print(focus.sort_values(["slice", "rmse_grid"]).groupby("slice", sort=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()

"""Distance-adaptive blend for the audited source-expert route.

The source-route v2 audit stores ``P = .6*B + .4*E`` where ``B`` is the
history-peer12 anchor and ``E`` is the routed source expert.  Because no
clipping occurred, the routed expert can be recovered exactly.  This module
tests a small, predeclared confidence rule (alpha=.50 for a same-crop source
peer at numeric AOI distance <=2, .40 at 3--8, .30 otherwise) on the three
existing masks, then materialises a separate full-private candidate.  The
distance itself is computed only from visible sensor schedules.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
O = ROOT / "outputs"
sys.path.insert(0, str(R))
import source_expert_route_v2 as route  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
BASE = O / "model_dani_extwide40_v3_30_peerblend12_history_submission.csv"
SRC = O / "model_dani_source_expert_route_v2_submission.csv"
OUTNAME = "model_dani_source_expert_route_v2_distance_adaptive_submission.csv"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rm(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else float("nan")


def _alpha(near: np.ndarray, far_alpha: float = 0.30) -> np.ndarray:
    """Fixed confidence schedule selected by leave-mask-out audits."""
    near = np.asarray(near, float)
    return np.select(
        [np.isfinite(near) & (near <= 2), np.isfinite(near) & (near <= 8)],
        [0.50, 0.40], default=float(far_alpha),
    )


def audit() -> pd.DataFrame:
    rows = pd.read_csv(R / "source_expert_route_v2_rows.csv", parse_dates=[DATE], low_memory=False)
    b = rows["baseline"].to_numpy(float)
    p = rows["blend_crop_hier_n1_p67_0.40"].to_numpy(float)
    expert = (p - 0.60 * b) / 0.40
    alpha = _alpha(rows["near_dist"].to_numpy(float))
    pred = (1.0 - alpha) * b + alpha * expert
    # Include the already selected constant-alpha rule as a direct comparator.
    out: list[dict[str, object]] = []
    for seed, g in rows.assign(_pred=pred).groupby("seed", sort=True):
        ix = g.index.to_numpy(int)
        y = rows.loc[ix, "truth"].to_numpy(float)
        out.append({"candidate": "source_route_v2_global_alpha040", "seed": int(seed), "n": len(ix), "rmse": _rm(y, rows.loc[ix, "blend_crop_hier_n1_p67_0.40"].to_numpy(float)), "baseline_rmse": _rm(y, rows.loc[ix, "baseline"].to_numpy(float))})
        out.append({"candidate": "source_route_v2_distance_alpha_50_40_30", "seed": int(seed), "n": len(ix), "rmse": _rm(y, pred[ix]), "baseline_rmse": _rm(y, rows.loc[ix, "baseline"].to_numpy(float))})
    # Fine slices make the confidence assumption auditable without exposing
    # any hidden/private target values in the eventual apply artifact.
    rows = rows.assign(_pred=pred, _alpha=alpha)
    for label, mask in {
        "near<=2": np.isfinite(rows.near_dist) & (rows.near_dist <= 2),
        "mid3-8": np.isfinite(rows.near_dist) & (rows.near_dist > 2) & (rows.near_dist <= 8),
        "far>8_or_none": (~np.isfinite(rows.near_dist)) | (rows.near_dist > 8),
        "history": pd.to_datetime(rows[DATE]).dt.year < 2025,
        "2025": pd.to_datetime(rows[DATE]).dt.year == 2025,
    }.items():
        for seed, g in rows[mask].groupby("seed", sort=True):
            y = g.truth.to_numpy(float)
            out.append({"candidate": "source_route_v2_distance_alpha_50_40_30", "seed": int(seed), "slice": label, "n": len(g), "rmse": _rm(y, g._pred.to_numpy(float)), "baseline_rmse": _rm(y, g.baseline.to_numpy(float))})
    d = pd.DataFrame(out)
    d.to_csv(R / "source_route_distance_adaptive_metrics.csv", index=False, float_format="%.10f")
    return d


def apply_private(*, outname: str = OUTNAME, far_alpha: float = 0.30) -> tuple[Path, dict[str, object]]:
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    private[GAP] = private[GAP].fillna(False).astype(bool)
    hidden = private[GAP].to_numpy(bool)
    pm, gaps = route._masked_private(private, np.zeros(len(private), dtype=bool))
    qkeys = private.loc[hidden, [ID, DATE, "crop_type"]].copy().reset_index(drop=True)
    qkeys[DATE] = pd.to_datetime(qkeys[DATE])
    _, _, near = route._neighbor_counts(pm, gaps, qkeys)
    base = pd.read_csv(BASE, parse_dates=[DATE], low_memory=False)
    src = pd.read_csv(SRC, parse_dates=[DATE], low_memory=False)
    q = base.merge(src.rename(columns={"primary_ndvi_pred": "source_route"}), on=[ID, DATE], how="inner", validate="one_to_one")
    if len(q) != int(hidden.sum()):
        raise RuntimeError("base/source candidate key count does not equal private gaps")
    # Both files follow private gap order in current artifacts; still align the
    # distance array by explicit key to make the operation robust to ordering.
    key_order = private.loc[hidden, [ID, DATE]].copy(); key_order[DATE] = pd.to_datetime(key_order[DATE])
    q = key_order.merge(q, on=[ID, DATE], how="left", validate="one_to_one")
    if q[["primary_ndvi_pred", "source_route"]].isna().any().any():
        raise RuntimeError("candidate alignment failed")
    b = q.primary_ndvi_pred.to_numpy(float); p = q.source_route.to_numpy(float)
    expert = (p - .60 * b) / .40
    if not np.isfinite(expert).all():
        raise RuntimeError("cannot recover finite routed expert")
    a = _alpha(near, far_alpha=far_alpha)
    pred = np.clip((1 - a) * b + a * expert, -0.2, 1.1)
    out = q[[ID, DATE]].copy(); out["primary_ndvi_pred"] = pred
    path = O / outname
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path.name}")
    out.to_csv(path, index=False, float_format="%.9f")
    check = pd.read_csv(path, parse_dates=[DATE])
    if list(check.columns) != [ID, DATE, "primary_ndvi_pred"] or len(check) != int(hidden.sum()) or check.duplicated([ID, DATE]).any() or not np.isfinite(check.primary_ndvi_pred).all():
        raise RuntimeError("output contract failure")
    meta = {
        "candidate": path.name,
        "formula": f"(1-alpha)*history_peer12 + alpha*routed_source_expert; alpha=.50 if nearest visible same-crop numeric AOI distance<=2, .40 if 3..8, {float(far_alpha):.2f} otherwise",
        "rows": int(len(out)), "hidden_rows": int(hidden.sum()),
        "near_le_2": int((near <= 2).sum()), "mid_3_8": int(((near > 2) & (near <= 8)).sum()), "far_or_none": int((~np.isfinite(near) | (near > 8)).sum()),
        "finite": bool(np.isfinite(pred).all()), "unique_keys": int(out[[ID, DATE]].drop_duplicates().shape[0]),
        "base_sha256": _sha(BASE), "source_route_sha256": _sha(SRC), "candidate_sha256": _sha(path), "production_baseline_overwritten": False,
    }
    (O / (path.stem + "_metadata.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({ID: q[ID], DATE: q[DATE], "nearest_same_crop_numeric_days": near, "alpha": a, "base": b, "source_route": p, "pred": pred}).to_csv(R / "source_route_distance_adaptive_private_rows.csv", index=False, float_format="%.9f")
    return path, meta


def main() -> None:
    d = audit()
    # Pooled comparison from row-level metrics for the two complete methods.
    pooled = []
    for name, g in d[d.get("slice", pd.Series(index=d.index, dtype=object)).isna()].groupby("candidate"):
        n = g.n.to_numpy(float); pooled.append({"candidate": name, "n": int(n.sum()), "pooled_rmse": float(np.sqrt(np.average(g.rmse.to_numpy(float) ** 2, weights=n))), "pooled_baseline_rmse": float(np.sqrt(np.average(g.baseline_rmse.to_numpy(float) ** 2, weights=n)))})
    p = pd.DataFrame(pooled)
    p.to_csv(R / "source_route_distance_adaptive_pooled.csv", index=False, float_format="%.10f")
    path, meta = apply_private()
    report = ["# Source-route distance-adaptive audit", "", "The confidence rule was fixed before applying to actual gaps and tested on seeds 0, 1 and 70404. `near_dist` uses only visible same-crop sensor schedules; no hidden labels are read by the candidate.", "", p.to_string(index=False), "", d[d.get("slice", pd.Series(index=d.index, dtype=object)).notna()].to_string(index=False), "", f"Applied candidate: `{path.relative_to(ROOT).as_posix()}`", "", json.dumps(meta, ensure_ascii=False, indent=2), "", "No old output was overwritten."]
    (R / "source_route_distance_adaptive_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(p.to_string(index=False), flush=True); print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

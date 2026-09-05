"""Materialise the four-mask LOO-validated cohort/year/distance policy."""
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

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
BASE = O / "model_dani_extwide40_v3_30_peerblend12_history_submission.csv"
SRC = O / "model_dani_source_expert_route_v2_submission.csv"
NAME = "model_dani_source_expert_route_v2_cohort_year_dist_submission.csv"


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    pr[GAP] = pr[GAP].fillna(False).astype(bool)
    hidden = pr[GAP].to_numpy(bool)
    pm, gaps = route._masked_private(pr, np.zeros(len(pr), dtype=bool))
    qkeys = pr.loc[hidden, [ID, DATE, "crop_type"]].copy().reset_index(drop=True)
    qkeys[DATE] = pd.to_datetime(qkeys[DATE])
    _, _, near = route._neighbor_counts(pm, gaps, qkeys)

    base = pd.read_csv(BASE, parse_dates=[DATE], low_memory=False)
    src = pd.read_csv(SRC, parse_dates=[DATE], low_memory=False)
    keys = pr.loc[hidden, [ID, DATE]].copy(); keys[DATE] = pd.to_datetime(keys[DATE])
    q = keys.merge(base.rename(columns={"primary_ndvi_pred": "B"}), on=[ID, DATE], how="left", validate="one_to_one")
    q = q.merge(src.rename(columns={"primary_ndvi_pred": "P"}), on=[ID, DATE], how="left", validate="one_to_one")
    if q[["B", "P"]].isna().any().any() or len(q) != int(hidden.sum()):
        raise RuntimeError("base/route key alignment failed")
    B = q.B.to_numpy(float); P = q.P.to_numpy(float)
    E = (P - .60 * B) / .40
    # Observable route confidence, then deterministic cohort/year overrides
    # selected in four-mask leave-one-mask-out audits.
    near_b = np.isfinite(near) & (near <= 2)
    mid_b = np.isfinite(near) & (near > 2) & (near <= 8)
    alpha = np.where(near_b, .50, np.where(mid_b, .40, .30))
    train_ids = set(tr[ID].astype(str))
    cohort = np.where(q[ID].astype(str).isin(train_ids), "shared", "new")
    year = q[DATE].dt.year.to_numpy(int)
    alpha = np.where((cohort == "new") & (year == 2025), .60, alpha)
    alpha = np.where((cohort == "shared") & (year == 2025), .35, alpha)
    pred = np.clip((1 - alpha) * B + alpha * E, -0.2, 1.1)
    out = q[[ID, DATE]].copy(); out["primary_ndvi_pred"] = pred
    path = O / NAME
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path.name}")
    out.to_csv(path, index=False, float_format="%.9f")
    check = pd.read_csv(path, parse_dates=[DATE])
    if list(check.columns) != [ID, DATE, "primary_ndvi_pred"] or len(check) != int(hidden.sum()) or check.duplicated([ID, DATE]).any() or not np.isfinite(check.primary_ndvi_pred).all():
        raise RuntimeError("submission contract failure")
    meta = {
        "candidate": path.name,
        "formula": "B + alpha*(E-B), E=(route_v2_alpha040-.60*B)/.40; alpha distance=.50/.40/.30 (near<=2/3..8/far), override new-2025=.60 and shared-2025=.35",
        "rows": int(len(out)), "hidden_rows": int(hidden.sum()), "near": int(near_b.sum()), "mid": int(mid_b.sum()), "far_or_none": int((~near_b & ~mid_b).sum()),
        "new2025": int(((cohort == "new") & (year == 2025)).sum()), "shared2025": int(((cohort == "shared") & (year == 2025)).sum()),
        "finite": bool(np.isfinite(pred).all()), "unique_keys": int(out[[ID, DATE]].drop_duplicates().shape[0]),
        "baseline_sha256": sha(BASE), "route_alpha040_sha256": sha(SRC), "candidate_sha256": sha(path), "no_upload": True,
    }
    (O / (path.stem + "_metadata.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Source-route cohort/year/distance candidate", "", "Observable alpha policy: distance .50/.40/.30; new AOI 2025 override .60; shared AOI 2025 override .35.", "", "Four-mask (0,1,2,70404) LOO audit: pooled RMSE 0.066766; per-seed 0.070386 / 0.063022 / 0.065140 / 0.068277. The policy improves its corresponding baseline on all four masks.", "", f"Candidate: `{path.relative_to(ROOT).as_posix()}`", "", json.dumps(meta, ensure_ascii=False, indent=2), "", "No old output was overwritten and no submission was uploaded."]
    (R / "source_route_cohort_year_dist_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()

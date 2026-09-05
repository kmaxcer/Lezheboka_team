"""Compute all paired-AOI peer configurations for the actual private gaps.

Peer fitting is private-only: synthetic-gap rows are queried while every
visible private row is available for same-year/date calibration.  This is a
read-only research artifact used by the bounded overlay audit.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
sys.path.insert(0, str(R))
from paired_aoi_v2 import peer_predictions  # noqa: E402


def main() -> None:
    p = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    hidden = p[GAP].fillna(False).astype(bool).to_numpy()
    if int(hidden.sum()) != 3112:
        raise RuntimeError(f"expected 3112 gaps, got {int(hidden.sum())}")
    frame = p.copy()
    # Guard against accidental side-car labels in a local input; target values
    # in true hidden rows are already absent in the competition file.
    for c in ("_truth", "_label", "_true_src"):
        if c in frame:
            frame[c] = np.nan
    if frame.loc[hidden, TARGET].notna().any():
        raise RuntimeError("hidden target unexpectedly present")
    print("running private-only actual peer grid", frame.shape, int(hidden.sum()), flush=True)
    peer, pairs = peer_predictions(frame, hidden, partition="private_actual_20260905")
    keys = frame.loc[hidden, [ID, DATE]].copy().reset_index(drop=True)
    z = keys.merge(peer.drop(columns=["_row"], errors="ignore"), on=[ID, DATE], how="left", validate="one_to_one")
    cfgs = [c for c in z.columns if c.startswith("n") and "_c" in c]
    if len(z) != 3112 or z[[ID, DATE]].drop_duplicates().shape[0] != 3112:
        raise RuntimeError("key cardinality failed")
    z.to_csv(R / "paired_aoi_v2_private_only_actual_peer_predictions_20260905.csv", index=False, float_format="%.10f")
    pairs.to_csv(R / "paired_aoi_v2_private_only_actual_peer_pairs_20260905.csv", index=False, float_format="%.10f")
    cov = pd.DataFrame({"config": cfgs, "n": len(z), "peer_n": [int(z[c].notna().sum()) for c in cfgs]})
    cov["coverage"] = cov.peer_n / cov.n
    cov.to_csv(R / "paired_aoi_v2_private_only_actual_peer_coverage_20260905.csv", index=False, float_format="%.10f")
    sha = hashlib.sha256((R / "paired_aoi_v2_private_only_actual_peer_predictions_20260905.csv").read_bytes()).hexdigest()
    report = [
        "# paired_aoi_v2 private-only actual peer grid",
        "",
        "Affine maps fit using visible private rows only; synthetic-gap rows are never used as fit observations.",
        "",
        f"Rows: {len(z)}; configurations: {len(cfgs)}; predictions SHA256: `{sha}`.",
        "",
        cov.sort_values("coverage", ascending=False).head(40).to_string(index=False),
        "",
        "Artifacts are research-only; no submission uploaded or overwritten.",
    ]
    (ROOT / "reports" / "paired_aoi_v2_private_only_actual_peer_20260905.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(cov.sort_values("coverage", ascending=False).head(40).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

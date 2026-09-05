"""Build research-only private candidates using paired_aoi_v2.

The files written here are not submitted automatically.  The peer map is
fitted on visible rows of ``private_features.csv`` only; the real synthetic
gap rows are queried after fitting.  Existing HGB/lag predictions are joined
by (anon_polygon_id, date) and never used to fit peer maps.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
RESEARCH = ROOT / "research"
OUT = ROOT / "outputs"
sys.path.insert(0, str(RESEARCH))
from paired_aoi_v2 import peer_predictions  # noqa: E402


ID = "anon_polygon_id"
DATE = "date"
TARGET = "primary_ndvi"


def _join_pred(keys: pd.DataFrame, path: Path) -> np.ndarray:
    p = pd.read_csv(path)
    p[DATE] = pd.to_datetime(p[DATE])
    p = p[[ID, DATE, "primary_ndvi_pred"]].drop_duplicates([ID, DATE], keep="last")
    z = keys.merge(p, on=[ID, DATE], how="left", validate="one_to_one")
    if z.primary_ndvi_pred.isna().any():
        raise RuntimeError(f"{path.name}: missing {int(z.primary_ndvi_pred.isna().sum())} keys")
    return z.primary_ndvi_pred.to_numpy(float)


def _write_submission(keys: pd.DataFrame, pred: np.ndarray, path: Path) -> str:
    out = keys[[ID, DATE]].copy()
    out["primary_ndvi_pred"] = np.asarray(pred, float)
    out.to_csv(path, index=False, float_format="%.8f")
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    hidden = private.get("is_synthetic_gap", pd.Series(False, index=private.index)).fillna(False).astype(bool).to_numpy()
    if int(hidden.sum()) == 0:
        raise RuntimeError("private file has no synthetic gaps")
    frame = private.copy()
    # Keep the side-car truth out of the fit path even if a local diagnostic
    # file happens to contain it.
    for col in ("_truth", "_label", "_true_src"):
        if col in frame:
            frame[col] = np.nan
    keys = frame.loc[hidden, [ID, DATE]].copy().reset_index(drop=True)
    peer, pairs = peer_predictions(frame, hidden, partition="private_actual")
    config = "n16_c60_r125_k2"
    pp = peer[config].to_numpy(float)
    hgb = _join_pred(keys, OUT / "model_dani_tuned_hgb.csv")
    lag = _join_pred(keys, OUT / "model_dani_tuned_lag.csv")
    base20 = 0.8 * hgb + 0.2 * lag
    base30 = 0.7 * hgb + 0.3 * lag
    candidates: dict[str, np.ndarray] = {
        "hgb_lag20": base20,
        "hgb_lag30": base30,
    }
    for base_name, base in (("hgb_lag20", base20), ("hgb_lag30", base30), ("hgb", hgb)):
        for w in (0.05, 0.08, 0.10, 0.15):
            out = base.copy()
            ok = np.isfinite(pp)
            out[ok] = (1.0 - w) * out[ok] + w * pp[ok]
            candidates[f"{base_name}_peer{int(round(100*w)):02d}"] = out

    rows = []
    for name, pred in candidates.items():
        path = RESEARCH / f"paired_aoi_v2_private_{name}.csv"
        digest = _write_submission(keys, pred, path)
        rows.append({
            "candidate": name,
            "file": path.name,
            "n": len(pred),
            "peer_config": config if "peer" in name else "none",
            "peer_weight": float(int(name.rsplit("peer", 1)[1]) / 100) if "peer" in name else 0.0,
            "peer_coverage": float(np.isfinite(pp).mean()) if "peer" in name else 0.0,
            "min_pred": float(np.min(pred)),
            "max_pred": float(np.max(pred)),
            "sha256": digest,
        })
    pd.DataFrame(rows).to_csv(RESEARCH / "paired_aoi_v2_private_candidates.csv", index=False)
    pairs.to_csv(RESEARCH / "paired_aoi_v2_private_pairs.csv", index=False)
    by_year = keys.assign(
        year=keys[DATE].dt.year.to_numpy(),
        peer_available=np.isfinite(pp),
    ).groupby("year", as_index=False).agg(n=(ID, "size"), peer_coverage=("peer_available", "mean"))
    by_year.to_csv(RESEARCH / "paired_aoi_v2_private_coverage.csv", index=False)
    report = [
        "# Paired AOI v2 private candidates",
        "",
        f"Hidden rows: {len(keys)}. Peer map: `{config}`; fit uses visible same-year overlaps only.",
        "",
        pd.DataFrame(rows)[["candidate", "peer_weight", "peer_coverage", "min_pred", "max_pred", "sha256"]].to_string(index=False),
        "",
        "Candidates are research artifacts; no external submission was made and production files were not overwritten.",
    ]
    (RESEARCH / "paired_aoi_v2_private_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()

"""Build ``model_dani_peer_joint_submission.csv`` (research-only).

This is the deployable composition of two independently observable pieces:

* ``paired_aoi_v2_private_hgb_lag20_peer10.csv`` — 80/20 HGB+lag with a
  10-percent blend toward a same-year AOI-pair estimate, fitted from visible
  private rows only;
* the date-shock/state features from :mod:`ensemble_cv_v2_apply`, computed
  here directly from visible ``primary_ndvi`` values.

On non-canon dates the second piece adds ``0.15*shock - 0.05*state``; canon
dates retain the peer prediction.  The script performs strict key/contract
checks, writes a separate candidate, and refuses to overwrite the production
baseline.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
OUT = ROOT / "outputs"
RESEARCH = ROOT / "research"
PRIVATE = DATA / "private_features.csv"
HGB = OUT / "model_dani_tuned_hgb.csv"
LAG = OUT / "model_dani_tuned_lag.csv"
BASELINE = OUT / "model_dani_tuned_submission.csv"
PEER = RESEARCH / "paired_aoi_v2_private_hgb_lag20_peer10.csv"
DEFAULT_OUT = OUT / "model_dani_peer_joint_submission.csv"
DEFAULT_META = OUT / "model_dani_peer_joint_metadata.json"
DEFAULT_ROWS = RESEARCH / "ensemble_cv_v2_peer_apply_rows.csv"
ROWKEY = ["anon_polygon_id", "date"]
CANON_DOYS = frozenset((97, 113, 129, 145, 161, 177, 193, 209, 225,
                        241, 257, 273, 289))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read_pred(path: Path) -> pd.DataFrame:
    z = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    req = set(ROWKEY + ["primary_ndvi_pred"])
    if not req.issubset(z.columns):
        raise ValueError(f"{path.name}: missing {req - set(z.columns)}")
    z = z[ROWKEY + ["primary_ndvi_pred"]].copy()
    if z.duplicated(ROWKEY).any():
        raise ValueError(f"{path.name}: duplicate keys")
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-name", default=DEFAULT_OUT.name)
    ap.add_argument("--metadata-name", default=DEFAULT_META.name)
    ap.add_argument("--rows-name", default=DEFAULT_ROWS.name)
    ap.add_argument("--shock-coef", type=float, default=0.15)
    ap.add_argument("--state-coef", type=float, default=-0.05)
    ap.add_argument("--peer-file", default=PEER.name,
                    help="peer candidate CSV name under research/")
    args = ap.parse_args()
    out_path = OUT / str(args.output_name)
    meta_path = OUT / str(args.metadata_name)
    rows_path = RESEARCH / str(args.rows_name)
    peer_path = RESEARCH / str(args.peer_file)
    if out_path.resolve() == BASELINE.resolve():
        raise ValueError("refusing to overwrite production baseline")
    if out_path.parent.resolve() != OUT.resolve() or meta_path.parent.resolve() != OUT.resolve():
        raise ValueError("output and metadata must be direct children of outputs/")
    if rows_path.parent.resolve() != RESEARCH.resolve():
        raise ValueError("diagnostic rows must be a direct child of research/")
    if peer_path.parent.resolve() != RESEARCH.resolve():
        raise ValueError("peer file must be a direct child of research/")

    # Import the feature-only implementation.  It never executes main on
    # import and reads no evaluation truth/status/source fields.
    sys.path.insert(0, str(RESEARCH))
    from ensemble_cv_v2_apply import _seasonal_residuals, _shock, _state  # noqa: E402

    private = pd.read_csv(PRIVATE, parse_dates=["date"], low_memory=False)
    if private.duplicated(ROWKEY).any():
        raise ValueError("private_features has duplicate AOI/date keys")
    hidden = private["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
    if int(hidden.sum()) == 0:
        raise ValueError("private_features has no synthetic gaps")
    known = private["primary_ndvi"].notna().to_numpy(bool) & ~hidden
    qi = np.flatnonzero(hidden)
    keys = private.loc[hidden, ROWKEY].copy().reset_index(drop=True)
    keys["date"] = pd.to_datetime(keys["date"])

    # Features are rebuilt from the current private table, not copied from a
    # scored/evaluation table.  Hidden rows contribute only their date/key and
    # the is_synthetic_gap mask.
    residual = _seasonal_residuals(private, known)
    shock, shock_n = _shock(private, known, residual, qi)
    state, state_n = _state(private, known, residual, qi)
    canon = keys["date"].dt.dayofyear.isin(CANON_DOYS).to_numpy(bool)

    hgb = _read_pred(HGB)
    lag = _read_pred(LAG)
    baseline = _read_pred(BASELINE)
    peer = _read_pred(peer_path)
    hidden_set = set(map(tuple, keys[ROWKEY].to_numpy()))
    for name, z in [("hgb", hgb), ("lag", lag), ("baseline", baseline), ("peer", peer)]:
        if set(map(tuple, z[ROWKEY].to_numpy())) != hidden_set or len(z) != len(keys):
            raise ValueError(f"{name} keys do not exactly match hidden private keys")

    # All joins are key-based and preserve ``keys`` order.
    q = keys.copy()
    for name, z in [("hgb", hgb), ("lag", lag), ("baseline", baseline), ("peer", peer)]:
        q[name] = q.merge(z, on=ROWKEY, how="left", validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
    if not np.allclose(q["baseline"], 0.8 * q["hgb"] + 0.2 * q["lag"], atol=1e-6, rtol=0):
        raise ValueError("saved baseline is not 0.8*hgb + 0.2*lag")
    q["shock"] = shock
    q["state"] = state
    q["shock_n"] = shock_n
    q["state_n"] = state_n
    q["canon"] = canon
    delta = np.where(canon, 0.0,
                     float(args.shock_coef) * np.nan_to_num(shock, nan=0.0) +
                     float(args.state_coef) * np.nan_to_num(state, nan=0.0))
    q["correction"] = delta
    q["primary_ndvi_pred"] = np.clip(q["peer"].to_numpy(float) + delta, -0.5, 1.2)
    out = q[ROWKEY + ["primary_ndvi_pred"]].copy()
    out.to_csv(out_path, index=False, float_format="%.8f")
    q.to_csv(rows_path, index=False, float_format="%.8f")

    check = pd.read_csv(out_path, parse_dates=["date"])
    if list(check.columns) != ROWKEY + ["primary_ndvi_pred"] or len(check) != int(hidden.sum()):
        raise ValueError("candidate contract mismatch")
    if set(map(tuple, check[ROWKEY].to_numpy())) != hidden_set:
        raise ValueError("candidate key mismatch")
    metadata = {
        "candidate": out_path.name,
        "generated_by": Path(__file__).name,
        "formula": f"peer={peer_path.name}; if canon=False: peer+{float(args.shock_coef):g}*shock{float(args.state_coef):+g}*state; else peer",
        "shock_coef": float(args.shock_coef),
        "state_coef": float(args.state_coef),
        "rows": int(len(out)),
        "hidden_rows_in_private": int(hidden.sum()),
        "known_rows_used_for_features": int(known.sum()),
        "key_match_hidden": True,
        "canon_true": int(canon.sum()),
        "canon_false": int((~canon).sum()),
        "shock_finite": int(np.isfinite(shock).sum()),
        "shock_at_least_3_peers": int((shock_n >= 3).sum()),
        "state_finite": int(np.isfinite(state).sum()),
        "correction_nonzero": int(np.count_nonzero(np.abs(delta) > 1e-12)),
        "correction_min": float(np.min(delta)),
        "correction_max": float(np.max(delta)),
        "correction_mean": float(np.mean(delta)),
        "peer_min": float(q["peer"].min()),
        "peer_max": float(q["peer"].max()),
        "candidate_min": float(out["primary_ndvi_pred"].min()),
        "candidate_max": float(out["primary_ndvi_pred"].max()),
        "sha256": {
            "private_features.csv": _sha256(PRIVATE),
            "model_dani_tuned_hgb.csv": _sha256(HGB),
            "model_dani_tuned_lag.csv": _sha256(LAG),
            "model_dani_tuned_submission.csv": _sha256(BASELINE),
            peer_path.name: _sha256(peer_path),
            out_path.name: _sha256(out_path),
        },
        "hidden_label_columns_read": [],
        "production_baseline_overwritten": False,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

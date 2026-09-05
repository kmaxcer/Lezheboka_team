"""Build the observable date-shock candidate submission.

The script is intentionally independent of evaluation labels.  It reads the
private feature table (where synthetic-gap rows have ``primary_ndvi`` and all
dynamic fields masked), the already-fitted HGB/lag component submissions, and
uses only visible ``primary_ndvi`` values to derive two features:

``shock``
    leave-AOI-out median seasonal residual of visible peers on the same date;
``state``
    distance-weighted recent seasonal residuals of the same AOI/year.

The fixed rule selected by ``ensemble_cv_v2.py`` is applied only on non-canon
dates: ``baseline + .15*shock - .05*state``.  Canon dates retain the baseline.
This writes a separate candidate and never overwrites ``model_dani_tuned*``.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
OUT = ROOT / "outputs"
RESEARCH = ROOT / "research"
PRIVATE = DATA / "private_features.csv"
BASELINE = OUT / "model_dani_tuned_submission.csv"
HGB = OUT / "model_dani_tuned_hgb.csv"
LAG = OUT / "model_dani_tuned_lag.csv"
JOINT = OUT / "model_dani_joint_submission.csv"
META = OUT / "model_dani_joint_metadata.json"
ROWS = RESEARCH / "ensemble_cv_v2_apply_rows.csv"

CANON_DOYS = frozenset((97, 113, 129, 145, 161, 177, 193, 209, 225,
                        241, 257, 273, 289))
ROWKEY = ["anon_polygon_id", "date"]
CLIP_LO, CLIP_HI = -0.5, 1.2


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _seasonal_residuals(frame: pd.DataFrame, known: np.ndarray) -> np.ndarray:
    """Return robust visible residuals; no target-like hidden column is read."""
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_yr"] = d["date"].dt.year.astype(int)
    d["_doy"] = d["date"].dt.dayofyear.astype(int)
    d["_bin"] = ((d["_doy"] - 1) // 16).astype(int)
    d["_known"] = np.asarray(known, dtype=bool)
    d["_y"] = pd.to_numeric(d["primary_ndvi"], errors="coerce")
    d.loc[~d["_known"], "_y"] = np.nan
    obs = d.loc[d["_known"] & d["_y"].notna(),
               ["anon_polygon_id", "_yr", "_bin", "_y"]].copy()
    if obs.empty:
        return np.full(len(d), np.nan, dtype=float)
    obs = obs[obs["_y"].between(-0.5, 1.2)].copy()
    if obs.empty:
        return np.full(len(d), np.nan, dtype=float)
    p1 = obs.groupby(["anon_polygon_id", "_yr", "_bin"], observed=True)["_y"].median().rename("p1").reset_index()
    p2 = obs.groupby(["anon_polygon_id", "_bin"], observed=True)["_y"].median().rename("p2").reset_index()
    p3 = obs.groupby(["_yr", "_bin"], observed=True)["_y"].median().rename("p3").reset_index()
    p4 = obs.groupby(["_bin"], observed=True)["_y"].median().rename("p4").reset_index()
    gmed = float(obs["_y"].median())
    z = d[["anon_polygon_id", "_yr", "_bin"]].copy()
    z = z.merge(p1, on=["anon_polygon_id", "_yr", "_bin"], how="left", sort=False)
    z = z.merge(p2, on=["anon_polygon_id", "_bin"], how="left", sort=False)
    z = z.merge(p3, on=["_yr", "_bin"], how="left", sort=False)
    z = z.merge(p4, on=["_bin"], how="left", sort=False)
    prof = z["p1"].combine_first(z["p2"]).combine_first(z["p3"]).combine_first(z["p4"]).fillna(gmed).to_numpy(float)
    residual = d["_y"].to_numpy(float) - prof
    residual[~d["_known"].to_numpy(bool)] = np.nan
    return np.clip(residual, -0.5, 0.5)


def _shock(frame: pd.DataFrame, known: np.ndarray, residual: np.ndarray,
           query_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    dates = d["date"].to_numpy()
    by_date: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    for dt, g in d.loc[known & np.isfinite(residual)].groupby("date", sort=False):
        ii = g.index.to_numpy(dtype=int)
        good = np.isfinite(residual[ii])
        by_date[pd.Timestamp(dt)] = (ids[ii][good], residual[ii][good])
    out = np.full(len(query_idx), np.nan, dtype=float)
    counts = np.zeros(len(query_idx), dtype=int)
    for j, i in enumerate(query_idx):
        pair = by_date.get(pd.Timestamp(dates[i]))
        if pair is None:
            continue
        peer_ids, vals = pair
        vals = vals[peer_ids != ids[i]]
        vals = vals[np.isfinite(vals)]
        counts[j] = len(vals)
        if len(vals) >= 3:
            out[j] = float(np.median(np.clip(vals, -0.3, 0.3)))
    return out, counts


def _state(frame: pd.DataFrame, known: np.ndarray, residual: np.ndarray,
           query_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_yr"] = d["date"].dt.year.astype(int)
    ords = d["date"].map(pd.Timestamp.toordinal).to_numpy(float)
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    out = np.full(len(query_idx), np.nan, dtype=float)
    counts = np.zeros(len(query_idx), dtype=int)
    groups: dict[tuple[str, int], np.ndarray] = {}
    for k, ix in d.loc[known & np.isfinite(residual)].groupby(["anon_polygon_id", "_yr"], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        ii = ii[np.isfinite(residual[ii])]
        if len(ii):
            groups[(str(k[0]), int(k[1]))] = ii
    for j, i in enumerate(query_idx):
        ii = groups.get((ids[i], int(d["_yr"].iat[i])))
        if ii is None or len(ii) < 2:
            continue
        dist = np.abs(ords[ii] - ords[i])
        take = np.argsort(dist)[: min(8, len(ii))]
        take = take[dist[take] <= 120]
        if len(take) < 2:
            continue
        vals = np.clip(residual[ii[take]], -0.3, 0.3)
        w = np.exp(-dist[take] / 45.0)
        if not np.isfinite(w).all() or w.sum() <= 0:
            continue
        out[j] = float(np.average(vals, weights=w))
        counts[j] = len(vals)
    return out, counts


def _read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    z = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    required = set(ROWKEY + ["primary_ndvi_pred"])
    if not required.issubset(z.columns):
        raise ValueError(f"{path.name}: missing required columns {required - set(z.columns)}")
    z = z[ROWKEY + ["primary_ndvi_pred"]].copy()
    if z.duplicated(ROWKEY).any():
        raise ValueError(f"{path.name}: duplicate keys")
    return z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shock-coef", type=float, default=0.15,
                        help="fixed coefficient for the observable date shock")
    parser.add_argument("--state-coef", type=float, default=-0.05,
                        help="fixed coefficient for the observable local state")
    parser.add_argument("--output-name", default=JOINT.name,
                        help="candidate CSV name under outputs/ (must not be the production baseline)")
    parser.add_argument("--metadata-name", default=META.name,
                        help="metadata JSON name under outputs/")
    parser.add_argument("--rows-name", default=ROWS.name,
                        help="optional diagnostics CSV name under research/")
    args = parser.parse_args()
    shock_coef = float(args.shock_coef)
    state_coef = float(args.state_coef)
    out_path = OUT / str(args.output_name)
    meta_path = OUT / str(args.metadata_name)
    rows_path = RESEARCH / str(args.rows_name)
    if out_path.resolve() == BASELINE.resolve():
        raise ValueError("refusing to overwrite model_dani_tuned_submission.csv")
    if out_path.parent.resolve() != OUT.resolve() or meta_path.parent.resolve() != OUT.resolve():
        raise ValueError("output paths must stay directly under outputs/")
    if rows_path.parent.resolve() != RESEARCH.resolve():
        raise ValueError("diagnostic path must stay directly under research/")
    private = pd.read_csv(PRIVATE, parse_dates=["date"], low_memory=False)
    if private.duplicated(ROWKEY).any():
        raise ValueError("private_features has duplicate AOI/date keys")
    hidden = private["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
    if hidden.sum() == 0:
        raise ValueError("no synthetic-gap rows found")
    known = private["primary_ndvi"].notna().to_numpy(bool) & ~hidden
    qi = np.flatnonzero(hidden)

    baseline = _read_submission(BASELINE)
    hgb = _read_submission(HGB)
    lag = _read_submission(LAG)
    hidden_keys = private.loc[hidden, ROWKEY].copy()
    hidden_keys["date"] = pd.to_datetime(hidden_keys["date"])
    for name, z in [("baseline", baseline), ("hgb", hgb), ("lag", lag)]:
        k = z.copy(); k["date"] = pd.to_datetime(k["date"])
        if len(k) != len(hidden_keys) or set(map(tuple, k[ROWKEY].to_numpy())) != set(map(tuple, hidden_keys[ROWKEY].to_numpy())):
            raise ValueError(f"{name} keys do not exactly match private hidden keys")

    residual = _seasonal_residuals(private, known)
    shock, shock_n = _shock(private, known, residual, qi)
    state, state_n = _state(private, known, residual, qi)

    # Join components by key to avoid relying on CSV ordering.
    q = hidden_keys.copy()
    q["baseline"] = q.merge(baseline, on=ROWKEY, how="left", validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
    q["hgb"] = q.merge(hgb, on=ROWKEY, how="left", validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
    q["lag"] = q.merge(lag, on=ROWKEY, how="left", validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
    if not np.allclose(q["baseline"].to_numpy(float), 0.8*q["hgb"].to_numpy(float) + 0.2*q["lag"].to_numpy(float), atol=1e-6, rtol=0):
        raise ValueError("baseline is not 0.8*hgb + 0.2*lag")
    q["shock"] = shock
    q["state"] = state
    q["shock_n"] = shock_n
    q["state_n"] = state_n
    q["canon"] = q["date"].dt.dayofyear.isin(CANON_DOYS)
    # Fixed, predeclared rule.  Missing evidence contributes zero, matching
    # the research evaluator; the canon branch never receives a correction.
    delta = np.where(q["canon"].to_numpy(bool), 0.0,
                     shock_coef * np.nan_to_num(q["shock"].to_numpy(float), nan=0.0) +
                     state_coef * np.nan_to_num(q["state"].to_numpy(float), nan=0.0))
    q["correction"] = delta
    q["primary_ndvi_pred"] = np.clip(q["baseline"].to_numpy(float) + delta, CLIP_LO, CLIP_HI)
    out = q[ROWKEY + ["primary_ndvi_pred"]].copy()
    out.to_csv(out_path, index=False, float_format="%.8f")
    # Diagnostic rows contain no hidden labels and make feature coverage
    # independently inspectable.
    q.to_csv(rows_path, index=False, float_format="%.8f")

    # Verify exact output contract/order and compute immutable hashes only after
    # writing the candidate.
    check = pd.read_csv(out_path, parse_dates=["date"])
    if list(check.columns) != ROWKEY + ["primary_ndvi_pred"] or len(check) != int(hidden.sum()):
        raise ValueError("joint submission contract mismatch")
    if set(map(tuple, check[ROWKEY].to_numpy())) != set(map(tuple, hidden_keys[ROWKEY].to_numpy())):
        raise ValueError("joint submission key mismatch")
    canon = q["canon"].to_numpy(bool)
    shock_ok = np.isfinite(q["shock"].to_numpy(float))
    state_ok = np.isfinite(q["state"].to_numpy(float))
    metadata = {
        "candidate": out_path.name,
        "generated_by": Path(__file__).name,
        "formula": f"baseline=0.8*hgb+0.2*lag; if canon=False: baseline+{shock_coef:g}*shock{state_coef:+g}*state; else baseline",
        "shock_coef": shock_coef,
        "state_coef": state_coef,
        "clip": [CLIP_LO, CLIP_HI],
        "rows": int(len(out)),
        "hidden_rows_in_private": int(hidden.sum()),
        "known_rows_used_for_features": int(known.sum()),
        "key_match_hidden": True,
        "canon_true": int(canon.sum()),
        "canon_false": int((~canon).sum()),
        "shock_finite": int(shock_ok.sum()),
        "shock_at_least_3_peers": int((q["shock_n"].to_numpy(int) >= 3).sum()),
        "state_finite": int(state_ok.sum()),
        "correction_nonzero": int(np.count_nonzero(np.abs(delta) > 1e-12)),
        "correction_min": float(np.min(delta)),
        "correction_max": float(np.max(delta)),
        "correction_mean": float(np.mean(delta)),
        "baseline_min": float(q["baseline"].min()),
        "baseline_max": float(q["baseline"].max()),
        "candidate_min": float(out["primary_ndvi_pred"].min()),
        "candidate_max": float(out["primary_ndvi_pred"].max()),
        "input_sha256": {
            "private_features.csv": _sha256(PRIVATE),
            "model_dani_tuned_submission.csv": _sha256(BASELINE),
            "model_dani_tuned_hgb.csv": _sha256(HGB),
            "model_dani_tuned_lag.csv": _sha256(LAG),
            out_path.name: _sha256(out_path),
        },
        "label_columns_read": ["primary_ndvi"],
        "hidden_label_columns_read": [],
        "production_baseline_overwritten": False,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Small read-only CV probe for exact-weather peer filling.

This research utility uses the same 2025 synthetic mask as
``teammate_sweep_ensemble.py`` (seed 0), discovers AOI peers whose observed
ERA5 temperature/precipitation trajectories are exactly equal, fills only
masked weather cells from those peers, and scores one HGB model.  It writes
only ``reverse_aoi_weather_fill_cv.csv`` under research/.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(ROOT / "research"))
from teammate_sweep_ensemble import (  # noqa: E402
    make_masked, make_reference, fit_oof_models, hgb_predictions, keyed_pred,
)

ID = "anon_polygon_id"
DATE = "date"
TEMP = "era5_temp_c"
PREC = "era5_precip_mm"
TARGET = "primary_ndvi"


def weather_peer_map(frame: pd.DataFrame, private_ids: list[str]) -> dict[str, str]:
    """Find exact weather-trajectory peers using only observed frame cells."""
    out: dict[str, str] = {}
    # Keep one row per AOI/date and compare pairs on common non-null dates.
    piv_t = frame.pivot_table(index=DATE, columns=ID, values=TEMP, aggfunc="first")
    piv_p = frame.pivot_table(index=DATE, columns=ID, values=PREC, aggfunc="first")
    all_ids = [str(x) for x in frame[ID].drop_duplicates()]
    for a in private_ids:
        if a not in piv_t.columns:
            continue
        best = None
        for b in all_ids:
            if b == a or b not in piv_t.columns:
                continue
            ok = piv_t[a].notna() & piv_p[a].notna() & piv_t[b].notna() & piv_p[b].notna()
            n = int(ok.sum())
            if n < 30:
                continue
            et = np.abs(piv_t.loc[ok, a].to_numpy(float) - piv_t.loc[ok, b].to_numpy(float))
            ep = np.abs(piv_p.loc[ok, a].to_numpy(float) - piv_p.loc[ok, b].to_numpy(float))
            exact = bool(np.all(et == 0.0) and np.all(ep == 0.0))
            if not exact:
                continue
            cand = (n, b)
            if best is None or cand > best:
                best = cand
        if best is not None:
            out[a] = best[1]
    return out


def fill_private_weather(train: pd.DataFrame, masked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], int]:
    """Fill masked private weather from exact peers; preserve sidecars."""
    # Mapping is learned from observed (non-null) weather in the combined frame.
    combined = pd.concat([train[[ID, DATE, TEMP, PREC]], masked[[ID, DATE, TEMP, PREC]]], ignore_index=True)
    combined[DATE] = pd.to_datetime(combined[DATE])
    pids = [str(x) for x in masked[ID].drop_duplicates()]
    peers = weather_peer_map(combined, pids)
    out = masked.copy()
    out[DATE] = pd.to_datetime(out[DATE])
    # Use a keyed lookup from each peer's observed weather; no target/sensor fields.
    lookup = combined.dropna(subset=[TEMP, PREC]).drop_duplicates([ID, DATE]).set_index([ID, DATE])[[TEMP, PREC]]
    filled = 0
    gap = out["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
    for i in np.flatnonzero(gap):
        a = str(out.at[i, ID])
        b = peers.get(a)
        if b is None:
            continue
        key = (b, out.at[i, DATE])
        if key not in lookup.index:
            continue
        vals = lookup.loc[key]
        if pd.isna(vals[TEMP]) or pd.isna(vals[PREC]):
            continue
        out.at[i, TEMP] = float(vals[TEMP])
        out.at[i, PREC] = float(vals[PREC])
        filled += 1
    return out, peers, filled


def rmse_mae(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(pred) & np.isfinite(truth)
    e = pred[ok] - truth[ok]
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    masked, mask = make_masked(pr, 0, mode="2025")
    filled, peers, nfilled = fill_private_weather(tr, masked)
    ref = make_reference(tr, filled)
    models = fit_oof_models(ref, fold_seed=42, model_seeds=[42])
    pred = hgb_predictions(ref, models)[42]
    q = masked.loc[mask, [ID, DATE, TARGET]].copy()
    q[DATE] = pd.to_datetime(q[DATE])
    y = q[TARGET].to_numpy(float)
    p_fill = keyed_pred(pred, q.rename(columns={TARGET: "_truth"}))

    # Existing sweep prediction is the exact unfilled HGB baseline for this mask.
    sweep = pd.read_csv(ROOT / "research" / "teammate_sweep_ensemble_predictions.csv", parse_dates=[DATE], low_memory=False)
    base = sweep[(sweep["mode"] == "2025") & (sweep["mask_seed"] == 0)][[ID, DATE, "_truth", "hgb_seed42"]].copy()
    p_base = base["hgb_seed42"].to_numpy(float)
    y_base = base["_truth"].to_numpy(float)
    r_fill, a_fill = rmse_mae(p_fill, y)
    r_base, a_base = rmse_mae(p_base, y_base)
    rows = [
        {"mode": "2025", "seed": 0, "method": "hgb_weather_fill", "n": len(y), "weather_peer_map": len(peers), "weather_cells_filled": nfilled, "coverage": int(np.isfinite(p_fill).sum()), "rmse": r_fill, "mae": a_fill},
        {"mode": "2025", "seed": 0, "method": "hgb_unfilled_baseline", "n": len(y_base), "weather_peer_map": np.nan, "weather_cells_filled": 0, "coverage": int(np.isfinite(p_base).sum()), "rmse": r_base, "mae": a_base},
    ]
    pd.DataFrame(rows).to_csv(ROOT / "research" / "reverse_aoi_weather_fill_cv.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("peer_map", peers)


if __name__ == "__main__":
    main()

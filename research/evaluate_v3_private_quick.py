"""Quick leakage-safe audit of feature_hgb_v3 on a private-like holdout.

The organiser's synthetic gaps are kept hidden while an additional fixed 15%
of visible private rows (per AOI/year) is hidden.  The model is fit on the
combined train + private reference using disjoint pseudo masks, then scored
only on the extra private holdout.  This is intentionally a small audit (two
pseudo masks) so it can be rerun while iterating.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
ARCH = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
R = ROOT / "research"
sys.path.insert(0, str(ARCH))
sys.path.insert(0, str(R))
from agropulse.pipeline import build_features  # noqa: E402
from feature_hgb_v2 import _clear  # noqa: E402
from feature_hgb_v3 import extra_features_v3  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"


def make_holdout(pr: pd.DataFrame, seed: int = 70404) -> np.ndarray:
    known = pr[TARGET].notna().to_numpy(bool) & ~pr[GAP].fillna(False).to_numpy(bool)
    out = np.zeros(len(pr), dtype=bool)
    rng = np.random.default_rng(seed)
    yy = pd.to_datetime(pr[DATE]).dt.year
    for _, ix0 in pr.loc[known].groupby([ID, yy], sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int)
        n = max(1, int(round(.15 * len(ix))))
        out[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
    return out


def matrix(d: pd.DataFrame, observed: pd.Series, mask: np.ndarray) -> pd.DataFrame:
    mask = np.asarray(mask, bool)
    fr = _clear(d, mask)
    bx = build_features(fr, observed, pd.Series(mask, index=fr.index))
    ex = extra_features_v3(fr, observed, mask)
    return pd.concat([bx.reset_index(drop=True), ex.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)


def fit_v3(ref: pd.DataFrame, gaps: np.ndarray, n_masks: int = 2) -> tuple[pd.DataFrame, dict]:
    """Fit v3 on pseudo-masked visible rows and return predictions for gaps."""
    d = ref.copy().reset_index(drop=True)
    d[DATE] = pd.to_datetime(d[DATE])
    d["year"] = d["year"].fillna(d[DATE].dt.year).astype(int)
    d["doy"] = d["doy"].fillna(d[DATE].dt.dayofyear).astype(int)
    d["_truth"] = pd.to_numeric(d[TARGET], errors="coerce")
    gaps = np.asarray(gaps, bool)
    known = d[TARGET].notna().to_numpy(bool) & ~gaps
    years = d[DATE].dt.year.to_numpy(int)
    tab = pd.DataFrame({"id": d[ID].astype(str), "year": years})
    blocks, ys = [], []
    t0 = time.time()
    for rep in range(n_masks):
        rng = np.random.default_rng(20260905 + rep)
        pm = np.zeros(len(d), bool)
        for _, ix0 in tab.loc[known].groupby(["id", "year"], sort=False).groups.items():
            ix = np.asarray(ix0, dtype=int)
            n = max(1, int(round(.18 * len(ix))))
            pm[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
        comb = gaps | pm
        obs = d[TARGET].where(~comb)
        print(f"v3 features train block {rep + 1}/{n_masks}: {int(pm.sum())} rows", flush=True)
        x = matrix(d, obs, comb)
        blocks.append(x.loc[pm].reset_index(drop=True))
        ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    obs = d[TARGET].where(~gaps)
    print(f"v3 features query: {int(gaps.sum())} rows", flush=True)
    qx = matrix(d, obs, gaps).loc[gaps].reset_index(drop=True)
    X = pd.concat(blocks, ignore_index=True)
    y = pd.concat(ys, ignore_index=True).astype(float)
    m = HistGradientBoostingRegressor(
        loss="squared_error", random_state=42, learning_rate=.03,
        max_iter=350, max_leaf_nodes=63, min_samples_leaf=30,
        l2_regularization=8.0,
    )
    print("v3 fit", X.shape, flush=True)
    m.fit(X, y)
    pred = np.clip(m.predict(qx), -.2, 1.1)
    keys = d.loc[gaps, [ID, DATE]].copy().reset_index(drop=True)
    out = keys.copy(); out["v3"] = pred
    return out, {"features": int(X.shape[1]), "train_rows": int(len(X)), "seconds": round(time.time() - t0, 1)}


def metric(x: pd.DataFrame, col: str) -> tuple[int, float, float]:
    ok = np.isfinite(x[col].to_numpy(float)) & np.isfinite(x.truth.to_numpy(float))
    if not ok.any(): return 0, float("nan"), float("nan")
    e = x.loc[ok, col].to_numpy(float) - x.loc[ok, "truth"].to_numpy(float)
    return int(ok.sum()), float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def main() -> None:
    t0 = time.time()
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    tr[GAP] = False; pr[GAP] = pr[GAP].fillna(False).astype(bool)
    hold = make_holdout(pr)
    hidden = pr[GAP].to_numpy(bool)
    gaps_pr = hidden | hold
    # Keep labels in _truth only; dynamic fields on all gaps are blanked by
    # _clear when each feature matrix is built.
    tr2 = tr.copy(); p2 = pr.copy()
    tr2["_origin"] = "train"; p2["_origin"] = "private"
    ref = pd.concat([tr2, p2], ignore_index=True, sort=False)
    ref[DATE] = pd.to_datetime(ref[DATE]); ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int); ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    ref["_truth"] = pd.to_numeric(ref[TARGET], errors="coerce")
    # Mask only private holdout/organiser gaps.  Preserve train labels and all
    # visible private labels for cross-year and shared-AOI features.
    gap_keys = set(map(tuple, pr.loc[gaps_pr, [ID, DATE]].to_numpy()))
    gaps_ref = np.array([tuple(x) in gap_keys for x in ref[[ID, DATE]].to_numpy()], dtype=bool)
    ref.loc[gaps_ref, TARGET] = np.nan
    print("reference", len(ref), "gaps", int(gaps_ref.sum()), "holdout", int(hold.sum()), flush=True)
    vp, info = fit_v3(ref, gaps_ref, n_masks=2)
    # Restrict scored rows to additional private holdout, in original order.
    hk = pr.loc[hold, [ID, DATE]].copy(); hk["truth"] = pr.loc[hold, TARGET].to_numpy(float)
    q = hk.merge(vp, on=[ID, DATE], how="left", validate="one_to_one")
    train_ids = set(tr[ID].astype(str)); q["cohort"] = np.where(q[ID].astype(str).isin(train_ids), "shared", "new")
    q["year"] = pd.to_datetime(q[DATE]).dt.year.astype(int)
    # Join previously computed leakage-safe components for the exact same
    # holdout.  These values were generated by evaluate_private_cohort_blend.
    old = pd.read_csv(R / "private_cohort_blend_holdout_predictions.csv", parse_dates=[DATE], low_memory=False)
    keep = [ID, DATE, "hgb", "lag", "base40", "peer40", "joint40", "extended", "ext20", "ext40"]
    q = q.merge(old[keep], on=[ID, DATE], how="left", validate="one_to_one")
    # Small grid over v3 weight on top of the strongest current components.
    q["joint_v3_20"] = .8*q.joint40 + .2*q.v3
    q["joint_v3_30"] = .7*q.joint40 + .3*q.v3
    q["joint_v3_40"] = .6*q.joint40 + .4*q.v3
    q["ext40_v3_20"] = .8*q.ext40 + .2*q.v3
    q["ext40_v3_30"] = .7*q.ext40 + .3*q.v3
    q["ext40_v3_40"] = .6*q.ext40 + .4*q.v3
    cols = ["v3", "joint40", "ext40", "joint_v3_20", "joint_v3_30", "joint_v3_40", "ext40_v3_20", "ext40_v3_30", "ext40_v3_40"]
    rows = []
    groups = {
        "all": q,
        "new": q[q.cohort == "new"],
        "shared": q[q.cohort == "shared"],
        "history": q[q.year < 2025],
        "2025": q[q.year == 2025],
        "new_history": q[(q.cohort == "new") & (q.year < 2025)],
        "new_2025": q[(q.cohort == "new") & (q.year == 2025)],
        "shared_2025": q[(q.cohort == "shared") & (q.year == 2025)],
    }
    for gname, g in groups.items():
        for c in cols:
            n, rm, ma = metric(g, c); rows.append({"cohort": gname, "method": c, "n": n, "rmse": rm, "mae": ma})
    res = pd.DataFrame(rows)
    q.to_csv(R / "v3_private_holdout_predictions.csv", index=False)
    res.to_csv(R / "v3_private_holdout_results.csv", index=False)
    meta = {"holdout_seed": 70404, "holdout_rows": int(hold.sum()), "actual_hidden_rows": int(hidden.sum()), "n_masks": 2, **info, "seconds_total": round(time.time() - t0, 1)}
    (R / "v3_private_holdout_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(res[res.cohort.isin(["all", "history", "2025", "new_history", "new_2025", "shared_2025"])].to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__": main()

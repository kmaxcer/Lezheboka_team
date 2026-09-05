"""Evaluate baseline/peer/extended HGB on a private-like all-year holdout.

This is a research-only diagnostic.  A fixed random 15% of currently visible
private rows is hidden within each AOI/year, together with the organiser's
actual gaps.  Every component is refit from the remaining observations before
the holdout is scored.  The resulting row table is useful for cohort-aware
routing (new AOI vs shared AOI and year 2025 vs history).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
ARCH = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ARCH))
from agropulse.pipeline import build_features, create_model, load_competition_data  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from infer_lag import predict_private_lag  # noqa: E402
sys.path.insert(0, str(RESEARCH))
from paired_aoi_v2 import peer_predictions, _config_name, CONFIGS  # noqa: E402
from ensemble_cv_v2_apply import _seasonal_residuals, _shock, _state  # noqa: E402
from build_extended_hgb_private import _clear as ext_clear, _fit as ext_fit, _matrix as ext_matrix  # noqa: E402

ID = "anon_polygon_id"
DATE = "date"
TARGET = "primary_ndvi"
GAP = "is_synthetic_gap"
CANON = {97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289}


def make_holdout(pr: pd.DataFrame, seed: int = 70404) -> np.ndarray:
    """Select visible private rows exactly once per AOI/year."""
    known = pr[TARGET].notna().to_numpy(bool) & ~pr[GAP].fillna(False).to_numpy(bool)
    out = np.zeros(len(pr), dtype=bool)
    rng = np.random.default_rng(seed)
    years = pd.to_datetime(pr[DATE]).dt.year
    for _, ix0 in pr.loc[known].groupby([ID, years], sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int)
        n = max(1, int(round(.15 * len(ix))))
        out[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
    return out


def fit_hgb(reference: pd.DataFrame, gaps: np.ndarray) -> np.ndarray:
    """Fit the archive HGB from five leakage-safe pseudo folds."""
    known = reference[TARGET].notna().to_numpy(bool) & ~gaps
    rng = np.random.default_rng(42)
    folds = pd.Series(-1, index=reference.index)
    for _, ix0 in reference.loc[known].groupby(ID, sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int); rng.shuffle(ix); folds.loc[ix] = np.arange(len(ix)) % 5
    xs = []; ys = []
    for f in range(5):
        pseudo = folds.eq(f).to_numpy(bool)
        hidden = gaps | pseudo
        obs = reference[TARGET].mask(hidden)
        x = build_features(reference, obs, pd.Series(hidden, index=reference.index))
        xs.append(x.loc[pseudo]); ys.append(reference.loc[pseudo, "_truth"])
    X = pd.concat(xs, ignore_index=True); y = pd.concat(ys, ignore_index=True)
    m = create_model(42); m.fit(X, y)
    obs = reference[TARGET].mask(gaps)
    xq = build_features(reference, obs, pd.Series(gaps, index=reference.index)).loc[gaps]
    return np.clip(m.predict(xq), -.5, 1.2)


def fit_extended(reference: pd.DataFrame, gaps: np.ndarray, seed: int = 4) -> np.ndarray:
    """Fit regular/wide extended HGB from disjoint pseudo masks."""
    # Three deterministic pseudo blocks; mirrors build_extended_hgb_private.
    d = reference.copy().reset_index(drop=True)
    d[DATE] = pd.to_datetime(d[DATE]); d["year"] = d["year"].fillna(d[DATE].dt.year).astype(int); d["doy"] = d["doy"].fillna(d[DATE].dt.dayofyear).astype(int)
    d["_truth"] = pd.to_numeric(d[TARGET], errors="coerce")
    known = d[TARGET].notna().to_numpy(bool) & ~gaps
    years = d[DATE].dt.year.to_numpy(int)
    blocks = []; ys = []
    for rep in range(3):
        rng = np.random.default_rng(20260905 + rep + seed)
        pm = np.zeros(len(d), bool)
        tab = pd.DataFrame({"id": d[ID].astype(str), "year": years})
        for _, ix0 in tab.loc[known].groupby(["id", "year"], sort=False).groups.items():
            ix = np.asarray(ix0, dtype=int); n = max(1, int(round(.18 * len(ix))))
            pm[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
        comb = gaps | pm; fr = ext_clear(d, comb); obs = fr[TARGET].where(~comb)
        xx = ext_matrix(d, obs, comb); blocks.append(xx.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    fr = ext_clear(d, gaps); obs = fr[TARGET].where(~gaps); qx = ext_matrix(d, obs, gaps).loc[gaps].reset_index(drop=True)
    X = pd.concat(blocks, ignore_index=True); y = pd.concat(ys, ignore_index=True).astype(float)
    m = ext_fit("wide", X, y, 42)
    return np.clip(m.predict(qx), -.5, 1.2)


def score(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for cohort, h in [("all", g), ("new", g[g.cohort == "new"]), ("shared", g[g.cohort == "shared"]), ("history", g[g.year < 2025]), ("2025", g[g.year == 2025]), ("new_history", g[(g.cohort == "new") & (g.year < 2025)]), ("new_2025", g[(g.cohort == "new") & (g.year == 2025)]), ("shared_2025", g[(g.cohort == "shared") & (g.year == 2025)])]:
        if len(h) == 0: continue
        y = h.truth.to_numpy(float)
        for c in cols:
            p = h[c].to_numpy(float); ok = np.isfinite(y) & np.isfinite(p)
            rows.append({"cohort": cohort, "method": c, "n": int(ok.sum()), "rmse": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), "mae": float(np.mean(np.abs(p[ok] - y[ok])))})
    return pd.DataFrame(rows)


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    tr[GAP] = False; pr[GAP] = pr[GAP].fillna(False).astype(bool)
    hold_pr = make_holdout(pr)
    hidden_pr = pr[GAP].to_numpy(bool)
    gaps_pr = hidden_pr | hold_pr
    # Preserve labels only in sidecar; models see NaN on every gap.
    truth = pr[TARGET].to_numpy(float)
    pr_mask = pr.copy(); pr_mask.loc[gaps_pr, TARGET] = np.nan; pr_mask.loc[gaps_pr, GAP] = True
    pr_mask.loc[gaps_pr, [c for c in pr_mask.columns if c in ["s2_ndvi","s2_evi","s2_ndwi","landsat_ndvi","landsat_evi","landsat_ndwi","modis_ndvi","modis_evi","modis_ndwi","era5_temp_c","era5_precip_mm","year","doy","ndvi_climatology_mean","ndvi_climatology_std","n_reference_years"]]] = np.nan
    # Hidden flag is retained so every helper excludes it.
    # Construct reference directly to retain masked private values and labels.
    tr2 = tr.copy(); p2 = pr_mask.copy(); tr2["_origin"] = "train"; p2["_origin"] = "test"; p2["_test_order"] = np.arange(len(p2)); tr2["_test_order"] = np.nan
    ref = pd.concat([tr2, p2], ignore_index=True, sort=False).sort_values([ID, DATE, "_origin"]).reset_index(drop=True)
    ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int); ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    # Labels in sorted reference are pulled from original concatenation by key.
    labels = pd.concat([tr[[ID, DATE, TARGET]], pr[[ID, DATE, TARGET]]], ignore_index=True)
    ref = ref.merge(labels.rename(columns={TARGET:"_truth"}), on=[ID, DATE], how="left", validate="one_to_one")
    gaps_ref = ref[GAP].fillna(False).to_numpy(bool)
    # Add holdout flag based on key, since private rows were sorted with train.
    hk = set(map(tuple, pr.loc[hold_pr, [ID, DATE]].to_numpy()))
    gaps_ref = gaps_ref | np.array([tuple(x) in hk for x in ref[[ID, DATE]].to_numpy()])
    ref.loc[gaps_ref, TARGET] = np.nan
    # HGB and extended models.
    print("fit hgb", flush=True); hp = fit_hgb(ref, gaps_ref)
    # Map predictions to private holdout keys in sorted reference order.
    qref = ref.loc[gaps_ref, [ID, DATE]].copy(); qref["hgb"] = hp
    print("fit lag", flush=True); lp = predict_private_lag(pr_mask, train=tr, k=16, degree=3, bin_days=30, use_date_prior=True, date_weight=1.0); lp[DATE] = pd.to_datetime(lp[DATE])
    lmap = lp.set_index([ID, DATE])["primary_ndvi_pred"]
    qkeys = pr.loc[hold_pr, [ID, DATE]].copy(); qkeys["lag"] = [lmap.get((i, d), np.nan) for i,d in qkeys[[ID,DATE]].itertuples(index=False, name=None)]
    q = qkeys.merge(qref, on=[ID, DATE], how="left", validate="one_to_one")
    q["truth"] = [truth[i] for i in np.flatnonzero(hold_pr)]
    # Peer maps use a private-like frame with holdout/actual gaps NaN.
    ppeer = pr_mask.copy(); ppeer.loc[gaps_pr, TARGET] = np.nan; ppeer.loc[gaps_pr, GAP] = True
    print("fit peer", flush=True); pp, _ = peer_predictions(ppeer, hold_pr, partition="private_holdout")
    pp = pp.drop(columns=["_row"], errors="ignore"); q = q.merge(pp, on=[ID, DATE], how="left", validate="one_to_one")
    # Use the leading peer config.
    peer_col = _config_name(16, .60, .125, 2)
    q["base40"] = .6*q.hgb + .4*q.lag
    q["peer40"] = q["base40"]
    ok = q[peer_col].notna(); q.loc[ok, "peer40"] = .9*q.loc[ok, "base40"] + .1*q.loc[ok, peer_col]
    # Observable shock/state correction.
    known = pr_mask[TARGET].notna().to_numpy(bool) & ~gaps_pr; qi = np.flatnonzero(hold_pr)
    resid = _seasonal_residuals(pr_mask, known); shock, _ = _shock(pr_mask, known, resid, qi); state, _ = _state(pr_mask, known, resid, qi)
    q["shock"] = shock; q["state"] = state; canon = q[DATE].dt.dayofyear.isin(CANON).to_numpy(bool)
    q["joint40"] = q.peer40.to_numpy(float) + np.where(canon, 0, .35*np.nan_to_num(shock) - .20*np.nan_to_num(state))
    # Extended is fit on reference and queried at holdout rows.
    print("fit extended", flush=True); ep = fit_extended(ref, gaps_ref, 4); qref2 = ref.loc[gaps_ref, [ID, DATE]].copy(); qref2["extended"] = ep; q = q.merge(qref2, on=[ID,DATE], how="left", validate="one_to_one")
    q["ext20"] = .8*q.joint40 + .2*q.extended; q["ext40"] = .6*q.joint40 + .4*q.extended
    q["cohort"] = np.where(q[ID].astype(str).isin(set(tr[ID].astype(str))), "shared", "new"); q["year"] = pd.to_datetime(q[DATE]).dt.year.astype(int)
    q.to_csv(RESEARCH / "private_cohort_blend_holdout_predictions.csv", index=False)
    methods = ["hgb","lag","base40","peer40","joint40","extended","ext20","ext40"] + [peer_col]
    score(q, methods).to_csv(RESEARCH / "private_cohort_blend_holdout_results.csv", index=False)
    print(score(q, methods).to_string(index=False), flush=True)


if __name__ == "__main__": main()

"""Research-only CatBoost screen for the CosmoHack NDVI gap task.

The script deliberately mirrors the competition masking protocol: every
feature matrix is built after the query rows (and the pseudo-training gaps)
have had all dynamic fields blanked.  It evaluates a small number of CatBoost
configurations on the private hidden day-of-year mask replayed onto train
years, then on a random train mask and a 2025 private-known proxy.  No
production files are touched.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
from agropulse.pipeline import build_features, FULL_FEATURES  # noqa: E402

TARGET = "primary_ndvi"
DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi",
    "era5_temp_c", "era5_precip_mm", "year", TARGET, "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def clear(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True)
    mask = np.asarray(mask, dtype=bool)
    for c in DYNAMIC:
        if c in d.columns:
            d.loc[mask, c] = np.nan
    d["is_synthetic_gap"] = mask
    d["year"] = d["year"].fillna(d.date.dt.year).astype(int)
    d["doy"] = d["doy"].fillna(d.date.dt.dayofyear).astype(int)
    return d


def _frame_x(frame: pd.DataFrame, observed: pd.Series, mask: np.ndarray,
             categorical: bool) -> tuple[pd.DataFrame, list[str]]:
    """Build numeric archive features and optional stable categorical IDs."""
    fr = clear(frame, mask)
    obs = observed.reset_index(drop=True)
    # build_features only uses observed_target for reconstruction.  Keep the
    # original index alignment explicit because all frames were reset above.
    x = build_features(fr, obs, pd.Series(np.asarray(mask, bool)))
    x = x.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    cats: list[str] = []
    if categorical:
        # CatBoost handles unseen categories in a query; use strings with an
        # explicit sentinel rather than pandas categorical codes (which can
        # accidentally assign different codes to train/query blocks).
        x["aoi_cat"] = fr["anon_polygon_id"].astype(str).fillna("<NA>").to_numpy()
        x["crop_cat"] = fr["crop_type"].astype(str).fillna("<NA>").to_numpy()
        cats = ["aoi_cat", "crop_cat"]
    return x, cats


def _with_cats(x: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach categorical columns to an already-built matrix (no rebuild)."""
    z = x.copy()
    z["aoi_cat"] = frame["anon_polygon_id"].astype(str).fillna("<NA>").to_numpy()
    z["crop_cat"] = frame["crop_type"].astype(str).fillna("<NA>").to_numpy()
    return z, ["aoi_cat", "crop_cat"]


def _pseudo_masks(frame: pd.DataFrame, excluded: np.ndarray, seed: int,
                  fraction: float = .18, n_masks: int = 2) -> list[np.ndarray]:
    """Stratified pseudo-gaps, excluding the outer query rows."""
    rng = np.random.default_rng(seed)
    known = frame[TARGET].notna().to_numpy(bool) & ~np.asarray(excluded, bool)
    years = pd.to_datetime(frame.date).dt.year.to_numpy(int)
    out: list[np.ndarray] = []
    for rep in range(n_masks):
        m = np.zeros(len(frame), dtype=bool)
        # Keep a little diversity while preserving AOI/year coverage.
        pool = known.copy()
        for (aoi, yr), ix0 in pd.DataFrame({"aoi": frame.anon_polygon_id.astype(str),
                                             "yr": years}).loc[pool].groupby(["aoi", "yr"], sort=False).groups.items():
            ix = np.asarray(ix0, dtype=int)
            if len(ix) == 0:
                continue
            nn = max(1, int(round(fraction * len(ix))))
            # Use a deterministic stream per replicate/group.
            pick = rng.choice(ix, size=min(nn, len(ix)), replace=False)
            m[pick] = True
        out.append(m)
    return out


def _fit_predict(blocks: list[pd.DataFrame], ys: list[pd.Series], qx: pd.DataFrame,
                 cats: list[str], kind: str) -> np.ndarray:
    specs = {
        # Short, deliberately conservative settings: this is a model-class
        # screen, not the final overnight fit.  Longer CatBoost runs were
        # disproportionately slow on the wide pseudo-gap matrix.
        "base": dict(iterations=220, depth=6, learning_rate=.05,
                      l2_leaf_reg=12., random_strength=.5),
        "deep": dict(iterations=320, depth=7, learning_rate=.04,
                     l2_leaf_reg=18., random_strength=1.0),
    }
    x = pd.concat(blocks, ignore_index=True)
    y = pd.concat(ys, ignore_index=True).astype(float)
    # CatBoost is sensitive to object columns outside cat_features.
    cat_idx = [x.columns.get_loc(c) for c in cats]
    m = CatBoostRegressor(
        loss_function="RMSE", eval_metric="RMSE", random_seed=42,
        verbose=False, allow_writing_files=False, thread_count=4,
        **specs[kind],
    )
    m.fit(x, y, cat_features=cat_idx)
    return np.clip(m.predict(qx), -.2, 1.1)


def _exact_screen(train: pd.DataFrame, private: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for yr in (2019, 2020, 2021, 2022, 2023, 2024):
        fold, truth = make_fold(train.copy(), private.copy(), yr)
        qm = fold.is_synthetic_gap.fillna(False).to_numpy(bool)
        if not qm.any():
            continue
        blocks_num: list[pd.DataFrame] = []
        blocks_cat: list[pd.DataFrame] = []
        targets: list[pd.Series] = []
        # Two pseudo masks are enough for a fast model-class screen.
        for pm in _pseudo_masks(fold, qm, 9100 + yr, .18, 2):
            comb = qm | pm
            fr = clear(fold, comb)
            obs = fr[TARGET].where(~comb)
            xn, _ = _frame_x(fold, obs, comb, False)
            xc, _ = _with_cats(xn, fr)
            blocks_num.append(xn.loc[pm].reset_index(drop=True))
            blocks_cat.append(xc.loc[pm].reset_index(drop=True))
            targets.append(fold.loc[pm, "_truth"].reset_index(drop=True))
        vf = clear(fold, qm)
        obs = vf[TARGET].where(~qm)
        qn, _ = _frame_x(fold, obs, qm, False)
        qc, cats = _with_cats(qn, vf)
        y = truth.to_numpy(float)
        for kind in ("base", "deep"):
            t0 = time.time()
            if kind == "base":
                p = _fit_predict(blocks_num, targets, qn.loc[qm].reset_index(drop=True), [], kind)
            else:
                p = _fit_predict(blocks_cat, targets, qc.loc[qm].reset_index(drop=True), cats, kind)
            e = p - y
            rows.append({"protocol": "exact", "year": yr, "kind": kind,
                         "n": len(y), "rmse": float(np.sqrt(np.mean(e * e))),
                         "mae": float(np.mean(np.abs(e))), "seconds": round(time.time() - t0, 1)})
            pred_rows.append(pd.DataFrame({"protocol": "exact", "year": yr,
                                           "anon_polygon_id": fold.loc[qm, "anon_polygon_id"].to_numpy(),
                                           "date": fold.loc[qm, "date"].to_numpy(),
                                           "truth": y, "kind": kind, "pred": p}))
        print("exact", yr, "n", len(y), flush=True)
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def _random_screen(train: pd.DataFrame, private: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One random private-like mask over train years (fast robustness check)."""
    d = train.copy().reset_index(drop=True)
    d["_truth"] = d[TARGET].astype(float)
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(77123)
    pool = d[TARGET].notna().to_numpy(bool)
    hold = np.zeros(len(d), bool)
    key = pd.DataFrame({"aoi": d.anon_polygon_id.astype(str), "yr": d.date.dt.year})
    for _, ix0 in key.loc[pool].groupby(["aoi", "yr"], sort=False).groups.items():
        ix = np.asarray(ix0, int); nn = max(1, int(round(.15 * len(ix))))
        hold[rng.choice(ix, size=min(nn, len(ix)), replace=False)] = True
    blocks_num: list[pd.DataFrame] = []; blocks_cat: list[pd.DataFrame] = []; ys: list[pd.Series] = []
    for pm in _pseudo_masks(d, hold, 88001, .16, 2):
        comb = hold | pm; fr = clear(d, comb); obs = fr[TARGET].where(~comb)
        xn, _ = _frame_x(d, obs, comb, False); xc, _ = _with_cats(xn, fr)
        blocks_num.append(xn.loc[pm].reset_index(drop=True)); blocks_cat.append(xc.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    vf = clear(d, hold); obs = vf[TARGET].where(~hold); qn, _ = _frame_x(d, obs, hold, False); qc, cats = _with_cats(qn, vf)
    rows = []; preds = []; y = d.loc[hold, "_truth"].to_numpy(float)
    for kind in ("base", "deep"):
        p = _fit_predict(blocks_num if kind == "base" else blocks_cat, ys,
                         (qn if kind == "base" else qc).loc[hold].reset_index(drop=True),
                         [] if kind == "base" else cats, kind)
        e = p - y; rows.append({"protocol": "random", "year": -1, "kind": kind, "n": len(y), "rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e)))})
        preds.append(pd.DataFrame({"protocol": "random", "year": d.loc[hold, "date"].dt.year.to_numpy(), "anon_polygon_id": d.loc[hold, "anon_polygon_id"].to_numpy(), "date": d.loc[hold, "date"].to_numpy(), "truth": y, "kind": kind, "pred": p}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def _proxy_2025(train: pd.DataFrame, private: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random holdout of visible private 2025 rows, using the full reference."""
    tr = train.copy(); pr = private.copy()
    tr["_origin"] = "train"; pr["_origin"] = "test"
    d = pd.concat([tr, pr], ignore_index=True, sort=False).reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date); d["year"] = d.year.fillna(d.date.dt.year).astype(int); d["doy"] = d.doy.fillna(d.date.dt.dayofyear).astype(int)
    d["_truth"] = d[TARGET].astype(float); d["is_synthetic_gap"] = d.get("is_synthetic_gap", False).fillna(False).astype(bool)
    target = d.date.dt.year.eq(2025) & d[TARGET].notna() & ~d.is_synthetic_gap
    ix_target = np.flatnonzero(target.to_numpy()); rng = np.random.default_rng(20250905)
    hold = np.zeros(len(d), bool)
    # AOI-stratified 15% holdout, ensuring each 2025 AOI has queries.
    for _, ix0 in d.loc[target].groupby("anon_polygon_id", sort=False).groups.items():
        ix = np.asarray(ix0, int); nn = max(1, int(round(.15 * len(ix)))); hold[rng.choice(ix, size=min(nn, len(ix)), replace=False)] = True
    blocks_num: list[pd.DataFrame] = []; blocks_cat: list[pd.DataFrame] = []; ys: list[pd.Series] = []
    for pm in _pseudo_masks(d, hold, 99001, .12, 1):
        comb = hold | pm; fr = clear(d, comb); obs = fr[TARGET].where(~comb)
        xn, _ = _frame_x(d, obs, comb, False); xc, _ = _with_cats(xn, fr)
        blocks_num.append(xn.loc[pm].reset_index(drop=True)); blocks_cat.append(xc.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    vf = clear(d, hold); obs = vf[TARGET].where(~hold); qn, _ = _frame_x(d, obs, hold, False); qc, cats = _with_cats(qn, vf)
    rows = []; preds = []; y = d.loc[hold, "_truth"].to_numpy(float)
    for kind in ("base", "deep"):
        p = _fit_predict(blocks_num if kind == "base" else blocks_cat, ys,
                         (qn if kind == "base" else qc).loc[hold].reset_index(drop=True),
                         [] if kind == "base" else cats, kind)
        e = p-y; rows.append({"protocol":"proxy2025","year":2025,"kind":kind,"n":len(y),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(np.abs(e)))})
        preds.append(pd.DataFrame({"protocol":"proxy2025","year":2025,"anon_polygon_id":d.loc[hold,"anon_polygon_id"].to_numpy(),"date":d.loc[hold,"date"].to_numpy(),"truth":y,"kind":kind,"pred":p}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    ex, ep = _exact_screen(train, private)
    print("exact done", round(time.time() - t0, 1), flush=True)
    rr, rp = _random_screen(train, private)
    print("random done", round(time.time() - t0, 1), flush=True)
    # Proxy is optional if the full reference is too large; run it here as a
    # single compact check requested by the experiment plan.
    pp, ppp = _proxy_2025(train, private)
    out = pd.concat([ex, rr, pp], ignore_index=True)
    preds = pd.concat([ep, rp, ppp], ignore_index=True)
    out.to_csv(RESEARCH / "catboost_screen_results.csv", index=False)
    preds.to_csv(RESEARCH / "catboost_screen_predictions.csv", index=False)
    agg = out.groupby(["protocol", "kind"], as_index=False).apply(
        lambda g: pd.Series({"n": int(g.n.sum()), "rmse_pooled": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))), "mae_pooled": float(np.average(g.mae, weights=g.n))}),
        include_groups=False,
    ).reset_index(drop=True).sort_values(["protocol", "rmse_pooled"])
    agg.to_csv(RESEARCH / "catboost_screen_aggregate.csv", index=False)
    (RESEARCH / "catboost_screen_report.md").write_text(
        "# CatBoost screen\n\n" + out.to_string(index=False) + "\n\n## pooled\n" + agg.to_string(index=False) + "\n",
        encoding="utf-8",
    )
    print(out.to_string(index=False)); print(agg.to_string(index=False)); print("elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()

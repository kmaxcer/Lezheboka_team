"""Robust post-correction sweep for the HGB/lag ensemble.

This is a diagnostic experiment only.  It never edits competition inputs or
the production ``outputs/model_dani_tuned*`` files.  Two proxy protocols are
used:

* random private-like masks (three seeds), with a dedicated 2025 slice;
* the actual private synthetic day-of-year pattern projected onto train years
  2019--2024 (``exact_hidden_doy``).

The script evaluates clipping, affine/bias calibration, shrinkage, date-peer
correction, and fixed/adaptive blends with the lag-aware local predictor.  All
parameters for a reported test partition are fitted on the other partitions
so that the sweep does not use the test labels.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from infer_lag import predict_private_lag  # noqa: E402
from validate import make_fold  # noqa: E402


DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi",
    "era5_temp_c", "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]

# The dominant MODIS acquisition cadence seen in the data.  This is a date
# feature (there is no hidden-row source available at inference time).
CANON_DOYS = frozenset(
    (97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289)
)


def _source(frame: pd.DataFrame) -> np.ndarray:
    return np.select(
        [frame["s2_ndvi"].notna(), frame["landsat_ndvi"].notna(), frame["modis_ndvi"].notna()],
        ["s2", "landsat", "modis"], default="none",
    )


def _mask_private(private: pd.DataFrame, seed: int, frac: float = 0.15):
    """Create a leakage-safe random mask and retain side-car truth metadata."""
    d = private.copy().sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["_true_src"] = _source(d)
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(int(seed))
    mask = np.zeros(len(d), dtype=bool)
    pool = d["primary_ndvi"].notna()
    years = d["date"].dt.year
    for _, ix in d.loc[pool].groupby(["anon_polygon_id", years], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        n = max(1, int(round(float(frac) * len(ii))))
        mask[rng.choice(ii, size=min(n, len(ii)), replace=False)] = True
    for col in DYNAMIC:
        if col in d.columns:
            d.loc[mask, col] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    return d, mask


def _span_and_peers(frame: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest same-AOI/year span and same-date observed peer count."""
    d = frame.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["_year_calc"] = d.date.dt.year.astype(int)
    d["_ord_calc"] = d.date.map(pd.Timestamp.toordinal).astype(float)
    known = d["primary_ndvi"].notna().to_numpy(bool) & ~d["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
    span = np.full(len(d), np.nan, dtype=float)
    ids = d["anon_polygon_id"].to_numpy(object)
    yrs = d["_year_calc"].to_numpy(int)
    xx = d["_ord_calc"].to_numpy(float)
    for _, ix in d.groupby(["anon_polygon_id", "_year_calc"], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        kk = ii[known[ii]]
        qq = ii[mask[ii]]
        if len(kk) == 0:
            continue
        kx = np.sort(xx[kk])
        for q in qq:
            pos = int(np.searchsorted(kx, xx[q], side="left"))
            left = xx[q] - kx[pos - 1] if pos > 0 else np.nan
            right = kx[pos] - xx[q] if pos < len(kx) else np.nan
            if np.isfinite(left) and np.isfinite(right):
                span[q] = left + right
            elif np.isfinite(left):
                span[q] = left
            elif np.isfinite(right):
                span[q] = right
    # Date-level peers are computed after masking; hidden rows cannot leak.
    date = d["date"]
    peer = d.loc[known].groupby(date.loc[known], observed=True)["primary_ndvi"].size()
    peers = date.map(peer).fillna(0).to_numpy(float)
    return span[mask], peers[mask]


def _metadata(
    frame: pd.DataFrame,
    mask: np.ndarray,
    train_ids: set[str],
    hidden_doys: dict[int, set[int]],
) -> pd.DataFrame:
    d = frame.copy()
    d["date"] = pd.to_datetime(d["date"])
    q = d.loc[mask, ["anon_polygon_id", "date", "_truth", "_true_src"]].copy()
    q["year"] = q["date"].dt.year.astype(int).to_numpy()
    q["doy"] = q["date"].dt.dayofyear.astype(int).to_numpy()
    q["canon"] = q["doy"].isin(CANON_DOYS).to_numpy()
    # ``validate.make_fold`` projects the private pattern by AOI using the
    # union of hidden DOYs across private years.  Use the same conservative
    # global DOY flag here; it is an observable date-only slice and avoids
    # pretending that a year-specific source schedule is known.
    hidden_union: set[int] = set()
    for vals in hidden_doys.values():
        hidden_union.update(int(v) for v in vals)
    q["hidden_doy"] = q["doy"].isin(hidden_union).astype(int).to_numpy()
    q["shared_id"] = q["anon_polygon_id"].isin(train_ids).astype(int).to_numpy()
    span, peers = _span_and_peers(d, mask)
    q["span"] = span
    q["peer_count"] = peers
    q["span_bin"] = pd.cut(
        q["span"], bins=[-np.inf, 4, 8, 14, 21, 35, 100, np.inf],
        labels=["<=4", "4-8", "8-14", "14-21", "21-35", "35-100", ">100"],
    ).astype(str)
    q["date_bin"] = (q["doy"] // 16).astype(int)
    return q.reset_index(drop=True)


def _attach_prediction(meta: pd.DataFrame, pred: pd.DataFrame, name: str) -> pd.Series:
    p = pred.copy()
    p["date"] = pd.to_datetime(p["date"])
    z = meta[["anon_polygon_id", "date"]].merge(
        p[["anon_polygon_id", "date", "primary_ndvi_pred"]],
        on=["anon_polygon_id", "date"], how="left", validate="one_to_one",
    )
    if z["primary_ndvi_pred"].isna().any():
        raise ValueError(f"missing {name} predictions")
    return z["primary_ndvi_pred"].astype(float)


def _metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return np.nan, np.nan
    e = p[ok] - y[ok]
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def _fit_affine(train: pd.DataFrame, robust: bool = False) -> tuple[float, float]:
    x = train["hgb"].to_numpy(float)
    y = train["_truth"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return 0.0, 1.0
    if robust:
        # Winsorized OLS is deterministic and less sensitive to a few extreme
        # synthetic proxy labels than a full unbounded regression.
        lo, hi = np.quantile(y[ok], [0.01, 0.99])
        yy = np.clip(y[ok], lo, hi)
        xx = x[ok]
    else:
        xx, yy = x[ok], y[ok]
    coef = np.linalg.lstsq(np.c_[np.ones(len(xx)), xx], yy, rcond=None)[0]
    a, b = float(coef[0]), float(coef[1])
    return float(np.clip(a, -0.08, 0.08)), float(np.clip(b, 0.85, 1.15))


def _fit_group_bias(train: pd.DataFrame, key: str, min_n: int = 40, robust: bool = True) -> dict[object, float]:
    z = train.copy()
    z["_resid"] = z["_truth"].to_numpy(float) - z["hgb"].to_numpy(float)
    out: dict[object, float] = {}
    for k, g in z.groupby(key, dropna=False, observed=True):
        if len(g) < min_n:
            continue
        val = float(g["_resid"].median() if robust else g["_resid"].mean())
        out[k] = float(np.clip(val, -0.04, 0.04))
    return out


def _fit_blend_weights(train: pd.DataFrame, key: str | None, min_n: int = 60) -> dict[object, float] | float:
    """Fit lag blend weight by least squares, clipped conservatively."""
    def one(g: pd.DataFrame) -> float:
        a = g["lag"].to_numpy(float) - g["hgb"].to_numpy(float)
        r = g["_truth"].to_numpy(float) - g["hgb"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(r)
        if ok.sum() < min_n or np.sum(a[ok] ** 2) < 1e-10:
            return np.nan
        # p = hgb + w*(lag-hgb), minimize squared error.
        return float(np.clip(np.sum(a[ok] * r[ok]) / np.sum(a[ok] ** 2), 0.0, 0.5))
    if key is None:
        w = one(train)
        return float(0.0 if not np.isfinite(w) else w)
    out: dict[object, float] = {}
    for k, g in train.groupby(key, dropna=False, observed=True):
        w = one(g)
        if np.isfinite(w):
            out[k] = w
    return out


def _date_peer_prediction(frame: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    d = frame.copy(); d["date"] = pd.to_datetime(d["date"])
    known = d["primary_ndvi"].notna().to_numpy(bool) & ~d["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
    vals = d.loc[known].groupby("date", observed=True)["primary_ndvi"].median()
    return d.loc[mask, "date"].map(vals).astype(float).reset_index(drop=True)


def _add_random_sets(
    private: pd.DataFrame,
    train: pd.DataFrame,
    train_ids: set[str],
    hidden_doys: dict[int, set[int]],
) -> list[pd.DataFrame]:
    sets: list[pd.DataFrame] = []
    for seed in (0, 1, 2):
        frame, mask = _mask_private(private, seed)
        meta = _metadata(frame, mask, train_ids, hidden_doys)
        # Precomputed HGB files were produced by the canonical hgb_cv protocol.
        hgb_path = RESEARCH / f"hgb_cv_pred_seed{seed}.csv"
        if not hgb_path.exists():
            raise FileNotFoundError(hgb_path)
        hgb = pd.read_csv(hgb_path)
        meta["hgb"] = _attach_prediction(meta, hgb, "hgb").to_numpy()
        lag_path = RESEARCH / f"teammate_sweep_postcorr_lag_random{seed}.csv"
        if lag_path.exists():
            lag = pd.read_csv(lag_path)
        else:
            lag = predict_private_lag(
                frame, train=train, k=16, degree=3, bin_days=30,
                use_date_prior=True, date_weight=1.0,
            )
            lag.to_csv(lag_path, index=False, float_format="%.8f")
        meta["lag"] = _attach_prediction(meta, lag, "lag").to_numpy()
        meta["peer"] = _date_peer_prediction(frame, mask).to_numpy()
        meta["partition"] = f"random{seed}"
        sets.append(meta)
    return sets


def _add_exact_set(
    train: pd.DataFrame,
    private: pd.DataFrame,
    train_ids: set[str],
    hidden_doys: dict[int, set[int]],
) -> pd.DataFrame:
    # exact_compare_preds contains the already-computed HGB and local outputs
    # for the real private hidden-DOY projection onto train years.
    ex = pd.read_csv(RESEARCH / "exact_compare_preds.csv", parse_dates=["date"])
    pieces: list[pd.DataFrame] = []
    for year, g in ex.groupby("year", sort=True):
        # Reconstruct the fold solely to derive observable span/peer features.
        fold, _truth = make_fold(train.copy(), private.copy(), int(year))
        # Evaluation-only source annotation recovered before masking.  The
        # production estimator never sees this field; it is used only for
        # diagnostics/slice reporting.
        fold["_true_src"] = _source(fold)
        syn = fold["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
        q = fold.loc[syn, ["anon_polygon_id", "date", "_truth"]].copy()
        q["_true_src"] = fold.loc[syn, "_true_src"].to_numpy()
        # Keep only keys represented in exact_compare_preds (defensive against
        # changes to the validation helper).
        q["date"] = pd.to_datetime(q["date"])
        q = q.merge(g[["anon_polygon_id", "date"]], on=["anon_polygon_id", "date"], how="inner", validate="one_to_one")
        keys = pd.MultiIndex.from_frame(fold[["anon_polygon_id", "date"]])
        q_idx = fold.set_index(["anon_polygon_id", "date"]).index.isin(pd.MultiIndex.from_frame(q[["anon_polygon_id", "date"]]))
        # Build a compact mask aligned to the full fold for _metadata.
        mm = fold[["anon_polygon_id", "date"]].apply(tuple, axis=1).isin(set(map(tuple, q[["anon_polygon_id", "date"]].to_numpy()))).to_numpy()
        m = _metadata(fold, mm, train_ids, hidden_doys)
        # Merge predictions by key; use source from the original exact file,
        # which is known before masking and is only an evaluation annotation.
        m = m.merge(g[["anon_polygon_id", "date", "_truth", "_true_src", "hgb", "lag_k16_d3"]],
                    on=["anon_polygon_id", "date"], how="inner", suffixes=("", "_g"), validate="one_to_one")
        # ``_truth``/``_true_src`` overlap with metadata and receive the
        # explicit ``_g`` suffix; predictions do not overlap and keep their
        # original names.
        if "_truth_g" in m:
            m["_truth"] = m["_truth_g"]
        if "_true_src_g" in m:
            m["_true_src"] = m["_true_src_g"]
        if "hgb_g" in m:
            m["hgb"] = m["hgb_g"]
        m["lag"] = m["lag_k16_d3"]
        m = m.drop(columns=[c for c in ["_truth_g", "_true_src_g", "hgb_g"] if c in m])
        # Recompute date peers for this exact fold, then align by key.
        peer = _date_peer_prediction(fold, mm)
        pk = fold.loc[mm, ["anon_polygon_id", "date"]].copy(); pk["peer"] = peer.to_numpy()
        m = m.drop(columns=["peer"], errors="ignore").merge(pk, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
        m["partition"] = f"exact{int(year)}"
        pieces.append(m)
    return pd.concat(pieces, ignore_index=True)


def _predict_methods(train_part: pd.DataFrame, test_part: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit safe corrections on train_part and return predictions for test_part."""
    y = test_part["_truth"].to_numpy(float)
    h = test_part["hgb"].to_numpy(float)
    l = test_part["lag"].to_numpy(float)
    out: dict[str, np.ndarray] = {"hgb_raw": h.copy()}

    # Clipping guards.  HGB outputs are already inside the broad guard, but we
    # keep these explicit to document the robustness check.
    out["hgb_clip_01"] = np.clip(h, 0.0, 1.0)
    out["hgb_clip_02_11"] = np.clip(h, -0.2, 1.1)
    out["hgb_clip_calib_q01_q99"] = np.clip(h, *np.quantile(train_part["hgb"].to_numpy(float), [0.01, 0.99]))

    a, b = _fit_affine(train_part, robust=False)
    ar, br = _fit_affine(train_part, robust=True)
    out["hgb_affine"] = np.clip(a + b * h, -0.5, 1.2)
    out["hgb_affine_winsor"] = np.clip(ar + br * h, -0.5, 1.2)
    bias = float(np.clip((train_part["_truth"] - train_part["hgb"]).median(), -0.04, 0.04))
    out["hgb_bias_median"] = np.clip(h + bias, -0.5, 1.2)

    # Conservative shrinkage around a calibration mean.  Lambda=1 is the raw
    # model; lower values test whether HGB variance is slightly too wide.
    mu = float(train_part["_truth"].mean())
    for lam in (0.90, 0.95, 0.98):
        out[f"hgb_shrink_{lam:.2f}"] = np.clip(mu + lam * (h - mu), -0.5, 1.2)

    # Date-level peer median is observable at inference and targets a common
    # acquisition-date anomaly.  Missing peer dates retain the HGB value.
    peer = test_part["peer"].to_numpy(float)
    valid_peer = np.isfinite(peer)
    for w in (0.05, 0.10, 0.20, 0.30):
        p = h.copy(); p[valid_peer] = (1.0 - w) * h[valid_peer] + w * peer[valid_peer]
        out[f"hgb_peer_{w:.2f}"] = p

    # Fixed lag blends and cross-fitted adaptive weights.
    for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        out[f"blend_lag_{w:.2f}"] = (1.0 - w) * h + w * l
    w0 = float(_fit_blend_weights(train_part, None))
    out["blend_lag_fit_global"] = (1.0 - w0) * h + w0 * l
    for key, label in [("canon", "canon"), ("hidden_doy", "hidden_doy"), ("span_bin", "span")]:
        ws = _fit_blend_weights(train_part, key, min_n=60)
        p = h.copy()
        if isinstance(ws, dict):
            for k, ww in ws.items():
                take = test_part[key].eq(k).to_numpy()
                p[take] = (1.0 - ww) * h[take] + ww * l[take]
        out[f"blend_lag_fit_{label}"] = p

    # Group residual corrections learned from other partitions.  They are
    # deliberately bounded and require enough calibration observations.
    for key, label in [("canon", "canon"), ("hidden_doy", "hidden_doy"), ("span_bin", "span"), ("year", "year")]:
        vals = _fit_group_bias(train_part, key, min_n=60, robust=True)
        p = h.copy()
        for k, v in vals.items():
            take = test_part[key].eq(k).to_numpy(); p[take] += v
        out[f"hgb_groupbias_{label}"] = np.clip(p, -0.5, 1.2)

    return out


def _run_crossfit(parts: list[pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for ti, test in enumerate(parts):
        train = pd.concat([p for j, p in enumerate(parts) if j != ti], ignore_index=True)
        methods = _predict_methods(train, test)
        base = test[["partition", "anon_polygon_id", "date", "_truth", "year", "doy", "canon", "hidden_doy", "span_bin", "shared_id"]].copy()
        for name, pred in methods.items():
            rm, mae = _metric(test["_truth"].to_numpy(float), pred)
            rec = {"protocol": "crossfit", "partition": str(test["partition"].iloc[0]), "method": name,
                   "n": len(test), "rmse": rm, "mae": mae}
            for sl, take in {
                "2025": test["year"].eq(2025).to_numpy(),
                "hidden_doy": test["hidden_doy"].astype(bool).to_numpy(),
                "non_hidden_doy": ~test["hidden_doy"].astype(bool).to_numpy(),
                "canon": test["canon"].astype(bool).to_numpy(),
                "noncanon": ~test["canon"].astype(bool).to_numpy(),
                "shared": test["shared_id"].astype(bool).to_numpy(),
                "private_only": ~test["shared_id"].astype(bool).to_numpy(),
            }.items():
                if np.any(take):
                    rr, aa = _metric(test.loc[take, "_truth"].to_numpy(float), pred[take]); rec[f"rmse_{sl}"] = rr; rec[f"mae_{sl}"] = aa; rec[f"n_{sl}"] = int(take.sum())
                else:
                    rec[f"rmse_{sl}"] = np.nan; rec[f"mae_{sl}"] = np.nan; rec[f"n_{sl}"] = 0
            records.append(rec)
            # Keep only a compact set of predictions needed for independent
            # inspection; full method grid is still represented in results.
            if name in {"hgb_raw", "hgb_affine", "hgb_affine_winsor", "hgb_bias_median", "hgb_shrink_0.95", "hgb_peer_0.10", "blend_lag_0.10", "blend_lag_0.20", "blend_lag_fit_global", "blend_lag_fit_canon", "blend_lag_fit_hidden_doy", "blend_lag_fit_span", "hgb_groupbias_canon", "hgb_groupbias_span", "hgb_clip_01"}:
                z = base.copy(); z["method"] = name; z["pred"] = pred; pred_rows.append(z)
    return pd.DataFrame(records), pd.concat(pred_rows, ignore_index=True)


def _oracle_summary(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Report optimistic fixed-weight context (diagnostic, not deployment)."""
    rows: list[dict[str, object]] = []
    allp = pd.concat(parts, ignore_index=True)
    for name, arr in [("hgb", allp.hgb.to_numpy()), ("lag", allp.lag.to_numpy())]:
        rm, ma = _metric(allp._truth.to_numpy(), arr); rows.append({"scope": "pooled_oracle_context", "method": name, "rmse": rm, "mae": ma})
    for w in np.linspace(0, 0.5, 21):
        arr = (1 - w) * allp.hgb.to_numpy() + w * allp.lag.to_numpy(); rm, ma = _metric(allp._truth.to_numpy(), arr)
        rows.append({"scope": "pooled_oracle_context", "method": f"fixed_lag_{w:.2f}", "rmse": rm, "mae": ma})
    return pd.DataFrame(rows).sort_values("rmse")


def main() -> None:
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    train_ids = set(train["anon_polygon_id"].astype(str))
    hidden = private.loc[private["is_synthetic_gap"].fillna(False).astype(bool)].copy()
    hidden["year"] = hidden["date"].dt.year.astype(int); hidden["doy"] = hidden["date"].dt.dayofyear.astype(int)
    hidden_doys = {int(y): set(g["doy"].astype(int)) for y, g in hidden.groupby("year")}

    random_parts = _add_random_sets(private, train, train_ids, hidden_doys)
    exact = _add_exact_set(train, private, train_ids, hidden_doys)
    exact_parts = [g.reset_index(drop=True) for _, g in exact.groupby("partition", sort=True)]

    # Random and exact protocols are evaluated independently; the latter is
    # intentionally weighted by year rather than mixed into random calibration.
    random_results, random_preds = _run_crossfit(random_parts)
    exact_results, exact_preds = _run_crossfit(exact_parts)
    random_results["dataset"] = "random_private_like"
    exact_results["dataset"] = "exact_hidden_doy"
    results = pd.concat([random_results, exact_results], ignore_index=True)
    results.to_csv(RESEARCH / "teammate_sweep_postcorr_results.csv", index=False)
    pd.concat([random_preds.assign(dataset="random_private_like"), exact_preds.assign(dataset="exact_hidden_doy")], ignore_index=True).to_csv(
        RESEARCH / "teammate_sweep_postcorr_preds.csv", index=False
    )
    oracle = pd.concat([_oracle_summary(random_parts).assign(dataset="random_private_like"), _oracle_summary(exact_parts).assign(dataset="exact_hidden_doy")], ignore_index=True)
    oracle.to_csv(RESEARCH / "teammate_sweep_postcorr_oracle.csv", index=False)

    # Compact aggregate table: pooled RMSE is reconstructed from per-partition
    # MSE and sample counts, avoiding a misleading mean of RMSEs.
    agg_rows = []
    for (dataset, method), g in results.groupby(["dataset", "method"], sort=False):
        ok = g["rmse"].notna()
        pooled = float(np.sqrt(np.average(g.loc[ok, "rmse"] ** 2, weights=g.loc[ok, "n"]))) if ok.any() else np.nan
        row = {"dataset": dataset, "method": method, "n": int(g.n.sum()), "rmse_pooled": pooled, "rmse_mean": float(g.loc[ok, "rmse"].mean()) if ok.any() else np.nan, "mae_mean": float(g.loc[ok, "mae"].mean()) if ok.any() else np.nan}
        for sl in ("2025", "hidden_doy", "non_hidden_doy", "canon", "noncanon", "shared", "private_only"):
            q = g[g[f"n_{sl}"] > 0]
            if len(q): row[f"rmse_{sl}"] = float(np.sqrt(np.average(q[f"rmse_{sl}"] ** 2, weights=q[f"n_{sl}"])))
            else: row[f"rmse_{sl}"] = np.nan
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values(["dataset", "rmse_pooled"])
    agg.to_csv(RESEARCH / "teammate_sweep_postcorr_aggregate.csv", index=False)

    lines = [
        "# Robust post-correction sweep",
        "",
        "This diagnostic does not modify input CSVs or `outputs/model_dani_tuned*`.",
        "Parameters for each tested partition were fitted only on the other partitions.",
        "Random protocol: 15% known private rows per AOI/year, seeds 0/1/2.",
        "Exact protocol: actual private synthetic DOYs projected onto train years 2019--2024.",
        "",
        "## Best cross-fitted variants",
        "",
        agg.groupby("dataset", sort=False).head(15).to_string(index=False),
        "",
        "## Optimistic context-only fixed blend (not used for deployment)",
        "",
        oracle.groupby("dataset", sort=False).head(8).to_string(index=False),
        "",
        "Interpretation: prefer a variant only when its cross-fitted pooled RMSE beats `hgb_raw` on both protocols or has a clearly stable slice improvement. Clipping is retained as a guard, not evidence of gain.",
    ]
    (ROOT / "reports" / "teammate_sweep_postcorr_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", RESEARCH / "teammate_sweep_postcorr_aggregate.csv")
    print(agg.groupby("dataset", sort=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()

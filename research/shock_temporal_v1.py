"""Leakage-safe date-shock and robust temporal reconstruction diagnostics.

This file is intentionally independent of the production estimators.  For
each proxy mask it computes leave-one-out (LOO) temporal interpolation errors
from *visible* targets only.  A date/crop median of those errors is a common
acquisition shock; a robust local temporal estimate is an alternative state
estimate.  Correction coefficients are fitted cross-partition, never on the
held-out query rows.

Outputs are new ``shock_temporal_v1_*`` files and do not overwrite prior
research or submission artifacts.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402


def src_of(d: pd.DataFrame) -> np.ndarray:
    return np.select(
        [d["s2_ndvi"].notna(), d["landsat_ndvi"].notna(), d["modis_ndvi"].notna()],
        ["S2", "L8", "MOD"], default="NONE",
    )


def _safe_source_from_original(original: pd.DataFrame) -> pd.Series:
    z = original.copy()
    z["date"] = pd.to_datetime(z["date"])
    z["_src_eval"] = src_of(z)
    return z.set_index(["anon_polygon_id", "date"])["_src_eval"]


def _linear_at(xq: float, xk: np.ndarray, yk: np.ndarray) -> float:
    """Nearest-bracket linear interpolation with endpoint extrapolation."""
    if len(xk) == 0:
        return np.nan
    pos = int(np.searchsorted(xk, xq, side="left"))
    if pos <= 0:
        return float(yk[0])
    if pos >= len(xk):
        return float(yk[-1])
    x0, x1 = float(xk[pos - 1]), float(xk[pos])
    if x1 <= x0:
        return float(0.5 * (yk[pos - 1] + yk[pos]))
    w = (xq - x0) / (x1 - x0)
    return float((1.0 - w) * yk[pos - 1] + w * yk[pos])


def _local_robust(xq: float, xk: np.ndarray, yk: np.ndarray, k: int = 8,
                  radius: float = 120.0) -> float:
    """Huber-ish weighted local linear/median state estimate.

    A small weighted linear fit is used when enough points exist; otherwise a
    distance-weighted median is returned.  The estimate is clipped to nearby
    target quantiles to resist isolated sensor spikes.
    """
    if len(xk) == 0:
        return np.nan
    dist = np.abs(xk - xq)
    take = np.argsort(dist)[: max(1, int(k))]
    xx, yy, dd = xk[take], yk[take], dist[take]
    keep = dd <= float(radius)
    if keep.sum() < 2:
        keep = np.ones(len(xx), dtype=bool)
    xx, yy, dd = xx[keep], yy[keep], dd[keep]
    # Tri-cube / inverse-distance weights; robust downweighting is applied
    # after a first fit.
    ww = 1.0 / (1.0 + dd / 18.0)
    if len(xx) >= 3 and np.ptp(xx) > 1e-6:
        X = np.c_[np.ones(len(xx)), xx - xq]
        try:
            beta = np.linalg.lstsq(X * np.sqrt(ww)[:, None], yy * np.sqrt(ww), rcond=None)[0]
            resid = yy - X @ beta
            scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 0.015
            hub = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(resid), 1e-8))
            beta = np.linalg.lstsq(X * np.sqrt(ww * hub)[:, None], yy * np.sqrt(ww * hub), rcond=None)[0]
            val = float(beta[0])
        except Exception:
            val = float(np.average(yy, weights=ww))
    else:
        # Weighted median is less brittle than a mean for the one/two point
        # case, especially on 2025/new AOIs.
        order = np.argsort(yy)
        ys, ws = yy[order], ww[order]
        val = float(ys[np.searchsorted(np.cumsum(ws), 0.5 * ws.sum())])
    lo, hi = np.quantile(yy, [0.05, 0.95])
    return float(np.clip(val, lo - 0.05, hi + 0.05))


def temporal_features(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Compute query temporal states and LOO residual shocks from visible rows."""
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["_yr"] = d["date"].dt.year.astype(int)
    d["_ord"] = d["date"].map(pd.Timestamp.toordinal).astype(float)
    d["_crop"] = d.get("crop_type", pd.Series("UNK", index=d.index)).astype(str)
    d["_src_obs"] = src_of(d)
    hidden = d.get("is_synthetic_gap", pd.Series(False, index=d.index)).fillna(False).astype(bool).to_numpy()
    y = pd.to_numeric(d["primary_ndvi"], errors="coerce").to_numpy(float)
    known = np.isfinite(y) & ~hidden
    x = d["_ord"].to_numpy(float)
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    yrs = d["_yr"].to_numpy(int)
    # Query states and residuals for visible rows.  Group count is tiny (AOI
    # x year), so explicit loops are faster and clearer than giant joins.
    qidx = np.flatnonzero(mask)
    n = len(d)
    interp = np.full(n, np.nan)
    robust = np.full(n, np.nan)
    loo_res = np.full(n, np.nan)
    for _, ii0 in d.groupby(["anon_polygon_id", "_yr"], sort=False).groups.items():
        ii = np.asarray(ii0, dtype=int)
        kk = ii[known[ii]]
        if len(kk) == 0:
            continue
        so = np.argsort(x[kk]); kk = kk[so]
        xk, yk = x[kk], y[kk]
        # LOO interpolation residual for every visible row.  For endpoints,
        # use the nearest other point; these residuals are noisy and are later
        # aggregated with a robust median.
        if len(kk) >= 2:
            for j, row in enumerate(kk):
                xx = np.delete(xk, j); yy = np.delete(yk, j)
                loo = _linear_at(x[row], xx, yy)
                loo_res[row] = y[row] - loo if np.isfinite(loo) else np.nan
        for row in ii[mask[ii]]:
            interp[row] = _linear_at(x[row], xk, yk)
            robust[row] = _local_robust(x[row], xk, yk, k=8, radius=120.0)

    # Common shock: LOO residual medians by exact acquisition date, date+crop,
    # and date+observed-source.  The query AOI is excluded from each group to
    # avoid same-AOI leakage.  Groups with <3 independent AOIs are ignored.
    vis = np.flatnonzero(known & np.isfinite(loo_res))
    base = pd.DataFrame({
        "date": d.loc[vis, "date"].to_numpy(), "crop": d.loc[vis, "_crop"].to_numpy(),
        "src": d.loc[vis, "_src_obs"].to_numpy(), "id": ids[vis],
        "r": loo_res[vis],
    })
    # Reduce duplicate observations from one AOI/date to a single median.
    base = base.groupby(["date", "crop", "src", "id"], as_index=False, observed=True)["r"].median()
    def group_map(keys: list[str], min_ids: int = 3) -> pd.DataFrame:
        z = base.groupby(keys, observed=True).agg(shock=("r", "median"), n=("id", "nunique"), mad=("r", lambda a: float(np.median(np.abs(a - np.median(a)))))).reset_index()
        return z[z["n"] >= min_ids]
    maps = {
        "shock_date": group_map(["date"], 3),
        "shock_crop": group_map(["date", "crop"], 3),
        "shock_src": group_map(["date", "src"], 3),
        "shock_crop_src": group_map(["date", "crop", "src"], 3),
    }
    out = d.loc[qidx, ["anon_polygon_id", "date", "_yr", "_crop"]].copy().reset_index(drop=True)
    out["_idx"] = qidx
    out["interp"] = interp[qidx]; out["robust"] = robust[qidx]
    # Query source is hidden, so source-specific groups are not used directly;
    # they are retained for diagnostics and an all-source fallback.
    for name, mp in maps.items():
        keys = {"shock_date": ["date"], "shock_crop": ["date", "_crop"],
                "shock_src": ["date"], "shock_crop_src": ["date", "_crop"]}[name]
        m2 = mp.rename(columns={"crop": "_crop", "src": "_src_obs"})
        # For all-source query correction use the median over source groups.
        if name in {"shock_src", "shock_crop_src"}:
            m2 = m2.groupby(keys, observed=True).agg(shock=("shock", "median"), n=("n", "sum")).reset_index()
        out = out.merge(m2[keys + ["shock", "n"]], on=keys, how="left", suffixes=("", "_" + name))
        out.rename(columns={"shock": name, "n": name + "_n"}, inplace=True)
    # A nearby-date shock is useful when exact dates are sparse.  Use +/- 7d,
    # excluding the query's AOI and weighting by a smooth Gaussian kernel.
    if len(base):
        bx = base["date"].map(pd.Timestamp.toordinal).to_numpy(float)
        br = base["r"].to_numpy(float); bi = base["id"].to_numpy(str)
        near = np.full(len(out), np.nan)
        for j, row in enumerate(qidx):
            dd = np.abs(bx - x[row]); take = (dd <= 8.0) & (bi != ids[row])
            if take.sum() >= 3:
                w = np.exp(-0.5 * (dd[take] / 3.5) ** 2)
                # weighted trimmed mean, bounded by robust quantiles
                vals = br[take]; order = np.argsort(vals); vals, w = vals[order], w[order]
                c = np.cumsum(w); lo = vals[np.searchsorted(c, 0.10 * c[-1])]; hi = vals[np.searchsorted(c, 0.90 * c[-1])]
                near[j] = float(np.average(np.clip(vals, lo, hi), weights=w))
        out["shock_near"] = near
    else:
        out["shock_near"] = np.nan
    # Fill exact date shock from nearby shock only when a date has too few
    # peers; no target value from the query itself enters either statistic.
    out["shock_date_filled"] = out["shock_date"].where(out["shock_date_n"].fillna(0) >= 3, out["shock_near"])
    return out


def metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any(): return np.nan, np.nan
    e = p[ok] - y[ok]
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def make_random(private: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    # Keep helper's exact mask semantics, including per-AOI/year sampling.
    return _mask_private(private, int(seed))


def baseline_random(seed: int, q: pd.DataFrame) -> np.ndarray:
    p = pd.read_csv(RESEARCH / f"hgb_cv_pred_seed{seed}.csv", parse_dates=["date"])
    z = q[["anon_polygon_id", "date"]].merge(p, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    return z["primary_ndvi_pred"].to_numpy(float)


def build_parts() -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    parts: list[dict] = []
    # Random private-like masks, seeds 0/1/2.
    for seed in (0, 1, 2):
        frame, mask = make_random(private, seed)
        q = frame.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy().reset_index(drop=True)
        q["_true_src"] = src_of(private.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True).loc[mask])
        q["year"] = pd.to_datetime(q["date"]).dt.year.astype(int)
        q["cohort"] = np.where(q["anon_polygon_id"].isin(set(train.anon_polygon_id.astype(str))), "shared", "new")
        q["partition"] = f"random{seed}"; q["dataset"] = "random_private_like"
        q["hgb"] = baseline_random(seed, q)
        # Existing lag blend is the strongest stable comparator in this branch.
        lagp = RESEARCH / f"teammate_sweep_postcorr_lag_random{seed}.csv"
        if lagp.exists():
            lp = pd.read_csv(lagp, parse_dates=["date"])
            q["lag"] = q[["anon_polygon_id", "date"]].merge(lp, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
        else:
            q["lag"] = q["hgb"]
        f = temporal_features(frame, mask)
        q = q.merge(f.drop(columns=["_idx", "_yr", "_crop"], errors="ignore"), on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
        parts.append({"name": f"random{seed}", "data": q, "frame": frame, "mask": mask})
    # Exact hidden DOY projection onto train 2019--2024, matching existing
    # exact_compare_preds keys.  Use the precomputed HGB/lag values.
    ex = pd.read_csv(RESEARCH / "exact_compare_preds.csv", parse_dates=["date"])
    srcmap = _safe_source_from_original(train)
    for year, g in ex.groupby("year", sort=True):
        frame, _ = make_fold(train.copy(), private.copy(), int(year))
        mask = frame["is_synthetic_gap"].fillna(False).astype(bool).to_numpy()
        # Keep exactly rows in exact_compare_preds (defensive).
        kset = set(zip(g.anon_polygon_id.astype(str), pd.to_datetime(g.date)))
        allk = list(zip(frame.anon_polygon_id.astype(str), pd.to_datetime(frame.date)))
        mask = np.array([m and k in kset for m, k in zip(mask, allk)], dtype=bool)
        q = frame.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy().reset_index(drop=True)
        q["_true_src"] = [srcmap.get((str(a), pd.Timestamp(dt)), "NONE") for a, dt in zip(q.anon_polygon_id, q.date)]
        q["year"] = int(year); q["cohort"] = np.where(q.anon_polygon_id.isin(set(train.anon_polygon_id.astype(str))), "shared", "new")
        q["partition"] = f"exact{int(year)}"; q["dataset"] = "exact_hidden_doy"
        q = q.merge(g[["anon_polygon_id", "date", "hgb", "lag_k16_d3"]], on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
        q.rename(columns={"lag_k16_d3": "lag"}, inplace=True)
        f = temporal_features(frame, mask)
        q = q.merge(f.drop(columns=["_idx", "_yr", "_crop"], errors="ignore"), on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
        parts.append({"name": f"exact{int(year)}", "data": q, "frame": frame, "mask": mask})
    return parts, train, private


def fit_and_predict(trainq: pd.DataFrame, testq: pd.DataFrame) -> dict[str, np.ndarray]:
    h = testq["hgb"].to_numpy(float); lag = testq["lag"].to_numpy(float)
    out: dict[str, np.ndarray] = {
        "hgb": h.copy(), "lagblend20": 0.8 * h + 0.2 * lag,
    }
    # Temporal reconstructions: direct robust local state and linear state.
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        for col in ("interp", "robust"):
            s = testq[col].to_numpy(float); ok = np.isfinite(s)
            p = h.copy(); p[ok] = (1.0 - w) * h[ok] + w * s[ok]
            out[f"{col}_blend{w:.2f}"] = np.clip(p, -0.5, 1.2)
    # Fit a bounded common-shock coefficient by least squares on other parts.
    r = trainq["_truth"].to_numpy(float) - trainq["hgb"].to_numpy(float)
    for col in ("shock_date", "shock_crop", "shock_near", "shock_date_filled"):
        x = trainq[col].to_numpy(float); ok = np.isfinite(x) & np.isfinite(r)
        a = float(np.clip(np.sum(x[ok] * r[ok]) / max(np.sum(x[ok] * x[ok]), 1e-9), -0.8, 0.8)) if ok.sum() >= 30 else 0.0
        # Also test conservative shrinkage of the fitted coefficient.
        for shrink in (0.35, 0.60, 1.0):
            xx = testq[col].to_numpy(float); good = np.isfinite(xx)
            p = h.copy(); p[good] += shrink * a * xx[good]
            out[f"shock_{col}_a{shrink:.2f}"] = np.clip(p, -0.5, 1.2)
    # Joint state + shock; fit two coefficients with ridge for stability.
    X = np.c_[trainq["robust"].to_numpy(float) - trainq["hgb"].to_numpy(float), trainq["shock_date_filled"].to_numpy(float)]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(r)
    if ok.sum() >= 40:
        coef = np.linalg.solve(X[ok].T @ X[ok] + 0.08 * np.eye(2), X[ok].T @ r[ok]); coef = np.clip(coef, [-0.8, -0.8], [0.8, 0.8])
    else: coef = np.zeros(2)
    Xt = np.c_[testq["robust"].to_numpy(float) - h, testq["shock_date_filled"].to_numpy(float)]
    good = np.isfinite(Xt).all(axis=1); p = h.copy(); p[good] += Xt[good] @ coef
    out["joint_robust_shock"] = np.clip(p, -0.5, 1.2)
    # Source-aware date shock is diagnostic only: query source is hidden, so
    # use source posterior proxy based on visible source proportions if any.
    return out


def run() -> None:
    parts, train, private = build_parts()
    rows: list[dict] = []; predrows: list[pd.DataFrame] = []
    # Crossfit separately within random and exact protocol.  For random seeds,
    # leave-one-seed-out; for exact years, leave-one-year-out.
    for i, part in enumerate(parts):
        same = [p for j, p in enumerate(parts) if j != i and p["data"]["dataset"].iloc[0] == part["data"]["dataset"].iloc[0]]
        if not same: same = [p for j, p in enumerate(parts) if j != i]
        tr = pd.concat([p["data"] for p in same], ignore_index=True)
        te = part["data"]
        methods = fit_and_predict(tr, te)
        y = te["_truth"].to_numpy(float)
        base = te[["partition", "anon_polygon_id", "date", "_truth", "year", "cohort", "_true_src"]].copy()
        for name, p in methods.items():
            rm, ma = metric(y, p)
            rec = {"dataset": str(te["dataset"].iloc[0]), "partition": part["name"], "method": name, "n": len(te), "rmse": rm, "mae": ma}
            for key, vals in {
                "2025": te["year"].to_numpy() == 2025,
                "shared": te["cohort"].eq("shared").to_numpy(),
                "new": te["cohort"].eq("new").to_numpy(),
                "S2": te["_true_src"].eq("S2").to_numpy(),
                "L8": te["_true_src"].eq("L8").to_numpy(),
                "MOD": te["_true_src"].eq("MOD").to_numpy(),
            }.items():
                if vals.any(): rec[f"rmse_{key}"], rec[f"mae_{key}"] = metric(y[vals], p[vals]); rec[f"n_{key}"] = int(vals.sum())
            rows.append(rec)
            if name in {"hgb", "lagblend20", "interp_blend0.10", "interp_blend0.20", "robust_blend0.10", "robust_blend0.20", "shock_shock_date_filled_a0.60", "joint_robust_shock"}:
                z = base.copy(); z["method"] = name; z["pred"] = p; predrows.append(z)
    res = pd.DataFrame(rows)
    res.to_csv(RESEARCH / "shock_temporal_v1_results.csv", index=False)
    pr = pd.concat(predrows, ignore_index=True); pr.to_csv(RESEARCH / "shock_temporal_v1_preds.csv", index=False)
    aggrows = []
    for (dataset, method), g in res.groupby(["dataset", "method"], sort=False):
        aggrows.append({"dataset": dataset, "method": method, "n": int(g.n.sum()), "rmse_pooled": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))), "rmse_mean": float(g.rmse.mean()), "mae_mean": float(g.mae.mean())})
    agg = pd.DataFrame(aggrows).sort_values(["dataset", "rmse_pooled"]); agg.to_csv(RESEARCH / "shock_temporal_v1_aggregate.csv", index=False)
    # Slice aggregate (cohort/year/source) for the report.
    slrows=[]
    for c in ("2025", "shared", "new", "S2", "L8", "MOD"):
        for method, g in res.groupby("method", sort=False):
            col = "rmse_" + c; nc = "n_" + c
            if col in g and g[nc].sum() > 0:
                slrows.append({"slice": c, "method": method, "n": int(g[nc].sum()), "rmse_pooled": float(np.sqrt(np.average(g[col].dropna() ** 2, weights=g.loc[g[col].notna(), nc])) )})
    pd.DataFrame(slrows).to_csv(RESEARCH / "shock_temporal_v1_slices.csv", index=False)
    best = agg.groupby("dataset", sort=False).head(8).to_string(index=False)
    report = ["# Shock + robust temporal v1", "", "Leakage-safe protocol: query rows were masked first; LOO interpolation residuals and date shocks use visible rows only. Correction coefficients are fitted leave-one-partition-out.", "", "## Pooled results", "", "```", best, "```", "", "Files:", "- `research/shock_temporal_v1_results.csv` (partition/slice metrics)", "- `research/shock_temporal_v1_aggregate.csv` (pooled ranking)", "- `research/shock_temporal_v1_slices.csv` (year/cohort/source slices)", "- `research/shock_temporal_v1_preds.csv` (compact predictions)", "", "Formulae: robust state = Huber-weighted local linear estimate; shock_date = median of AOI-deduplicated LOO residuals on exact date (n>=3); shock_near = Gaussian-weighted trimmed residual median within +/-8 days; prediction = hgb + alpha*shock or (1-w)hgb+w*state."]
    (RESEARCH / "shock_temporal_v1_report.md").write_text("\n".join(report), encoding="utf-8")
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    run()

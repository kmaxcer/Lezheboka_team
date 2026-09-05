"""Compact leakage-safe screen of alternative regressors/losses.

The feature matrix is built once per protocol (v3 = 130 numeric features),
then several estimators are fit on the same pseudo-OOF blocks.  This keeps the
comparison fair and avoids touching production outputs.  Results are written
to ``research/model_variants_screen_results.csv`` and predictions to a CSV.
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
from agropulse.pipeline import build_features  # noqa: E402
sys.path.insert(0, str(R))
from feature_hgb_v2 import _clear, extra_features  # noqa: E402
from feature_hgb_v3 import extra_features_v3  # noqa: E402

TARGET = "primary_ndvi"
ID, DATE = "anon_polygon_id", "date"


def matrix(d: pd.DataFrame, observed: pd.Series, mask: np.ndarray) -> pd.DataFrame:
    mask = np.asarray(mask, bool)
    fr = _clear(d, mask)
    bx = build_features(fr, observed, pd.Series(mask, index=fr.index))
    # v2 is the default for a fast model-class screen.  Set VARIANT_V3=1 to
    # include the considerably slower cross-year/date feature expansion.
    import os
    ex = extra_features_v3(fr, observed, mask) if os.environ.get("VARIANT_V3", "0") == "1" else extra_features(fr, observed, mask)
    x = pd.concat([bx.reset_index(drop=True), ex.reset_index(drop=True)], axis=1)
    # HGB handles NaNs; tree baselines get median-imputed below.
    return x.replace([np.inf, -np.inf], np.nan)


def pseudo_masks(d: pd.DataFrame, excluded: np.ndarray, seed: int, n_masks: int = 2,
                 frac: float = .18) -> list[np.ndarray]:
    y = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
    known = np.isfinite(y) & ~np.asarray(excluded, bool)
    years = pd.to_datetime(d[DATE]).dt.year.to_numpy(int)
    tab = pd.DataFrame({"id": d[ID].astype(str).to_numpy(), "year": years})
    out = []
    for k in range(n_masks):
        rng = np.random.default_rng(seed + k)
        pm = np.zeros(len(d), bool)
        for _, ix0 in tab.loc[known].groupby(["id", "year"], sort=False).groups.items():
            ix = np.asarray(ix0, int)
            if len(ix):
                n = max(1, int(round(frac * len(ix))))
                pm[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
        out.append(pm)
    return out


def make_protocol(name: str):
    """Return reference frame and query mask/truth for one quick protocol."""
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    if name == "exact2024":
        d, truth = make_fold(tr.copy(), pr.copy(), 2024)
        # make_fold blanks target but retains the unmasked labels in _truth.
        # Do not overwrite _truth from the now-blank target column.
        q = d["is_synthetic_gap"].fillna(False).to_numpy(bool)
        d["date"] = pd.to_datetime(d.date)
        d["year"] = d.year.fillna(d.date.dt.year).astype(int)
        d["doy"] = d.doy.fillna(d.date.dt.dayofyear).astype(int)
        return d.reset_index(drop=True), q, d.loc[q, "_truth"].to_numpy(float)
    if name == "random":
        d = tr.copy().reset_index(drop=True)
        d["is_synthetic_gap"] = False
        d["_truth"] = pd.to_numeric(d[TARGET], errors="coerce")
        d["date"] = pd.to_datetime(d.date)
        d["year"] = d.year.fillna(d.date.dt.year).astype(int)
        d["doy"] = d.doy.fillna(d.date.dt.dayofyear).astype(int)
        q = np.zeros(len(d), bool)
        rng = np.random.default_rng(20260905)
        for _, ix0 in d.groupby([ID, d.date.dt.year], sort=False).groups.items():
            ix = np.asarray(ix0, int)
            ix = ix[np.isfinite(d.loc[ix, TARGET].to_numpy(float))]
            if len(ix):
                q[rng.choice(ix, size=min(len(ix), max(1, int(round(.15 * len(ix))))), replace=False)] = True
        # all dynamic fields are blanked by _clear when q/pm are applied.
        return d, q, d.loc[q, "_truth"].to_numpy(float)
    raise ValueError(name)


def estimators() -> dict:
    """Small but representative model set; all use identical X/y."""
    # 160 iterations are sufficient to rank losses/model classes and keep the
    # screen runnable while other overnight jobs use the CPU.  Final models
    # should retain the production 300--350 iteration settings.
    common = dict(max_iter=160, max_leaf_nodes=63, min_samples_leaf=30,
                  learning_rate=.03, l2_regularization=8.0, random_state=42,
                  early_stopping=False)
    return {
        "hgb_sq": HistGradientBoostingRegressor(loss="squared_error", **common),
        "hgb_abs": HistGradientBoostingRegressor(loss="absolute_error", **common),
        "hgb_q50": HistGradientBoostingRegressor(loss="quantile", quantile=.50, **common),
        # Mean of two quantiles is a simple robust/variance compromise.
        "extra_leaf5": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=5, max_features=.75,
            max_depth=None, n_jobs=-1, random_state=42)),
        "extra_leaf10": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=10, max_features=.75,
            max_depth=None, n_jobs=-1, random_state=42)),
        "rf_leaf5": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=140, min_samples_leaf=5, max_features=.65,
            max_depth=None, n_jobs=-1, random_state=42)),
    }


def run_one(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time(); d, q, yq = make_protocol(name)
    # Preserve query truth in _truth, while dynamic values are masked through
    # _clear for every feature construction.
    blocks, ys = [], []
    for pm in pseudo_masks(d, q, 1000 if name == "exact2024" else 2000, 2):
        comb = q | pm
        fr = _clear(d, comb)
        obs = fr[TARGET].where(~comb)
        print(f"{name}: build train matrix ({int(pm.sum())})", flush=True)
        x = matrix(d, obs, comb)
        blocks.append(x.loc[pm].reset_index(drop=True))
        ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    frq = _clear(d, q); obsq = frq[TARGET].where(~q)
    print(f"{name}: build query matrix ({int(q.sum())})", flush=True)
    xq = matrix(d, obsq, q).loc[q].reset_index(drop=True)
    X = pd.concat(blocks, ignore_index=True); y = pd.concat(ys, ignore_index=True).astype(float)
    print(f"{name}: X={X.shape}, q={xq.shape}, matrix_sec={time.time()-t0:.1f}", flush=True)
    rows, preds = [], []
    for mname, model in estimators().items():
        started = time.time()
        try:
            model.fit(X, y)
            p = np.clip(model.predict(xq), -.2, 1.1)
            ok = np.isfinite(p) & np.isfinite(yq)
            err = p[ok] - yq[ok]
            rmse = float(np.sqrt(np.mean(err * err)))
            mae = float(np.mean(np.abs(err)))
            rows.append({"protocol": name, "method": mname, "n": int(ok.sum()),
                         "rmse": rmse, "mae": mae, "seconds": time.time()-started,
                         "features": X.shape[1]})
            preds.append(pd.DataFrame({ID: d.loc[q, ID].to_numpy(), DATE: d.loc[q, DATE].to_numpy(),
                                       "truth": yq, "method": mname, "pred": p,
                                       "protocol": name}))
            print(name, mname, "rmse", f"{rmse:.6f}", "sec", f"{time.time()-started:.1f}", flush=True)
        except Exception as exc:
            print(name, mname, "FAILED", repr(exc), flush=True)
    print(name, "total_sec", round(time.time()-t0, 1), flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def main():
    warnings.filterwarnings("ignore")
    allr, allp = [], []
    # Environment variable lets us run one protocol first for a fast audit.
    import os
    modes = [os.environ.get("VARIANT_MODE")] if os.environ.get("VARIANT_MODE") else ["exact2024", "random"]
    for mode in modes:
        if mode not in ("exact2024", "random"): raise ValueError(mode)
        r, p = run_one(mode); allr.append(r); allp.append(p)
    rr = pd.concat(allr, ignore_index=True); pp = pd.concat(allp, ignore_index=True)
    rr.to_csv(R / "model_variants_screen_results.csv", index=False)
    pp.to_csv(R / "model_variants_screen_predictions.csv", index=False)
    print(rr.to_string(index=False), flush=True)


if __name__ == "__main__": main()

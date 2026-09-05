"""Evaluate selective canonical-MODIS blends on private-like train folds."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import (SOURCES, _prepare, _fit_source_maps, _mode_posteriors,
                   _query_posterior, _local_source_prediction, predict_private)
from validate import make_fold

DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
MOD_PERIOD = set([97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289])


def one_fold(fold: pd.DataFrame, ks=(6, 8, 10, 12, 16)):
    d = _prepare(fold.reset_index(drop=True))
    syn = d.is_synthetic_gap.to_numpy(bool)
    known = np.isfinite(d.primary_ndvi.to_numpy(float))
    y = d.primary_ndvi.to_numpy(float)
    x = d._ord.to_numpy(float)
    src = d._src.to_numpy(object)
    doy = d._doy.to_numpy(int)
    maps = _fit_source_maps(d, known, 30)
    aoi, crop, glob, date = _mode_posteriors(d, known)
    # ``predict_private`` returns only synthetic rows; place them back on the
    # full frame so candidate arrays share the same query index.
    qidx = np.flatnonzero(syn)
    base = np.full(len(d), np.nan, dtype=float)
    base[qidx] = predict_private(fold, k=8).primary_ndvi_pred.to_numpy(float)
    out = {"base": base}
    for k in ks:
        out[f"mod{k}"] = np.full(len(d), np.nan)
    out["pmod"] = np.full(len(d), np.nan)
    groups = d.groupby(["anon_polygon_id", "_year"], sort=False).groups
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        kk = ii[known[ii]]
        for q in ii[syn[ii]]:
            p = _query_posterior(d, int(q), aoi, crop, glob, date,
                                 date_weight=1.0)
            out["pmod"][q] = p[2]
            for k in ks:
                out[f"mod{k}"][q] = _local_source_prediction(
                    x[q], kk, x, y, src, "modis", maps,
                    int(doy[q]), 30, k)
    q = np.flatnonzero(syn)
    return {k: v[q] for k, v in out.items()}, fold.loc[q, "_truth"].to_numpy(float), doy[q]


def main():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"])
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"])
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    rec = []
    for year in years:
        fold, _ = make_fold(tr, pr, year)
        pred, truth, doy = one_fold(fold)
        for meth in ["base", "mod6", "mod8", "mod10", "mod12", "mod16"]:
            e = pred[meth] - truth
            rec.append((year, meth, 1.0, 0.0, len(e),
                        float(np.sqrt(np.mean(e * e)))))
        # p_modis-weighted and thresholded blends.  ``alpha`` is the maximum
        # pull toward canonical MODIS; pmod supplies confidence from date/AOI.
        for mod in ["mod6", "mod8", "mod10", "mod12", "mod16"]:
            for alpha in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
                w = alpha * pred["pmod"]
                ph = pred["base"] + w * (pred[mod] - pred["base"])
                e = ph - truth
                rec.append((year, mod, alpha, -1.0, len(e),
                            float(np.sqrt(np.mean(e * e)))))
            for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
                w = (pred["pmod"] >= th).astype(float)
                ph = np.where(w > 0, pred[mod], pred["base"])
                e = ph - truth
                rec.append((year, mod, 1.0, th, len(e),
                            float(np.sqrt(np.mean(e * e)))))
        print("fold", year, "done", flush=True)
    out = pd.DataFrame(rec, columns=["year", "method", "alpha", "threshold", "n", "rmse"])
    agg = out.groupby(["method", "alpha", "threshold"], as_index=False).apply(
        lambda z: pd.Series({"n": z.n.sum(), "rmse": np.sqrt(np.average(z.rmse ** 2, weights=z.n))}),
        include_groups=False,
    ).reset_index(drop=True).sort_values("rmse")
    print(agg.head(40).to_string(index=False))
    out.to_csv(ROOT / "research" / "_modis_blend_cv.csv", index=False)
    agg.to_csv(ROOT / "research" / "_modis_blend_agg.csv", index=False)


if __name__ == "__main__":
    main()

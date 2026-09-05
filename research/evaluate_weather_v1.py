"""Quick leakage-safe evaluation of v3 plus engineered weather features."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
sys.path.insert(0, str(ROOT / "research"))
from validate import make_fold  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402
from agropulse.pipeline import build_features  # noqa: E402
from feature_hgb_v2 import _clear  # noqa: E402
from feature_hgb_v3 import extra_features_v3  # noqa: E402
from weather_features_v1 import weather_features  # noqa: E402

TARGET = "primary_ndvi"
GAP = "is_synthetic_gap"


def matrix(d, obs, mask):
    m = np.asarray(mask, bool)
    fr = _clear(d, m)
    bx = build_features(fr, obs, pd.Series(m))
    ex = extra_features_v3(fr, obs, m)
    wx = weather_features(fr, m)
    return pd.concat([bx.reset_index(drop=True), ex.reset_index(drop=True), wx.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)


def fitpred(d, query, pool, seed):
    tab = pd.DataFrame({"id": d.anon_polygon_id.astype(str), "year": pd.to_datetime(d.date).dt.year})
    blocks, ys = [], []
    for rep in range(2):
        rng = np.random.default_rng(seed + rep)
        pm = np.zeros(len(d), bool)
        for _, ix0 in tab.loc[pool].groupby(["id", "year"], sort=False).groups.items():
            ix = np.asarray(ix0, int)
            nn = max(1, int(round(.18 * len(ix))))
            pm[rng.choice(ix, size=min(nn, len(ix)), replace=False)] = True
        comb = np.asarray(query, bool) | pm
        obs = d[TARGET].where(~comb)
        x = matrix(d, obs, comb)
        blocks.append(x.loc[pm].reset_index(drop=True))
        ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    obs = d[TARGET].where(~np.asarray(query, bool))
    qx = matrix(d, obs, query).loc[query].reset_index(drop=True)
    X = pd.concat(blocks, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    model = HistGradientBoostingRegressor(loss="squared_error", random_state=42, learning_rate=.03, max_iter=350, max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=8.)
    model.fit(X, y)
    return np.clip(model.predict(qx), -.2, 1.1), X.shape[1]


def main():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    tr["_truth"] = tr[TARGET].astype(float)
    cases = []
    f, t = make_fold(tr.copy(), pr.copy(), 2024)
    f = f.reset_index(drop=True)
    q = f[GAP].fillna(False).to_numpy(bool)
    pool = f[TARGET].notna().to_numpy(bool) & ~q & f.date.dt.year.ne(2024).to_numpy()
    cases.append(("exact2024", f, q, pool, f.loc[q, "_truth"].to_numpy(float), 20261001))
    for seed in (0, 1, 2):
        f, m = _mask_private(pr.copy(), seed)
        f = f.reset_index(drop=True)
        f["_truth"] = pd.to_numeric(f["_truth"], errors="coerce")
        q = np.asarray(m, bool)
        pool = f[TARGET].notna().to_numpy(bool) & ~q
        cases.append((f"random{seed}", f, q, pool, f.loc[q, "_truth"].to_numpy(float), 20262000 + seed))
    rows = []
    for name, d, q, pool, truth, seed in cases:
        t0 = time.time()
        pred, nf = fitpred(d, q, pool, seed)
        row = {"case": name, "n": len(truth), "features": nf, "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))), "mae": float(np.mean(np.abs(pred - truth))), "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "research" / "weather_v1_results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

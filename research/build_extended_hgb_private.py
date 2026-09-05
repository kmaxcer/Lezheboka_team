"""Fit the research extended HGB model on the full visible reference.

This is intentionally separate from the production builder.  It reuses the
leakage-safe feature construction from ``feature_hgb_v2`` and trains only on
pseudo-masked visible rows, then predicts the actual private synthetic gaps.
The resulting component CSVs are suitable for offline blending, but no
submission baseline is overwritten.
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
OUT = ROOT / "outputs"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
from agropulse.pipeline import build_features, FULL_FEATURES  # noqa: E402
sys.path.insert(0, str(RESEARCH))
from feature_hgb_v2 import _clear, extra_features  # noqa: E402

TARGET = "primary_ndvi"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _matrix(frame: pd.DataFrame, obs: pd.Series, mask: np.ndarray) -> pd.DataFrame:
    fr = _clear(frame, mask)
    bx = build_features(fr, obs, pd.Series(np.asarray(mask, bool)))
    ex = extra_features(fr, obs, np.asarray(mask, bool))
    x = pd.concat([bx.reset_index(drop=True), ex.reset_index(drop=True)], axis=1)
    return x.replace([np.inf, -np.inf], np.nan)


def _pseudo_masks(d: pd.DataFrame, hidden: np.ndarray, n_masks: int = 3,
                  fraction: float = .18) -> list[np.ndarray]:
    known = d[TARGET].notna().to_numpy(bool) & ~hidden
    years = pd.to_datetime(d.date).dt.year.to_numpy(int)
    out: list[np.ndarray] = []
    for rep in range(n_masks):
        rng = np.random.default_rng(20260905 + rep)
        m = np.zeros(len(d), dtype=bool)
        tab = pd.DataFrame({"id": d.anon_polygon_id.astype(str), "year": years})
        for _, ix0 in tab.loc[known].groupby(["id", "year"], sort=False).groups.items():
            ix = np.asarray(ix0, dtype=int)
            if len(ix):
                n = max(1, int(round(fraction * len(ix))))
                m[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
        out.append(m)
    return out


def _fit(kind: str, x: pd.DataFrame, y: pd.Series, seed: int) -> HistGradientBoostingRegressor:
    specs = {
        "regular": dict(learning_rate=.03, max_iter=350, max_leaf_nodes=48,
                         min_samples_leaf=50, l2_regularization=12.0),
        "wide": dict(learning_rate=.03, max_iter=350, max_leaf_nodes=63,
                     min_samples_leaf=30, l2_regularization=8.0),
    }
    m = HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **specs[kind])
    m.fit(x, y)
    return m


def main() -> None:
    t0 = time.time()
    train_path = DATA / "train_dataset.csv"; private_path = DATA / "private_features.csv"
    tr = pd.read_csv(train_path, parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(private_path, parse_dates=["date"], low_memory=False)
    tr["is_synthetic_gap"] = False
    pr["is_synthetic_gap"] = pr["is_synthetic_gap"].fillna(False).astype(bool)
    tr["_origin"] = "train"; pr["_origin"] = "private"
    d = pd.concat([tr, pr], ignore_index=True, sort=False)
    d["date"] = pd.to_datetime(d.date)
    d["year"] = d["year"].fillna(d.date.dt.year).astype(int)
    d["doy"] = d["doy"].fillna(d.date.dt.dayofyear).astype(int)
    d["_truth"] = pd.to_numeric(d[TARGET], errors="coerce")
    hidden = d["is_synthetic_gap"].to_numpy(bool)
    qi = np.flatnonzero(hidden)
    if len(qi) != int(pr["is_synthetic_gap"].sum()):
        raise RuntimeError("hidden-row alignment mismatch")
    blocks: list[pd.DataFrame] = []; ys: list[pd.Series] = []
    for no, pm in enumerate(_pseudo_masks(d, hidden, 3, .18), 1):
        comb = hidden | pm
        fr = _clear(d, comb)
        obs = fr[TARGET].where(~comb)
        print("features train block", no, "rows", int(pm.sum()), flush=True)
        x = _matrix(d, obs, comb)
        blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm, "_truth"].reset_index(drop=True))
    vf = _clear(d, hidden); obs = vf[TARGET].where(~hidden)
    print("features query", len(qi), flush=True)
    qx = _matrix(d, obs, hidden).loc[hidden].reset_index(drop=True)
    xall = pd.concat(blocks, ignore_index=True); yall = pd.concat(ys, ignore_index=True).astype(float)
    keys = d.loc[hidden, ["anon_polygon_id", "date"]].copy().reset_index(drop=True)
    for kind in ("regular", "wide"):
        print("fit", kind, "n", len(xall), "p", xall.shape[1], flush=True)
        m = _fit(kind, xall, yall, 42)
        p = np.clip(m.predict(qx), -.2, 1.1)
        out = keys.copy(); out["primary_ndvi_pred"] = p
        path = OUT / f"model_dani_extended_hgb_{kind}.csv"
        out.to_csv(path, index=False, float_format="%.8f")
        print(kind, "range", float(p.min()), float(p.max()), "mean", float(p.mean()), flush=True)
    meta = {
        "rows": int(len(qi)), "visible_rows": int((~hidden & d[TARGET].notna().to_numpy(bool)).sum()),
        "pseudo_masks": 3, "features": int(xall.shape[1]), "feature_names": list(xall.columns),
        "train_sha256": _sha(train_path), "private_sha256": _sha(private_path),
        "seconds": round(time.time() - t0, 1), "production_baseline_overwritten": False,
    }
    (OUT / "model_dani_extended_hgb_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

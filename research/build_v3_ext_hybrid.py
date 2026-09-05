"""Build validated full-private hybrids from lag/peer/shock, ext-v2 and v3.

No source component is refit here: this script only joins already generated
three-column predictions and enforces the organiser hidden-key contract.  It
emits a global 30% v3 blend and a conservative cohort-routed variant.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
OUT = ROOT / "outputs"
KEY = ["anon_polygon_id", "date"]
TARGET = "primary_ndvi_pred"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_component(path: Path, keys: pd.DataFrame) -> np.ndarray:
    x = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    x["date"] = pd.to_datetime(x["date"])
    if list(x.columns) != KEY + [TARGET] or x.duplicated(KEY).any():
        raise ValueError(f"invalid component schema: {path}")
    z = keys.merge(x, on=KEY, how="left", validate="one_to_one")
    p = z[TARGET].to_numpy(float)
    if len(p) != len(keys) or not np.isfinite(p).all():
        raise ValueError(f"component key/finite mismatch: {path}")
    return p


def write(keys: pd.DataFrame, p: np.ndarray, name: str, meta: dict) -> dict:
    p = np.clip(np.asarray(p, float), -.2, 1.1)
    out = keys.copy(); out[TARGET] = p
    if list(out.columns) != KEY + [TARGET] or out.duplicated(KEY).any() or not np.isfinite(p).all():
        raise ValueError("output contract failed")
    path = OUT / name
    out.to_csv(path, index=False, float_format="%.8f")
    info = {"candidate": name, "rows": int(len(out)), "min": float(p.min()), "max": float(p.max()), "mean": float(p.mean()), "sha256": sha(path), **meta}
    return info


def main() -> None:
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    private["date"] = pd.to_datetime(private["date"])
    hidden = private["is_synthetic_gap"].fillna(False).astype(bool)
    keys = private.loc[hidden, KEY].copy().reset_index(drop=True)
    train_ids = set(pd.read_csv(DATA / "train_dataset.csv", usecols=["anon_polygon_id"], low_memory=False)["anon_polygon_id"].astype(str))
    years = keys["date"].dt.year.to_numpy(int)
    shared = keys["anon_polygon_id"].astype(str).isin(train_ids).to_numpy(bool)
    history = years < 2025
    new2025 = (~shared) & (~history)
    shared2025 = shared & (~history)
    base_path = OUT / "model_dani_lag40_peer10_a350_b200_submission.csv"
    ext_path = OUT / "model_dani_lag40_peer10_extwide40_submission.csv"
    v3_path = OUT / "model_dani_extended_hgb_v3_wide.csv"
    base = read_component(base_path, keys)
    ext = read_component(ext_path, keys)
    v3 = read_component(v3_path, keys)
    # ext-v2 40% is already (0.6*base + 0.4*extended-v2).  Add v3 on top.
    global_w = np.full(len(keys), .30)
    cohort_w = np.where(history, .40, np.where(shared2025, .30, .20))
    p_global = .70 * ext + .30 * v3
    p_cohort = (1.0 - cohort_w) * ext + cohort_w * v3
    common = {
        "base_component": base_path.name,
        "base_sha256": sha(base_path),
        "ext_v2_component": ext_path.name,
        "ext_v2_sha256": sha(ext_path),
        "v3_component": v3_path.name,
        "v3_sha256": sha(v3_path),
        "private_sha256": sha(DATA / "private_features.csv"),
        "hidden_rows": int(len(keys)),
        "history_rows": int(history.sum()),
        "new_2025_rows": int(new2025.sum()),
        "shared_2025_rows": int(shared2025.sum()),
        "formula": "ext40=(0.6*base_lag40_peer10_shock + 0.4*ext_v2); final=(1-w_v3)*ext40+w_v3*v3",
        "production_baseline_overwritten": False,
    }
    infos = []
    infos.append(write(keys, p_global, "model_dani_lag40_peer10_extwide40_v3_30_submission.csv", {**common, "v3_weight": .30, "routing": "global"}))
    infos.append(write(keys, p_cohort, "model_dani_lag40_peer10_extwide40_v3_cohort_submission.csv", {**common, "v3_weights": {"history": .40, "shared_2025": .30, "new_2025": .20}, "routing": "cohort"}))
    meta = {"recommended": infos[0]["candidate"], "alternatives": infos, **common}
    (OUT / "model_dani_v3_ext_hybrid_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

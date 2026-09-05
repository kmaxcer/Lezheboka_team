"""Build cohort-aware blends for the full private hidden set.

The base component is the validated lag40 + peer10 + shock/state candidate.
The extended wide/regular HGB is routed by the observable cohort of the AOI:
new AOIs in 2010--2024, new AOIs in 2025, and train-overlap (shared) AOIs in
2025.  Existing baseline files are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
PRIVATE = DATA / "private_features.csv"
TRAIN = DATA / "train_dataset.csv"
BASE = OUT / "model_dani_lag40_peer10_a350_b200_submission.csv"
EXT = {
    "wide": OUT / "model_dani_extended_hgb_wide.csv",
    "regular": OUT / "model_dani_extended_hgb_regular.csv",
}
HOLDOUT = ROOT / "research/private_cohort_blend_holdout_predictions.csv"
KEY = ["anon_polygon_id", "date"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def component(path: Path) -> pd.DataFrame:
    z = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    req = KEY + ["primary_ndvi_pred"]
    if list(z.columns) != req:
        raise ValueError(f"bad columns: {path}: {z.columns.tolist()}")
    if z.duplicated(KEY).any() or not np.isfinite(z.primary_ndvi_pred).all():
        raise ValueError(f"bad values/duplicate keys: {path}")
    return z


def q_score(q: pd.DataFrame, w: dict[str, float]) -> dict[str, float]:
    """Score a route on the saved private holdout, where available."""
    if not HOLDOUT.exists():
        return {}
    q = pd.read_csv(HOLDOUT)
    need = {"truth", "joint40", "extended", "cohort", "year"}
    if not need.issubset(q.columns):
        return {}
    route = np.where(
        (q.cohort == "new") & (q.year < 2025), w["new_history"],
        np.where((q.cohort == "new") & (q.year >= 2025), w["new_2025"], w["shared_2025"]),
    )
    pred = (1 - route) * q.joint40.to_numpy(float) + route * q.extended.to_numpy(float)
    e = pred - q.truth.to_numpy(float)
    out = {"overall_rmse": float(np.sqrt(np.mean(e * e)))}
    out["overall_gapscore_proxy"] = float(30 * max(0.0, 1.0 - out["overall_rmse"] / 0.10))
    for label, mask in {
        "new_history": (q.cohort == "new") & (q.year < 2025),
        "new_2025": (q.cohort == "new") & (q.year >= 2025),
        "shared_2025": (q.cohort == "shared") & (q.year >= 2025),
    }.items():
        m = np.asarray(mask)
        out[label + "_rmse"] = float(np.sqrt(np.mean(e[m] * e[m])))
    return out


def main() -> None:
    private = pd.read_csv(PRIVATE, parse_dates=["date"], low_memory=False)
    train = pd.read_csv(TRAIN, usecols=["anon_polygon_id"], low_memory=False)
    shared_ids = set(train.anon_polygon_id.unique())
    hidden = private.is_synthetic_gap.fillna(False).astype(bool)
    keys = private.loc[hidden, KEY].reset_index(drop=True)
    years = keys.date.dt.year.to_numpy(int)
    cohorts = np.where(keys.anon_polygon_id.isin(shared_ids), "shared", "new")

    base = keys.merge(component(BASE), on=KEY, how="left", validate="one_to_one")
    if base.primary_ndvi_pred.isna().any():
        raise ValueError("base does not cover hidden keys")
    base_p = base.primary_ndvi_pred.to_numpy(float)

    routes = {
        # Main robust choice: moderate extension everywhere, with extra weight
        # only where the holdout has repeatedly shown historical gains.
        "balanced": {"new_history": 0.40, "new_2025": 0.30, "shared_2025": 0.30},
        # Holdout optimum rounded to stable tenths.
        "holdout_opt": {"new_history": 0.50, "new_2025": 0.40, "shared_2025": 0.30},
        # More conservative on the difficult new-2025 cohort.
        "history_focus": {"new_history": 0.50, "new_2025": 0.20, "shared_2025": 0.30},
        # Higher extension as a riskier upside candidate.
        "aggressive": {"new_history": 0.60, "new_2025": 0.40, "shared_2025": 0.40},
    }
    manifest = {
        "base": BASE.name,
        "base_sha256": sha(BASE),
        "private_sha256": sha(PRIVATE),
        "hidden_rows": int(hidden.sum()),
        "cohorts": {"shared_ids": len(shared_ids), "hidden_shared": int((cohorts == "shared").sum()),
                    "hidden_new": int((cohorts == "new").sum()),
                    "hidden_new_history": int(((cohorts == "new") & (years < 2025)).sum()),
                    "hidden_new_2025": int(((cohorts == "new") & (years >= 2025)).sum()),
                    "hidden_shared_2025": int(((cohorts == "shared") & (years >= 2025)).sum())},
        "outputs": [], "production_baseline_overwritten": False,
    }

    for kind, ext_path in EXT.items():
        ext = keys.merge(component(ext_path), on=KEY, how="left", validate="one_to_one")
        if ext.primary_ndvi_pred.isna().any():
            raise ValueError(f"extension does not cover hidden keys: {kind}")
        ext_p = ext.primary_ndvi_pred.to_numpy(float)
        for route_name, w in routes.items():
            rw = np.where((cohorts == "new") & (years < 2025), w["new_history"],
                          np.where((cohorts == "new") & (years >= 2025), w["new_2025"], w["shared_2025"]))
            pred = np.clip((1 - rw) * base_p + rw * ext_p, -.2, 1.1)
            out = keys.copy(); out["primary_ndvi_pred"] = pred
            name = f"model_dani_lag40_peer10_cohort_{route_name}_{kind}_submission.csv"
            path = OUT / name
            out.to_csv(path, index=False, float_format="%.8f")
            # Re-read so contract checks include the exact serialized artifact.
            chk = pd.read_csv(path, parse_dates=["date"])
            if list(chk.columns) != KEY + ["primary_ndvi_pred"] or chk.duplicated(KEY).any() or len(chk) != len(keys) or not np.isfinite(chk.primary_ndvi_pred).all():
                raise ValueError(f"submission contract failed: {path}")
            manifest["outputs"].append({
                "candidate": name, "kind": kind, "route": w,
                "rows": len(chk), "sha256": sha(path),
                "min": float(chk.primary_ndvi_pred.min()), "max": float(chk.primary_ndvi_pred.max()),
                "holdout_proxy": q_score(pd.DataFrame(), w),
            })
    (OUT / "model_dani_cohort_routing_metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

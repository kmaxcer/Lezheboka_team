"""Materialize the hidden-DOY-selected HGB/lag candidate in research/."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
OUT = ROOT / "research"


def flag(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def main() -> None:
    private = pd.read_csv(DATA / "private_features.csv", low_memory=False)
    h = pd.read_csv(ROOT / "outputs" / "model_dani_tuned_hgb.csv", low_memory=False)
    l = pd.read_csv(ROOT / "outputs" / "model_dani_tuned_lag.csv", low_memory=False)
    key = ["anon_polygon_id", "date"]
    hidden = private.loc[flag(private["is_synthetic_gap"]), key].reset_index(drop=True)
    z = h.merge(l, on=key, how="outer", validate="one_to_one", suffixes=("_hgb", "_lag"))
    if len(z) != len(h) or z.primary_ndvi_pred_hgb.isna().any() or z.primary_ndvi_pred_lag.isna().any():
        raise ValueError("component keys do not align")
    z["primary_ndvi_pred"] = np.clip(
        0.70*z["primary_ndvi_pred_hgb"].to_numpy(float)
        + 0.30*z["primary_ndvi_pred_lag"].to_numpy(float), -0.2, 1.1
    )
    out = hidden.merge(z[key + ["primary_ndvi_pred"]], on=key, how="left", validate="one_to_one")
    if not np.isfinite(out.primary_ndvi_pred).all() or out[key].duplicated().any():
        raise ValueError("invalid candidate")
    path = OUT / "teammate_sweep_hidden_doy_submission_w30.csv"
    out.to_csv(path, index=False, float_format="%.8f")
    d20 = pd.read_csv(ROOT / "outputs" / "model_dani_tuned_submission.csv")
    d = out.primary_ndvi_pred.to_numpy(float) - d20.primary_ndvi_pred.to_numpy(float)
    stats = pd.DataFrame([{
        "candidate": "hgb_lag_w30_hidden_doy",
        "rows": len(out),
        "mean": float(out.primary_ndvi_pred.mean()),
        "std": float(out.primary_ndvi_pred.std()),
        "min": float(out.primary_ndvi_pred.min()),
        "max": float(out.primary_ndvi_pred.max()),
        "mean_abs_delta_vs_w20": float(np.mean(np.abs(d))),
        "max_abs_delta_vs_w20": float(np.max(np.abs(d))),
    }])
    stats.to_csv(OUT / "teammate_sweep_hidden_doy_candidate_stats.csv", index=False)
    print(path)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()

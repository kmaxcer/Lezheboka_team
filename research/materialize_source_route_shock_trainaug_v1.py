"""Materialise train-augmented source-route shock candidates (new names)."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; O = ROOT / "outputs"; D = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(R)); from shock_bin_sweep_v1 import _features  # noqa: E402
ID, DATE = "anon_polygon_id", "date"; BASE = O / "model_dani_source_expert_route_v2_cohort_year_dist_submission.csv"


def sha(p):
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def run():
    tr = pd.read_csv(D / "train_dataset.csv", parse_dates=["date"], low_memory=False); pr = pd.read_csv(D / "private_features.csv", parse_dates=["date"], low_memory=False); actual = pr.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    combo = pd.concat([tr, pr], ignore_index=True, sort=False); combo["_truth"] = pd.to_numeric(combo.primary_ndvi, errors="coerce"); mc = np.r_[np.zeros(len(tr), bool), actual]; ft = _features(combo, mc, 24); shock = ft.set_index([ID, DATE])["crop_shock"]
    base = pd.read_csv(BASE, parse_dates=[DATE], low_memory=False).set_index([ID, DATE]); keys = pr.loc[actual, [ID, DATE]].copy(); keys[DATE] = pd.to_datetime(keys[DATE]); keys = keys.set_index([ID, DATE]); b = base.loc[keys.index, "primary_ndvi_pred"].to_numpy(float); s = np.asarray([shock.get(k, np.nan) for k in keys.index], float); finite = np.isfinite(s)
    tids = set(tr[ID].astype(str)); co = np.array(["shared" if str(i) in tids else "new" for i in keys.index.get_level_values(ID)], object); yr = keys.index.get_level_values(DATE).year.to_numpy(int)
    policies = {"global15": np.full(len(keys), .15), "new25_05": np.where((co == "new") & (yr == 2025), .05, .15), "new25_00": np.where((co == "new") & (yr == 2025), 0.0, .15)}; rec=[]
    for name, alpha in policies.items():
        p = np.clip(b + alpha * np.nan_to_num(s, nan=0.0), -0.2, 1.1); out = keys.reset_index(); out["primary_ndvi_pred"] = p; out[DATE] = pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d"); path = O / f"model_dani_source_expert_route_v2_cohort_year_dist_shock_trainaug_{name}_submission.csv"
        if path.exists(): raise RuntimeError(f"refusing overwrite {path.name}")
        out.to_csv(path, index=False, float_format="%.9f"); chk = pd.read_csv(path, parse_dates=[DATE]); ok = list(chk.columns) == [ID, DATE, "primary_ndvi_pred"] and len(chk) == int(actual.sum()) and not chk.duplicated([ID, DATE]).any() and np.isfinite(chk.primary_ndvi_pred).all(); m = {"candidate": path.name, "formula": f"base=source_route_cohort_year_dist; shock=visible+train 24day date_crop; policy={name}", "rows": int(len(out)), "finite": bool(ok), "shock_finite": int(finite.sum()), "shock_mean": float(np.nanmean(s)), "shock_std": float(np.nanstd(s)), "base_sha256": sha(BASE), "candidate_sha256": sha(path), "production_baseline_overwritten": False, "no_upload": True}; path.with_name(path.stem + "_metadata.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf8"); rec.append(m)
    (R / "source_route_shock_trainaug_v1_actual_metadata.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf8"); (R / "source_route_shock_trainaug_v1_actual_report.md").write_text("# Train-augmented source-route shock candidates\n\n" + json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf8"); print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__": run()

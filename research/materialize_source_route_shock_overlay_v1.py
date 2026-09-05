"""Materialise separate source-route + observable 24-day crop-shock candidates.

The base is the existing ``cohort_year_dist`` source-route submission.  The
only added feature is a visible-only, AOI-deduplicated date×crop median of
seasonal residuals (24-day profile bins).  Two conservative policies are
written under new names: global alpha=.15 and alpha=.15 except new-AOI 2025
alpha=.05.  Existing candidates are never overwritten.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; O = ROOT / "outputs"; DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(R))
from shock_bin_sweep_v1 import _features  # noqa: E402

ID, DATE = "anon_polygon_id", "date"
BASE = O / "model_dani_source_expert_route_v2_cohort_year_dist_submission.csv"


def sha(p):
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def run():
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    actual = private.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    frame = private.copy(); frame["_truth"] = pd.to_numeric(frame.primary_ndvi, errors="coerce")
    feat = _features(frame, actual, 24)
    shock = feat.set_index([ID, DATE])["crop_shock"]
    base = pd.read_csv(BASE, parse_dates=[DATE], low_memory=False).set_index([ID, DATE])
    keys = private.loc[actual, [ID, DATE]].copy(); keys[DATE] = pd.to_datetime(keys[DATE]); keys = keys.set_index([ID, DATE])
    b = base.loc[keys.index, "primary_ndvi_pred"].to_numpy(float)
    s = np.asarray([shock.get(k, np.nan) for k in keys.index], float)
    yr = keys.index.get_level_values(DATE).year.to_numpy(int)
    train_ids = set(train[ID].astype(str)); co = np.array(["shared" if str(i) in train_ids else "new" for i in keys.index.get_level_values(ID)], dtype=object)
    finite = np.isfinite(s)
    policies = {
        "global15": np.full(len(keys), .15),
        "new25_05": np.where((co == "new") & (yr == 2025), .05, .15),
        "new25_00": np.where((co == "new") & (yr == 2025), 0.0, .15),
    }
    records=[]
    for name, alpha in policies.items():
        p = np.clip(b + alpha * np.nan_to_num(s, nan=0.0), -0.2, 1.1)
        out = keys.reset_index(); out["primary_ndvi_pred"] = p; out[DATE] = pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d")
        path = O / f"model_dani_source_expert_route_v2_cohort_year_dist_shock_{name}_submission.csv"
        if path.exists(): raise RuntimeError(f"refusing to overwrite {path.name}")
        out.to_csv(path, index=False, float_format="%.9f")
        check = pd.read_csv(path, parse_dates=[DATE]); ok = list(check.columns) == [ID, DATE, "primary_ndvi_pred"] and len(check) == int(actual.sum()) and not check.duplicated([ID, DATE]).any() and np.isfinite(check.primary_ndvi_pred).all()
        meta = {"candidate": path.name, "formula": f"base=source_route_cohort_year_dist; pred=clip(base+alpha*visible_24day_date_crop_shock); policy={name}", "alpha_global": .15, "alpha_new25": float(np.unique(alpha[(co == 'new') & (yr == 2025)])[0]) if ((co == 'new') & (yr == 2025)).any() else None, "rows": int(len(out)), "finite": bool(ok), "shock_finite": int(finite.sum()), "shock_mean": float(np.nanmean(s)), "shock_std": float(np.nanstd(s)), "base_sha256": sha(BASE), "candidate_sha256": sha(path), "production_baseline_overwritten": False, "no_upload": True}
        path.with_name(path.stem + "_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(meta)
    pd.DataFrame(records).to_json(R / "source_route_shock_overlay_v1_actual_metadata.json", orient="records", force_ascii=False, indent=2)
    report = ["# Materialised source-route + 24-day crop-shock candidates", "", "Base: `outputs/model_dani_source_expert_route_v2_cohort_year_dist_submission.csv`. Shock uses only visible private targets and a 24-day seasonal profile; hidden rows contribute no values.", "", json.dumps(records, ensure_ascii=False, indent=2), "", "Candidates:"] + [f"- `outputs/{m['candidate']}`" for m in records] + ["", "No prior output was overwritten; no submission was uploaded."]
    (R / "source_route_shock_overlay_v1_actual_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__": run()

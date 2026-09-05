"""Audit and materialise local-peer correction on the trainaug-r2 route.

The source-route r2 branch is currently the strongest source-aware base.  We
reuse its four-mask row sidecar, attach the independently-built r8 crop peer
residual feature, perform leave-mask-out alpha/slice checks, and write only
new actual-gap candidates.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; O = ROOT / "outputs"
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(R))
from local_peer_residual_v1 import _peer_features  # noqa: E402
from shock_bin_sweep_v1 import _features as shock_features  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402

ID, DATE, GAP = "anon_polygon_id", "date", "is_synthetic_gap"
SEEDS = (0, 1, 2, 70404)
BASE_ROWS = R / "source_expert_route_v2_fixed_radius_trainaug_rows.csv"
PROBE_ROWS = R / "source_schedule_route_probe_rows.csv"


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def build_r2_base() -> pd.DataFrame:
    """Reconstruct exact trainaug-r2 cohort/year/distance baseline rows."""
    rows = pd.read_csv(BASE_ROWS, parse_dates=[DATE], low_memory=False)
    probe = pd.read_csv(PROBE_ROWS, parse_dates=[DATE], low_memory=False)
    q = rows.merge(probe[[ID, DATE, "seed", "sp_crop_2_n", "sp_crop_8_n"]],
                   on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    n2 = q.sp_crop_2_n.fillna(0).to_numpy(float); n8 = q.sp_crop_8_n.fillna(0).to_numpy(float)
    near = n2 > 0; mid = (~near) & (n8 > 0)
    yr = q.year.to_numpy(int); co = q.cohort.astype(str).to_numpy()
    a = np.where(near, .50, np.where(mid, .40, .30))
    a = np.where((co == "new") & (yr == 2025), .60, a)
    a = np.where((co == "shared") & (yr == 2025), .35, a)
    q["route_base"] = (1 - a) * q.baseline.to_numpy(float) + a * q.expert_trainaug_r2.to_numpy(float)
    q["alpha_route"] = a; q["near_r2"] = near; q["mid_r8"] = mid
    return q


def attach_features(q: pd.DataFrame) -> pd.DataFrame:
    """Attach saved local features and independently rebuilt shock features."""
    lf = pd.read_csv(R / "local_peer_residual_v1_features.csv", parse_dates=[DATE], low_memory=False)
    take = [ID, DATE, "seed", "r8_crop_resmean", "r4_crop_resmean", "r16_crop_resmean"]
    q = q.merge(lf[take], on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    # Rebuild crop shock per mask to guarantee the same visible-only context.
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    shocks = []
    for seed in SEEDS:
        f, m = _mask_private(pr, int(seed)); combo = pd.concat([tr, f], ignore_index=True, sort=False)
        combo["_truth"] = pd.to_numeric(combo.primary_ndvi, errors="coerce")
        sf = shock_features(combo, np.r_[np.zeros(len(tr), bool), m], 24)
        keys = f.loc[m, [ID, DATE]].copy().reset_index(drop=True); keys[DATE] = pd.to_datetime(keys[DATE])
        z = keys.join(sf.drop(columns=["idx", ID, DATE], errors="ignore")); z["seed"] = int(seed); shocks.append(z)
    sh = pd.concat(shocks, ignore_index=True)
    q = q.merge(sh[[ID, DATE, "seed", "crop_shock", "date_shock", "state"]],
                on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    return q


def audit(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-mask-out fixed-alpha and residual-joint audit with slices."""
    local = "r8_crop_resmean"; rec = []
    for held in SEEDS:
        te = q[q.seed == held]; tr = q[q.seed != held]
        for a in np.arange(0., .301, .025):
            p = np.clip(te.route_base + a * te[local].fillna(0.), -.5, 1.2)
            rec.append({"experiment": "local_fixed", "held_seed": int(held), "alpha_local": float(a), "alpha_shock": 0., "alpha_state": 0., "n": len(te), "rmse": rmse(te.truth, p), "base_rmse": rmse(te.truth, te.route_base), "coverage": float(te[local].notna().mean())})
        for cols, name in [([local], "local_loo"), ([local, "crop_shock"], "local_cropshock_loo"), ([local, "crop_shock", "state"], "local_crop_state_loo")]:
            X = tr[cols].fillna(0.).to_numpy(float); yres = tr.truth.to_numpy(float) - tr.route_base.to_numpy(float)
            c = np.linalg.solve(X.T @ X + .1 * np.eye(len(cols)), X.T @ yres); c = np.clip(c, -.8, .8)
            p = np.clip(te.route_base.to_numpy(float) + te[cols].fillna(0.).to_numpy(float) @ c, -.5, 1.2)
            rec.append({"experiment": name, "held_seed": int(held), "alpha_local": float(c[0]), "alpha_shock": float(c[cols.index("crop_shock")]) if "crop_shock" in cols else 0., "alpha_state": float(c[cols.index("state")]) if "state" in cols else 0., "n": len(te), "rmse": rmse(te.truth, p), "base_rmse": rmse(te.truth, te.route_base), "coverage": float(te[local].notna().mean())})
    results = pd.DataFrame(rec); results["delta"] = results.rmse - results.base_rmse
    results.to_csv(R / "source_expert_trainaug_r2_localpeer_v1_results.csv", index=False, float_format="%.10f")
    # Slice diagnostics for fixed local alpha=.20.
    z = q.copy(); z["pred_local020"] = np.clip(z.route_base + .20 * z[local].fillna(0.), -.5, 1.2)
    z["year_group"] = np.where(z.year == 2025, "2025", "history")
    z["distance_group"] = np.where(z.near_r2, "near_r2", np.where(z.mid_r8, "mid_r3_8", "far_or_none"))
    dims = {"seed": z.seed, "year": z.year_group, "cohort": z.cohort, "source": z.true_src, "distance": z.distance_group}
    sl = []
    for typ, gkey in dims.items():
        for key, g in z.groupby(gkey, dropna=False):
            sl.append({"slice_type": typ, "slice": key, "n": len(g), "coverage": float(g[local].notna().mean()), "rmse_base": rmse(g.truth, g.route_base), "rmse_local020": rmse(g.truth, g.pred_local020), "delta": rmse(g.truth, g.pred_local020) - rmse(g.truth, g.route_base)})
    slices = pd.DataFrame(sl); slices.to_csv(R / "source_expert_trainaug_r2_localpeer_v1_slices.csv", index=False, float_format="%.10f")
    agg = results.groupby("experiment", as_index=False).apply(lambda g: pd.Series({"n": int(g.n.sum()), "rmse_pooled": float(np.sqrt(np.average(g.rmse**2, weights=g.n))), "base_rmse_pooled": float(np.sqrt(np.average(g.base_rmse**2, weights=g.n))), "delta_pooled": float(np.sqrt(np.average(g.rmse**2, weights=g.n)) - np.sqrt(np.average(g.base_rmse**2, weights=g.n))), "wins": int((g.rmse < g.base_rmse).sum())}), include_groups=False).reset_index(drop=True).sort_values("rmse_pooled")
    agg.to_csv(R / "source_expert_trainaug_r2_localpeer_v1_aggregate.csv", index=False, float_format="%.10f")
    return agg, slices


def materialize() -> list[dict]:
    """Build local-peer candidate(s) on the actual 3,112 hidden gaps."""
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    actual = pr[GAP].fillna(False).astype(bool).to_numpy()
    tr0 = tr.copy(); tr0["_truth"] = pd.to_numeric(tr0.primary_ndvi, errors="coerce"); tr0["_hidden"] = False
    f = pr.copy(); f["_truth"] = pd.to_numeric(f.primary_ndvi, errors="coerce"); f["_hidden"] = actual
    combo = pd.concat([tr0, f], ignore_index=True, sort=False)
    known = combo.primary_ndvi.notna().to_numpy(bool) & ~combo._hidden.to_numpy(bool)
    qidx = np.flatnonzero(np.r_[np.zeros(len(tr), bool), actual])
    lf = _peer_features(combo, known, qidx, width=24)
    keys = f.loc[actual, [ID, DATE]].copy().reset_index(drop=True); keys[DATE] = pd.to_datetime(keys[DATE])
    lf = keys.join(lf.drop(columns=[ID, DATE], errors="ignore"))
    # Visible-only shock (for a separate diagnostic combination).
    sf = shock_features(combo, np.r_[np.zeros(len(tr), bool), actual], 24)
    sf = keys.join(sf.drop(columns=["idx", ID, DATE], errors="ignore"))
    feat = lf.merge(sf[[ID, DATE, "crop_shock", "state"]], on=[ID, DATE], how="left", validate="one_to_one")
    basepath = O / "model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_submission.csv"
    base = pd.read_csv(basepath, parse_dates=[DATE], low_memory=False)
    q = keys.merge(base, on=[ID, DATE], how="left", validate="one_to_one").merge(feat[[ID, DATE, "r8_crop_resmean", "crop_shock", "state"]], on=[ID, DATE], how="left", validate="one_to_one")
    if q.primary_ndvi_pred.isna().any(): raise RuntimeError("base alignment failed")
    b = q.primary_ndvi_pred.to_numpy(float); l = q.r8_crop_resmean.fillna(0.).to_numpy(float); s = q.crop_shock.fillna(0.).to_numpy(float); st = q.state.fillna(0.).to_numpy(float)
    configs = [
        ("model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_r8_a020_submission.csv", b + .20 * l, "base=trainaug_r2_cyd_v1; pred=clip(base+0.20*r8 crop same-date/same-crop ID-radius8 inverse-distance residual mean)"),
        ("model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_r8_a019_submission.csv", b + .19 * l, "base=trainaug_r2_cyd_v1; pred=clip(base+0.19*r8 crop same-date/same-crop ID-radius8 inverse-distance residual mean)"),
        ("model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_joint_diag_submission.csv", b + .23 * l - .067 * s - .05 * st, "base=trainaug_r2_cyd_v1; diagnostic pred=clip(base+0.23*localpeer-0.067*crop_shock-0.05*temporal_state)"),
    ]
    metas = []
    for name, pred, formula in configs:
        path = O / name
        if path.exists(): raise RuntimeError(f"refusing overwrite {path}")
        out = keys.copy(); out["primary_ndvi_pred"] = np.clip(pred, -.2, 1.1); out[DATE] = pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d"); out = out[[ID, DATE, "primary_ndvi_pred"]]; out.to_csv(path, index=False, float_format="%.9f")
        chk = pd.read_csv(path); ok = len(chk) == int(actual.sum()) and list(chk.columns) == [ID, DATE, "primary_ndvi_pred"] and chk[[ID, DATE]].drop_duplicates().shape[0] == len(chk) and np.isfinite(chk.primary_ndvi_pred).all()
        meta = {"candidate": path.name, "formula": formula, "rows": int(len(out)), "finite": bool(ok), "unique_keys": int(chk[[ID, DATE]].drop_duplicates().shape[0]), "local_feature_finite": int(np.isfinite(l).sum()), "local_feature_coverage": float(np.isfinite(l).mean()), "state_feature_finite": int(np.isfinite(st).sum()), "base_sha256": sha(basepath), "candidate_sha256": sha(path), "production_baseline_overwritten": False, "no_upload": True}
        path.with_name(path.stem + "_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf8"); metas.append(meta)
    return metas


def main():
    t0 = time.time(); q = build_r2_base(); q = attach_features(q); agg, slices = audit(q); metas = materialize()
    report = ["# Trainaug-r2 local peer residual audit", "", "The base is the fixed-radius-2 source expert route fit on train + visible private rows. Local features use only visible rows and a 24-day AOI seasonal profile; LOO coefficients are fitted across four masks.", "", "## Aggregate", "", agg.to_string(index=False), "", "## Slices (fixed local alpha=.20)", "", slices.to_string(index=False), "", "## Materialized actual-gap candidates", "", json.dumps(metas, indent=2), "", f"Elapsed seconds: {time.time()-t0:.1f}", "Existing candidates were not overwritten; no upload performed."]
    (R / "source_expert_trainaug_r2_localpeer_v1_report.md").write_text("\n".join(report)+"\n", encoding="utf8")
    print(agg.to_string(index=False)); print(json.dumps(metas, indent=2))


if __name__ == "__main__": main()

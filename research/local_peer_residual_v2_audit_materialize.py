"""Audit and materialise a local peer-residual correction.

The v1 screen found a stable gain from an inverse-distance weighted mean of
same-date/same-crop peer residuals.  This script adds slice/alpha audits,
checks whether the seasonal date shock is redundant, and writes separate
actual-gap candidates.  It never overwrites an existing output.
"""
from __future__ import annotations

from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"; O = ROOT / "outputs"
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(R))
from teammate_sweep_postcorr import _mask_private  # noqa: E402
from shock_bin_sweep_v1 import _features as shock_features  # noqa: E402
from local_peer_residual_v1 import _peer_features  # noqa: E402

ID, DATE = "anon_polygon_id", "date"
SEEDS = (0, 1, 2, 70404)


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _merge_true_source(z: pd.DataFrame) -> pd.DataFrame:
    """Attach scoring-only source labels to the saved feature sidecar."""
    from source_expert_route_v2_seed2_policy_audit import load
    rr = load()[[ID, DATE, "seed", "true_src"]].copy(); rr[DATE] = pd.to_datetime(rr[DATE])
    return z.merge(rr, on=[ID, DATE, "seed"], how="left", validate="one_to_one")


def _build_shock_sidecar() -> pd.DataFrame:
    """Rebuild visible-only 24-day crop shocks for each audit mask."""
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    out = []
    for seed in SEEDS:
        f, m = _mask_private(pr, int(seed))
        combo = pd.concat([tr, f], ignore_index=True, sort=False)
        combo["_truth"] = pd.to_numeric(combo.primary_ndvi, errors="coerce")
        mc = np.r_[np.zeros(len(tr), bool), m]
        s = shock_features(combo, mc, 24)
        keys = f.loc[m, [ID, DATE]].copy().reset_index(drop=True); keys[DATE] = pd.to_datetime(keys[DATE])
        q = keys.join(s.drop(columns=["idx", ID, DATE], errors="ignore"))
        q["seed"] = int(seed); out.append(q)
    return pd.concat(out, ignore_index=True)


def _fit_ridge(train: pd.DataFrame, cols: list[str], ridge: float = .1) -> np.ndarray:
    X = train[cols].fillna(0.).to_numpy(float)
    r = train.truth.to_numpy(float) - train.route_base.to_numpy(float)
    c = np.linalg.solve(X.T @ X + ridge * np.eye(len(cols)), X.T @ r)
    return np.clip(c, -.8, .8)


def audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run four-mask LOO alpha/slice audit and shock overlay test."""
    z = pd.read_csv(R / "local_peer_residual_v1_features.csv", parse_dates=[DATE], low_memory=False)
    z = _merge_true_source(z)
    sh = _build_shock_sidecar()
    z = z.merge(sh[[ID, DATE, "seed", "crop_shock", "date_shock", "state"]],
                on=[ID, DATE, "seed"], how="left", validate="one_to_one")
    z["year_calc"] = z[DATE].dt.year.astype(int)
    z["distance_bin"] = np.select([z.near_dist <= 2, z.near_dist <= 8],
                                   ["near_le2", "mid_3_8"], default="far_or_none")
    local = "r8_crop_resmean"
    # LOO global alpha and fixed alpha grid.
    records = []
    for held in SEEDS:
        te = z[z.seed == held].copy(); tr = z[z.seed != held].copy()
        for a in np.arange(0., .301, .025):
            p = np.clip(te.route_base.to_numpy() + a * te[local].fillna(0.).to_numpy(), -.5, 1.2)
            records.append({"experiment": "local_fixed", "held_seed": int(held),
                            "alpha_local": float(a), "alpha_shock": 0., "alpha_date": 0.,
                            "n": len(te), "coverage": float(te[local].notna().mean()),
                            "rmse": rmse(te.truth, p), "base_rmse": rmse(te.truth, te.route_base)})
        # OLS local only and joint local + shock/date/state, coefficients fit
        # on the other masks.  This is a leakage-safe residual correction.
        for cols, name in [([local], "local_loo"),
                           ([local, "crop_shock"], "local_cropshock_loo"),
                           ([local, "date_shock"], "local_dateshock_loo"),
                           ([local, "crop_shock", "date_shock"], "local_jointshock_loo"),
                           ([local, "crop_shock", "state"], "local_crop_state_loo")]:
            c = _fit_ridge(tr, cols, ridge=.1)
            X = te[cols].fillna(0.).to_numpy(float); p = np.clip(te.route_base.to_numpy() + X @ c, -.5, 1.2)
            records.append({"experiment": name, "held_seed": int(held),
                            "alpha_local": float(c[0]),
                            "alpha_shock": float(c[cols.index("crop_shock")]) if "crop_shock" in cols else 0.,
                            "alpha_date": float(c[cols.index("date_shock")]) if "date_shock" in cols else 0.,
                            "alpha_state": float(c[cols.index("state")]) if "state" in cols else 0.,
                            "n": len(te), "coverage": float(te[local].notna().mean()),
                            "rmse": rmse(te.truth, p), "base_rmse": rmse(te.truth, te.route_base)})
    results = pd.DataFrame(records)
    results["delta"] = results.rmse - results.base_rmse
    # Slice table for the selected fixed alpha=.20 and the best simple joint
    # candidate (reported only; actual materialisation uses local-only).
    z["pred_local020"] = np.clip(z.route_base + .20 * z[local].fillna(0.), -.5, 1.2)
    slices = []
    dimensions = {"all": np.ones(len(z), bool), "seed": z.seed, "year": z.year_calc,
                  "cohort": z.cohort, "source": z.true_src, "distance": z.distance_bin}
    for dim, groups in dimensions.items():
        if dim == "all": groups = pd.Series(["all"] * len(z), index=z.index)
        else: groups = pd.Series(groups, index=z.index)
        for key, g in z.groupby(groups, dropna=False):
            slices.append({"slice_type": dim, "slice": key, "n": len(g),
                           "coverage": float(g[local].notna().mean()),
                           "rmse_base": rmse(g.truth, g.route_base),
                           "rmse_local020": rmse(g.truth, g.pred_local020),
                           "delta": rmse(g.truth, g.pred_local020) - rmse(g.truth, g.route_base)})
    slices = pd.DataFrame(slices)
    # Aggregate fixed grid and LOO records weighted by row count.
    aggregate = results.groupby("experiment", as_index=False).apply(
        lambda g: pd.Series({"n": int(g.n.sum()),
                             "rmse_pooled": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))),
                             "base_rmse_pooled": float(np.sqrt(np.average(g.base_rmse ** 2, weights=g.n))),
                             "delta_pooled": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n)) - np.sqrt(np.average(g.base_rmse ** 2, weights=g.n))),
                             "wins": int((g.rmse < g.base_rmse).sum())}), include_groups=False).reset_index(drop=True)
    # Keep alpha columns from grouping only if present; aggregate is sorted by score.
    aggregate = aggregate.sort_values("rmse_pooled")
    results.to_csv(R / "local_peer_residual_v2_results.csv", index=False, float_format="%.10f")
    slices.to_csv(R / "local_peer_residual_v2_slices.csv", index=False, float_format="%.10f")
    aggregate.to_csv(R / "local_peer_residual_v2_aggregate.csv", index=False, float_format="%.10f")
    return z, results, slices


def materialize(z_audit: pd.DataFrame | None = None) -> dict:
    """Write local-only and diagnostic local+shock actual-gap candidates."""
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    pr["_truth"] = pd.to_numeric(pr.primary_ndvi, errors="coerce")
    actual = pr.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    # Match the leakage-safe context used in the audit: train rows + private
    # visible rows; all actual gaps remain hidden from feature construction.
    tr0 = tr.copy(); tr0["_truth"] = pd.to_numeric(tr0.primary_ndvi, errors="coerce"); tr0["_hidden"] = False
    f = pr.copy(); f["_hidden"] = actual
    combo = pd.concat([tr0, f], ignore_index=True, sort=False)
    known = combo.primary_ndvi.notna().to_numpy(bool) & ~combo._hidden.to_numpy(bool)
    qidx = np.flatnonzero(np.r_[np.zeros(len(tr), bool), actual])
    lf = _peer_features(combo, known, qidx, width=24)
    keys = f.loc[actual, [ID, DATE]].copy().reset_index(drop=True); keys[DATE] = pd.to_datetime(keys[DATE])
    lf = keys.join(lf.drop(columns=[ID, DATE], errors="ignore"))
    # Rebuild visible-only shock independently (same 24-day profile).
    sf = shock_features(combo, np.r_[np.zeros(len(tr), bool), actual], 24)
    sf = keys.join(sf.drop(columns=["idx", ID, DATE], errors="ignore"))
    feat = lf.merge(sf[[ID, DATE, "crop_shock"]], on=[ID, DATE], how="left", validate="one_to_one")
    # Existing source-route candidate is itself fit with train + visible
    # private context.  Read it as the base; never overwrite it.
    basepath = O / "model_dani_source_expert_route_v2_cohort_year_dist_submission.csv"
    base = pd.read_csv(basepath, parse_dates=[DATE], low_memory=False)
    q = keys.merge(base, on=[ID, DATE], how="left", validate="one_to_one")
    q = q.merge(feat[[ID, DATE, "r8_crop_resmean", "crop_shock"]], on=[ID, DATE], how="left", validate="one_to_one")
    if q.primary_ndvi_pred.isna().any(): raise RuntimeError("base alignment failed")
    b = q.primary_ndvi_pred.to_numpy(float); l = q.r8_crop_resmean.fillna(0.).to_numpy(float); s = q.crop_shock.fillna(0.).to_numpy(float)
    pred_local = np.clip(b + .20 * l, -.2, 1.1)
    # Positive shock is intentionally retained only as a diagnostic candidate;
    # four-mask audit shows it is redundant after local peer correction.
    pred_joint = np.clip(b + .20 * l + .175 * s, -.2, 1.1)
    written = []
    for stem, pred, formula in [
        ("model_dani_source_expert_route_v2_cohort_year_dist_localpeer_r8_a020_submission.csv", pred_local,
         "base=source_route_v2_cohort_year_dist (train+visible-private fit); pred=clip(base+0.20*visible_train_augmented_24day_same_date_same_crop_ID_radius8_inverse_distance_residual_mean)"),
        ("model_dani_source_expert_route_v2_cohort_year_dist_localpeer_r8_a020_shock175_diag_submission.csv", pred_joint,
         "base=source_route_v2_cohort_year_dist; pred=clip(base+0.20*local_peer_residual+0.175*visible_train_augmented_24day_crop_shock) [diagnostic; shock redundant]"),
    ]:
        path = O / stem
        if path.exists(): raise RuntimeError(f"refusing to overwrite {path}")
        out = keys.copy(); out["primary_ndvi_pred"] = pred; out[DATE] = pd.to_datetime(out[DATE]).dt.strftime("%Y-%m-%d")
        out = out[[ID, DATE, "primary_ndvi_pred"]]
        out.to_csv(path, index=False, float_format="%.9f")
        chk = pd.read_csv(path)
        ok = len(chk) == int(actual.sum()) and chk[[ID, DATE]].drop_duplicates().shape[0] == len(chk) and np.isfinite(chk.primary_ndvi_pred).all()
        meta = {"candidate": path.name, "formula": formula, "rows": int(len(out)), "finite": bool(ok),
                "unique_keys": int(chk[[ID, DATE]].drop_duplicates().shape[0]),
                "local_feature_finite": int(np.isfinite(l).sum()), "local_feature_coverage": float(np.isfinite(l).mean()),
                "shock_feature_finite": int(np.isfinite(s).sum()), "base_sha256": sha(basepath),
                "candidate_sha256": sha(path), "production_baseline_overwritten": False, "no_upload": True}
        path.with_name(path.stem + "_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
        written.append(meta)
    return {"written": written, "base": str(basepath), "local_nonzero": int(np.count_nonzero(l)), "shock_finite": int(np.isfinite(s).sum())}


def run():
    t0 = time.time(); z, results, slices = audit(); mats = materialize(z)
    agg = pd.read_csv(R / "local_peer_residual_v2_aggregate.csv")
    report = ["# Local peer residual v2 audit and actual-gap candidates", "",
              "All peer/profile features use only visible train + unmasked private rows. Coefficients are leave-mask-out across seeds 0, 1, 2, 70404.", "",
              "## Aggregate experiments", "", agg.head(40).to_string(index=False), "",
              "## Selected fixed alpha=.20 slices", "", slices.to_string(index=False), "",
              "## Materialized candidates", "", json.dumps(mats, indent=2), "",
              f"Elapsed seconds: {time.time()-t0:.1f}", "No existing candidate was overwritten; no submission/upload performed."]
    (R / "local_peer_residual_v2_report.md").write_text("\n".join(report) + "\n", encoding="utf8")
    print(agg.head(30).to_string(index=False), flush=True); print(json.dumps(mats, indent=2), flush=True)


if __name__ == "__main__":
    run()

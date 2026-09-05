"""Overlay the 24-day date×crop shock on the four-mask source route.

The source-route policy is loaded from the existing four-mask audit.  The
shock is rebuilt independently from each mask's visible targets and is never
computed from source labels or hidden rows.  All alpha/grid decisions are
leave-one-mask-out; this script only writes new research artifacts.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"; DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(R))
from source_expert_route_v2_seed2_policy_audit import load as load_route, policy_pred  # noqa: E402
from shock_bin_sweep_v1 import _mask_private, _features  # noqa: E402

KEY = ["anon_polygon_id", "date"]
ROUTE = "crop_hier_n1_p67"
POLICY = "cohort_year_dist"


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def prepare():
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    route = load_route().copy(); route["date"] = pd.to_datetime(route["date"])
    parts = []
    for seed in (0, 1, 2, 70404):
        frame, mask = _mask_private(private, seed)
        f = _features(frame, mask, 24)
        q = route[route.seed.astype(int).eq(seed)].copy()
        q = q.merge(f[KEY + ["crop_shock", "date_shock", "state", "crop_n", "date_n"]], on=KEY, how="left", validate="one_to_one")
        q["route_pred"] = policy_pred(q, ROUTE, POLICY)
        q["dataset"] = "source_route_four_mask"; q["seed"] = seed
        parts.append(q)
    return parts


def fit_alpha(cal, feature):
    x = cal[feature].to_numpy(float); r = cal.truth.to_numpy(float) - cal.route_pred.to_numpy(float); ok = np.isfinite(x) & np.isfinite(r)
    den = np.sum(x[ok] ** 2)
    return float(np.clip(np.sum(x[ok] * r[ok]) / max(den, 1e-9), -0.8, 0.8)) if ok.sum() >= 30 else 0.0


def run():
    parts = prepare(); rows = []; predrows = []
    for i, test in enumerate(parts):
        cal = pd.concat([p for j, p in enumerate(parts) if j != i], ignore_index=True)
        y = test.truth.to_numpy(float); b = test.route_pred.to_numpy(float)
        # Baseline and conservative fixed/grid alphas for each observable
        # shock.  Fitted alpha is held out by mask; fixed grid is reported for
        # deployment sensitivity only.
        for feat in ["crop_shock", "date_shock", "state"]:
            a = fit_alpha(cal, feat); x = test[feat].to_numpy(float)
            for label, aa in [("loo", a), ("fixed010", .10), ("fixed015", .15), ("fixed020", .20), ("fixed025", .25)]:
                p = b.copy(); ok = np.isfinite(x); p[ok] += aa * x[ok]; p = np.clip(p, -0.5, 1.2)
                rows.append({"seed": int(test.seed.iloc[0]), "feature": feat, "variant": label, "alpha": float(aa), "n": len(test), "finite": int(ok.sum()), "rmse": rmse(y, p), "baseline_rmse": rmse(y, b), "delta": rmse(y, p) - rmse(y, b)})
                if feat == "crop_shock" and label in {"loo", "fixed020"}:
                    z = test[KEY + ["truth", "route_pred", "year", "cohort", "near_dist"]].copy(); z["seed"] = int(test.seed.iloc[0]); z["feature"] = feat; z["variant"] = label; z["pred"] = p; z["shock"] = x; predrows.append(z)
        # Joint crop shock + local state, fitted on other masks with a ridge.
        X = cal[["crop_shock", "state"]].fillna(0.0).to_numpy(float); rr = cal.truth.to_numpy(float) - cal.route_pred.to_numpy(float); ok = np.isfinite(rr)
        coef = np.linalg.solve(X[ok].T @ X[ok] + .15 * np.eye(2), X[ok].T @ rr[ok]); coef = np.clip(coef, -0.8, 0.8)
        Xt = test[["crop_shock", "state"]].fillna(0.0).to_numpy(float); p = np.clip(b + Xt @ coef, -0.5, 1.2)
        rows.append({"seed": int(test.seed.iloc[0]), "feature": "joint_crop_state", "variant": "loo", "alpha": float(np.linalg.norm(coef)), "coef_crop": float(coef[0]), "coef_state": float(coef[1]), "n": len(test), "finite": int(np.isfinite(test.crop_shock).sum()), "rmse": rmse(y, p), "baseline_rmse": rmse(y, b), "delta": rmse(y, p) - rmse(y, b)})
        # Slice diagnostics for the promising crop-shock LOO variant.
        aa = fit_alpha(cal, "crop_shock"); x = test.crop_shock.to_numpy(float); pp = b.copy(); good = np.isfinite(x); pp[good] += aa * x[good]
        for scope, sel in {"all": np.ones(len(test), bool), "history": test.year.to_numpy(int) < 2025, "2025": test.year.to_numpy(int) == 2025, "new2025": (test.year.to_numpy(int) == 2025) & test.cohort.eq("new").to_numpy(), "shared2025": (test.year.to_numpy(int) == 2025) & test.cohort.eq("shared").to_numpy()}.items():
            if sel.any(): rows.append({"seed": int(test.seed.iloc[0]), "feature": "crop_shock", "variant": "loo_slice_" + scope, "alpha": float(aa), "n": int(sel.sum()), "finite": int((sel & good).sum()), "rmse": rmse(y[sel], pp[sel]), "baseline_rmse": rmse(y[sel], b[sel]), "delta": rmse(y[sel], pp[sel]) - rmse(y[sel], b[sel])})
    res = pd.DataFrame(rows); res.to_csv(R / "source_route_shock_overlay_v1_results.csv", index=False)
    if predrows: pd.concat(predrows, ignore_index=True).to_csv(R / "source_route_shock_overlay_v1_preds.csv", index=False)
    main = res[res.variant.isin(["loo", "fixed010", "fixed015", "fixed020", "fixed025"]) & res.feature.isin(["crop_shock", "date_shock", "state", "joint_crop_state"])].copy()
    agg = main.groupby(["feature", "variant"], as_index=False).apply(lambda g: pd.Series({"n": int(g.n.sum()), "rmse_pooled": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))), "baseline_rmse_pooled": float(np.sqrt(np.average(g.baseline_rmse ** 2, weights=g.n))), "delta": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n)) - np.sqrt(np.average(g.baseline_rmse ** 2, weights=g.n))), "wins": int((g.rmse < g.baseline_rmse).sum()), "masks": int(len(g))}), include_groups=False).reset_index(drop=True)
    agg.to_csv(R / "source_route_shock_overlay_v1_aggregate.csv", index=False)
    sl = res[res.variant.str.startswith("loo_slice")].sort_values(["feature", "variant", "seed"]); sl.to_csv(R / "source_route_shock_overlay_v1_slices.csv", index=False)
    report = ["# Source-route + 24-day date×crop shock overlay", "", "Four independent masks (0,1,2,70404); source route policy is `cohort_year_dist`; shock uses visible rows only and alpha is fitted leave-one-mask-out.", "", "## Pooled overlay grid", "", agg.sort_values("rmse_pooled").to_string(index=False), "", "## Slice diagnostics", "", sl.to_string(index=False), "", "Artifacts: `research/source_route_shock_overlay_v1_results.csv`, `research/source_route_shock_overlay_v1_aggregate.csv`, `research/source_route_shock_overlay_v1_slices.csv`, `research/source_route_shock_overlay_v1_preds.csv`", "", "No candidate or production output was overwritten."]
    (R / "source_route_shock_overlay_v1_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(agg.sort_values("rmse_pooled").to_string(index=False)); print("\nSlices:\n", sl.to_string(index=False))


if __name__ == "__main__": run()

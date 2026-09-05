"""Source/cohort slices for the 24-day seasonal crop shock."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(R))
from shock_bin_sweep_v1 import _mask_private, _features  # noqa: E402


def metric(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def run():
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    train_ids = set(train.anon_polygon_id.astype(str))
    orig = private.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    src_all = np.select([orig.s2_ndvi.notna(), orig.landsat_ndvi.notna(), orig.modis_ndvi.notna()], ["S2", "L8", "MOD"], "NONE")
    parts = []
    baseall = pd.read_csv(R / "teammate_sweep_postcorr_preds.csv", parse_dates=["date"], low_memory=False)
    baseall = baseall[baseall.method.eq("blend_lag_0.20")]
    for seed in (0, 1, 2):
        frame, mask = _mask_private(private, seed)
        q = frame.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy().reset_index(drop=True)
        q = q.merge(baseall[baseall.partition.eq(f"random{seed}")][["anon_polygon_id", "date", "pred"]].rename(columns={"pred": "baseline"}), on=["anon_polygon_id", "date"], validate="one_to_one")
        q = q.merge(_features(frame, mask, 24).drop(columns=["idx"], errors="ignore"), on=["anon_polygon_id", "date"], validate="one_to_one")
        q["src"] = src_all[mask]
        q["year"] = q.date.dt.year.astype(int)
        q["cohort"] = np.where(q.anon_polygon_id.astype(str).isin(train_ids), "shared", "new")
        q["seed"] = seed
        parts.append(q)
    rows = []
    for i, test in enumerate(parts):
        trainq = pd.concat([p for j, p in enumerate(parts) if j != i], ignore_index=True)
        for scope, sm in {
            "all": np.ones(len(test), bool),
            "2025": test.year.to_numpy() == 2025,
            "shared25": (test.year.to_numpy() == 2025) & test.cohort.eq("shared").to_numpy(),
            "new25": (test.year.to_numpy() == 2025) & test.cohort.eq("new").to_numpy(),
            "history": test.year.to_numpy() < 2025,
        }.items():
            for source in ["ALL", "S2", "L8", "MOD"]:
                sel = sm.copy()
                if source != "ALL": sel &= test.src.eq(source).to_numpy()
                x = trainq.crop_shock.to_numpy(float); r = trainq._truth.to_numpy(float) - trainq.baseline.to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(r)
                alpha = float(np.clip(np.sum(x[ok] * r[ok]) / max(np.sum(x[ok] ** 2), 1e-9), -0.8, 0.8))
                xx = test.crop_shock.to_numpy(float); good = sel & np.isfinite(xx)
                y = test._truth.to_numpy(float); b = test.baseline.to_numpy(float); p = b.copy(); p[good] += alpha * xx[good]
                if sel.any():
                    rows.append({"seed": int(test.seed.iloc[0]), "scope": scope, "source": source, "n": int(sel.sum()), "finite": int(good.sum()), "alpha": alpha, "baseline_rmse": metric(y[sel], b[sel]), "corrected_rmse": metric(y[sel], p[sel])})
    out = pd.DataFrame(rows)
    out.to_csv(R / "shock_bin_source_v1_results.csv", index=False)
    pooled = []
    for (scope, source), g in out.groupby(["scope", "source"], sort=False):
        pooled.append({"scope": scope, "source": source, "n": int(g.n.sum()), "baseline_rmse": float(np.sqrt(np.average(g.baseline_rmse ** 2, weights=g.n))), "corrected_rmse": float(np.sqrt(np.average(g.corrected_rmse ** 2, weights=g.n))), "mean_alpha": float(g.alpha.mean())})
    po = pd.DataFrame(pooled).sort_values(["scope", "corrected_rmse"])
    po.to_csv(R / "shock_bin_source_v1_aggregate.csv", index=False)
    (R / "shock_bin_source_v1_report.md").write_text("# 24-day crop-shock source slices\n\n" + po.to_string(index=False) + "\n\nSource labels are evaluation-only diagnostics; correction itself uses only visible date×crop residuals.\n", encoding="utf-8")
    print(po.to_string(index=False))


if __name__ == "__main__":
    run()

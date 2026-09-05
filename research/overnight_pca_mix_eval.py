"""Check whether the research PCA local prediction can improve Dani blend."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"


def main() -> None:
    pca = pd.read_csv(R / "overnight_correction_predictions.csv", low_memory=False)
    pca["date"] = pd.to_datetime(pca["date"])
    pca = pca[pca["method"].isin(["safe_pca_rank2_b0.5", "safe_pca_rank1_b0.5"])].copy()
    base = pd.read_csv(R / "teammate_sweep_postcorr_preds.csv", low_memory=False)
    base["date"] = pd.to_datetime(base["date"])
    base = base[base["method"].eq("blend_lag_0.20")][["dataset", "partition", "anon_polygon_id", "date", "_truth", "pred"]].rename(columns={"pred": "prod"})
    pca = pca.merge(base, on=["partition", "anon_polygon_id", "date"], how="inner", suffixes=("", "_base"))
    rows = []
    for method, g in pca.groupby("method", sort=False):
        y = g["_truth"].to_numpy(float); x = g["pred"].to_numpy(float); b = g["prod"].to_numpy(float)
        for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
            z = (1.0 - w) * b + w * x; e = z - y
            rows.append({"method": method, "weight_pca": w, "n": len(e), "rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e)))})
        for dataset, q in g.groupby("dataset", sort=False):
            yy = q["_truth"].to_numpy(float); xx = q["pred"].to_numpy(float); bb = q["prod"].to_numpy(float)
            for w in [0.0, 0.10, 0.20, 0.30, 0.50, 1.0]:
                ee = (1.0 - w) * bb + w * xx - yy
                rows.append({"method": method, "weight_pca": w, "dataset": dataset, "n": len(ee), "rmse": float(np.sqrt(np.mean(ee * ee))), "mae": float(np.mean(np.abs(ee)))})
    out = pd.DataFrame(rows)
    out.to_csv(R / "overnight_pca_mix_metrics.csv", index=False)
    # Human-readable summary with the best blend weight per protocol.
    lines = ["# PCA mix against production", "", "PCA predictions are observed-only-calibrated diagnostics. Production reference is HGB+lag 80/20.", ""]
    for key, g in out.groupby(["method", "dataset"], dropna=False):
        lines.append(str(key) + "\n" + g.sort_values("rmse").head(8).to_string(index=False) + "\n")
    (R / "overnight_pca_mix_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.sort_values("rmse").head(30).to_string(index=False))


if __name__ == "__main__":
    main()

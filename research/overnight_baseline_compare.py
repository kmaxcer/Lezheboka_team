"""Compare Dani's HGB/lag production components on the same proxy rows.

This consumes the already cross-fitted row-level table produced by
``teammate_sweep_postcorr.py`` and never writes under ``outputs/``.  The lag
component is recovered algebraically from the stored 80/20 blend so that the
comparison is exactly on the same masks as the source-aware evaluator.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"


def _src(d: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select(
        [d["s2_ndvi"].notna(), d["landsat_ndvi"].notna(), d["modis_ndvi"].notna()],
        ["s2", "landsat", "modis"], default="none",
    ), index=d.index)


def main() -> None:
    p = pd.read_csv(R / "teammate_sweep_postcorr_preds.csv", low_memory=False)
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["method"].isin(["hgb_raw", "blend_lag_0.20"])].copy()
    wide = p.pivot_table(index=["dataset", "partition", "anon_polygon_id", "date", "year", "doy", "_truth"], columns="method", values="pred", aggfunc="first").reset_index()
    wide["lag_component"] = (wide["blend_lag_0.20"] - 0.8 * wide["hgb_raw"]) / 0.2
    wide["blend20"] = wide["blend_lag_0.20"]
    # Source labels are restored from the unmasked source files solely for
    # evaluation slicing; they are not features used by either model.
    tr = pd.read_csv(DATA / "train_dataset.csv", low_memory=False, parse_dates=["date"])
    pr = pd.read_csv(DATA / "private_features.csv", low_memory=False, parse_dates=["date"])
    tr["_src"] = _src(tr); pr["_src"] = _src(pr)
    trm = tr[["anon_polygon_id", "date", "_src"]].rename(columns={"_src": "true_src"})
    prm = pr[["anon_polygon_id", "date", "_src"]].rename(columns={"_src": "true_src"})
    source_map = pd.concat([trm, prm], ignore_index=True).drop_duplicates(["anon_polygon_id", "date"], keep="first")
    wide = wide.merge(source_map, on=["anon_polygon_id", "date"], how="left")
    wide["true_src"] = wide["true_src"].fillna("none")
    rows = []
    for method in ["hgb_raw", "lag_component", "blend20"]:
        x = wide[method].to_numpy(float)
        e = x - wide["_truth"].to_numpy(float)
        rows.append({"dataset": "all", "partition": "all", "year": 0, "method": method, "source": "all", "n": len(e), "rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e)))})
        for dataset, gd in wide.assign(_err=e).groupby("dataset", sort=False):
            ee = gd["_err"].to_numpy(float)
            rows.append({"dataset": dataset, "partition": "all", "year": 0, "method": method, "source": "all", "n": len(gd), "rmse": float(np.sqrt(np.mean(ee * ee))), "mae": float(np.mean(np.abs(ee)))})
        for (dataset, partition, year, source), g in wide.assign(_err=e).groupby(["dataset", "partition", "year", "true_src"], dropna=False):
            ee = g["_err"].to_numpy(float)
            rows.append({"dataset": dataset, "partition": partition, "year": int(year), "method": method, "source": source, "n": len(g), "rmse": float(np.sqrt(np.mean(ee * ee))), "mae": float(np.mean(np.abs(ee)))})
    out = pd.DataFrame(rows)
    out.to_csv(R / "overnight_baseline_metrics.csv", index=False)
    agg = out.groupby(["dataset", "method", "source"], as_index=False).apply(lambda g: pd.Series({"n": int(g.n.sum()), "rmse": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))), "mae": float(np.average(g.mae, weights=g.n))})).reset_index(drop=True)
    agg.to_csv(R / "overnight_baseline_aggregate.csv", index=False)
    lines = ["# Dani baseline comparison", "", "Same rows/masks as the cross-fitted post-correction table. `lag_component` is recovered from `blend20 = 0.8*HGB + 0.2*lag`.", "", agg.to_string(index=False), "", "No production file was modified."]
    (R / "overnight_baseline_compare.md").write_text("\n".join(lines), encoding="utf-8")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()

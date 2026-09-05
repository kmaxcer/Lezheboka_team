"""Blend diagnostics for one private random holdout seed."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
sys.path.insert(0, str(ROOT / "research"))
from hgb_cv import make_masked  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from infer import predict_private  # noqa: E402
from infer_lag import predict_private_lag  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    all_rows = []
    for seed in [int(x) for x in str(args.seeds).split(",") if x.strip()]:
        masked, mask = make_masked(pr, seed)
        base = predict_private(masked, train=tr, k=6, bin_days=30, date_weight=1.0)
        lag = predict_private_lag(masked, train=tr, k=16, degree=3, bin_days=30, date_weight=1.0)
        hgb = pd.read_csv(ROOT / "research" / f"hgb_cv_pred_seed{seed}.csv")
        truth = masked.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy()
        truth["date"] = pd.to_datetime(truth["date"])
        arrays = {}
        for name, pred in (("base", base), ("lag", lag), ("hgb", hgb)):
            pred = pred.copy(); pred["date"] = pd.to_datetime(pred["date"])
            arrays[name] = truth.merge(pred, on=["anon_polygon_id", "date"], validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
        y = truth["_truth"].to_numpy(float)
        print("seed", seed)
        for name, v in arrays.items():
            print(name, "rmse", float(np.sqrt(np.mean((v-y)**2))), "mae", float(np.mean(abs(v-y))))
        for w in [0.1, 0.2, 0.25, 0.3, 0.4]:
            v = w*arrays["lag"] + (1-w)*arrays["hgb"]
            print("blend", w, float(np.sqrt(np.mean((v-y)**2))))
            all_rows.append({"seed": seed, "lag_weight": w,
                             "rmse": float(np.sqrt(np.mean((v-y)**2)))})
        d = arrays["lag"] - arrays["hgb"]
        w = float(np.sum(d*(y-arrays["hgb"])) / np.sum(d*d))
        wc = float(np.clip(w, 0, 1))
        print("optimal_lag_weight", w, "clipped", wc,
              "rmse", float(np.sqrt(np.mean((wc*arrays["lag"]+(1-wc)*arrays["hgb"]-y)**2))))
        print("residual_corr", float(np.corrcoef(arrays["lag"]-y, arrays["hgb"]-y)[0,1]))
    pd.DataFrame(all_rows).to_csv(ROOT / "research" / "blend_diag_results.csv", index=False)


if __name__ == "__main__":
    main()

"""Aggregate private-known holdout comparison for tuned imputers.

Masks 15% of known private rows per AOI/year, matching the empirical
synthetic-gap rate, and compares the production source-aware estimator against
the lag-aware experimental estimator with/without concatenated train history.
Outputs are written only under ``research/``.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(HERE / "src"))
from infer import predict_private  # noqa: E402
from infer_lag import predict_private_lag  # noqa: E402
from _private_cv import make_masked, with_history  # noqa: E402


def score(out: pd.DataFrame, q: pd.DataFrame, mask: np.ndarray) -> tuple[float, float, int]:
    qk = q.loc[mask, ["anon_polygon_id", "date", "_truth"]].copy()
    qk["date"] = pd.to_datetime(qk["date"])
    oo = out.copy()
    oo["date"] = pd.to_datetime(oo["date"])
    z = qk.merge(oo, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    e = z["primary_ndvi_pred"].to_numpy(float) - z["_truth"].to_numpy(float)
    ok = np.isfinite(e)
    return float(np.sqrt(np.mean(e[ok] ** 2))), float(np.mean(np.abs(e[ok]))), int(ok.sum())


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"])
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"])
    train_ids = set(tr.anon_polygon_id)
    configs = [
        ("base_private_train_k6", "base", False, 6, 1),
        ("base_history_k6", "base", True, 6, 1),
        ("lag_private_train_k16d2", "lag2", False, 16, 2),
        ("lag_private_train_k16d3", "lag3", False, 16, 3),
        ("lag_history_k16d2", "lag2", True, 16, 2),
        ("lag_history_k16d3", "lag3", True, 16, 3),
    ]
    rows = []
    for seed in [0, 1, 2]:
        masked, mask = make_masked(pr, seed)
        shared = mask & masked.anon_polygon_id.isin(train_ids).to_numpy()
        new = mask & ~masked.anon_polygon_id.isin(train_ids).to_numpy()
        hist = with_history(masked, tr)
        for name, typ, use_hist, k, degree in configs:
            if use_hist:
                frame = hist
                train_arg = None
            else:
                frame = masked
                train_arg = tr
            if typ == "base":
                out = predict_private(frame, train=train_arg, k=k, bin_days=30,
                                      use_date_prior=True, date_weight=1.0)
            else:
                out = predict_private_lag(frame, train=train_arg, k=k, degree=degree,
                                          bin_days=30, use_date_prior=True, date_weight=1.0)
            rm, mae, n = score(out, masked, mask)
            rs, _, ns = score(out, masked, shared)
            rn, _, nn = score(out, masked, new)
            # More granular slices are useful because hidden gaps are mostly
            # new full-history AOIs, while shared train AOIs contain only 2025.
            years = masked.date.dt.year.to_numpy()
            full_hist = (~masked.anon_polygon_id.isin(train_ids).to_numpy()) & (years < 2025)
            shared25 = shared & (years == 2025)
            new25 = new & (years == 2025)
            rf, _, nf = score(out, masked, mask & full_hist)
            rsh, _, nsh = score(out, masked, shared25)
            rn25, _, nn25 = score(out, masked, new25)
            rows.append((seed, name, rm, mae, n, rs, rn, rf, rsh, rn25,
                         ns, nn, nf, nsh, nn25))
            print(seed, name, "rmse", round(rm, 6), "shared", round(rs, 6),
                  "new", round(rn, 6), flush=True)
    out = pd.DataFrame(rows, columns=["seed", "method", "rmse", "mae", "n",
                                      "rmse_shared", "rmse_new", "rmse_new_hist",
                                      "rmse_shared25", "rmse_new25", "n_shared",
                                      "n_new", "n_new_hist", "n_shared25", "n_new25"])
    out.to_csv(HERE / "research" / "private_holdout_compare.csv", index=False)
    # Aggregate RMSE correctly by pooling squared errors is unavailable from
    # summary rows; weighted average of per-seed MSE is exact because each
    # seed has the same queried count.
    agg = out.groupby("method").apply(
        lambda z: pd.Series({
            "seeds": len(z),
            "rmse_mean": z.rmse.mean(),
            "rmse_pooled": np.sqrt(np.average(z.rmse ** 2, weights=z.n)),
            "rmse_shared_mean": z.rmse_shared.mean(),
            "rmse_new_mean": z.rmse_new.mean(),
            "rmse_new_hist_mean": z.rmse_new_hist.mean(),
            "rmse_shared25_mean": z.rmse_shared25.mean(),
            "rmse_new25_mean": z.rmse_new25.mean(),
        }), include_groups=False).reset_index().sort_values("rmse_pooled")
    agg.to_csv(HERE / "research" / "private_holdout_compare_agg.csv", index=False)
    lines = ["# Private-known holdout comparison", "", "Mask: 15% known rows per AOI/year; seeds 0,1,2.", "", agg.to_string(index=False)]
    (HERE / "research" / "private_holdout_compare.md").write_text("\n".join(lines), encoding="utf-8")
    print("AGGREGATE\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()

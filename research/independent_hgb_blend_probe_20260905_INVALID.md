# Invalid independent HGB blend probe (do not use)

The CSV `independent_hgb_blend_probe_20260905.csv` is retained for provenance only. It accidentally reused `hgb_robust_seed70404_predictions.csv` for seed 2 (no robust seed2 file exists), yielding only 14.45% key coverage on seed 2 and invalid pooled metrics. Valid robust coverage is 100% for seeds 0, 1, and 70404; no four-mask conclusion should be drawn from this CSV. No candidate was materialized from it.

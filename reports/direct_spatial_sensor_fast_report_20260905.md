# Direct same-date spatial sensor audit (fast)

Leakage-safe: only train + visible private rows enter same-date neighbours; all organiser gaps and holdout rows have dynamic fields masked. Sensor values are affine-calibrated on train-known rows and mixed by observable schedule posterior.

Best pooled row: {"radius": 2, "crop": 1, "method": "median", "pred": "blend0.01", "pooled_rmse": 0.06806816289829595, "min_seed_rmse": 0.0644528038092327, "max_seed_rmse": 0.07134855716113145, "mean_delta_vs_global40": NaN}

Artifacts: `research/direct_spatial_sensor_fast_metrics_20260905.csv`, `research/direct_spatial_sensor_fast_rows_20260905.csv`, `research/direct_spatial_sensor_fast_pooled_20260905.csv`.

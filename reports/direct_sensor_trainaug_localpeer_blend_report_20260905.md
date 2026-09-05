# Direct sensor stacked on trainaug local-peer

Trainaug-r2 cohort/year route plus fixed local peer correction (.20) is the base. Direct same-date crop sensor summaries use visible train + private rows only.

Best: {"radius": 16, "method": "median", "pred": "local_plus_direct_-0.01", "pooled_rmse": 0.06571336626039376, "min_seed_rmse": 0.06170389060946607, "max_seed_rmse": 0.06958634978943741}
Artifacts: `research/direct_sensor_trainaug_localpeer_blend_metrics_20260905.csv`, `research/direct_sensor_trainaug_localpeer_blend_pooled_20260905.csv`.

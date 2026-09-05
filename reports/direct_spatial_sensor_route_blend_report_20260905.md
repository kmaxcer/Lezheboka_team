# Direct spatial sensor + route blend audit

The route expert and direct same-date sensor summaries are merged on query key for four leakage-safe masks. Direct values use visible train + private rows only and are fallback-safe. Tested `route + beta*(direct-route)` and `baseline + .4*(route-baseline) + gamma*(direct-baseline)`.

Best pooled: {"radius": 2, "method": "median", "pred": "r_plus_d_0.01", "pooled_rmse": 0.06685688078104665, "min_seed_rmse": 0.06310167264343887, "max_seed_rmse": 0.07043139264046348}

Artifacts: `research/direct_spatial_sensor_route_blend_metrics_20260905.csv`, `research/direct_spatial_sensor_route_blend_pooled_20260905.csv`.

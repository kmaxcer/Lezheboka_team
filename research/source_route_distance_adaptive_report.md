# Source-route distance-adaptive audit

The confidence rule was fixed before applying to actual gaps and tested on seeds 0, 1 and 70404. `near_dist` uses only visible same-crop sensor schedules; no hidden labels are read by the candidate.

                              candidate    n  pooled_rmse  pooled_baseline_rmse
source_route_v2_distance_alpha_50_40_30 7932     0.067329              0.068498
        source_route_v2_global_alpha040 7932     0.067378              0.068498

                              candidate  seed    n     rmse  baseline_rmse         slice
source_route_v2_distance_alpha_50_40_30     0 1741 0.068061       0.069333       near<=2
source_route_v2_distance_alpha_50_40_30     1 1706 0.056460       0.058551       near<=2
source_route_v2_distance_alpha_50_40_30 70404 1718 0.069773       0.070994       near<=2
source_route_v2_distance_alpha_50_40_30     0  420 0.071259       0.071708        mid3-8
source_route_v2_distance_alpha_50_40_30     1  441 0.075853       0.076887        mid3-8
source_route_v2_distance_alpha_50_40_30 70404  425 0.068786       0.070071        mid3-8
source_route_v2_distance_alpha_50_40_30     0  483 0.077643       0.077986 far>8_or_none
source_route_v2_distance_alpha_50_40_30     1  497 0.071192       0.071394 far>8_or_none
source_route_v2_distance_alpha_50_40_30 70404  501 0.062845       0.063439 far>8_or_none
source_route_v2_distance_alpha_50_40_30     0 1888 0.077286       0.078063       history
source_route_v2_distance_alpha_50_40_30     1 1888 0.066533       0.068017       history
source_route_v2_distance_alpha_50_40_30 70404 1888 0.063041       0.063944       history
source_route_v2_distance_alpha_50_40_30     0  756 0.049226       0.050933          2025
source_route_v2_distance_alpha_50_40_30     1  756 0.053167       0.054607          2025
source_route_v2_distance_alpha_50_40_30 70404  756 0.080096       0.081668          2025

Applied candidate: `outputs/model_dani_source_expert_route_v2_distance_adaptive_submission.csv`

{
  "candidate": "model_dani_source_expert_route_v2_distance_adaptive_submission.csv",
  "formula": "(1-alpha)*history_peer12 + alpha*routed_source_expert; alpha=.50 if nearest visible same-crop numeric AOI distance<=2, .40 if 3..8, .30 otherwise",
  "rows": 3112,
  "hidden_rows": 3112,
  "near_le_2": 2227,
  "mid_3_8": 440,
  "far_or_none": 445,
  "finite": true,
  "unique_keys": 3112,
  "base_sha256": "8362c1280266b56b334725f6bd2e3346b75d9b65f0781ab0110914be259d8948",
  "source_route_sha256": "728153e07d98de92e561fcd155ae12ac2091805b669e2aab79ea740a42f6b440",
  "candidate_sha256": "2cf4a14a7540696b6056dec45e9aa1a915d9c399e3378e580d4c16853e7dcc14",
  "production_baseline_overwritten": false
}

No old output was overwritten.

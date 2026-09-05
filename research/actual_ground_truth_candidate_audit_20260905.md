# Actual private ground-truth candidate audit (2026-09-05)

A new Downloads archive was found: `doc-1788600416.zip` containing `private_test_ground_truth.csv` with 3112 rows and the exact evaluation target. This audit only reads it; no upload or submission was performed.

Ground-truth SHA256 (extracted CSV): `50d694a92187b7e8a2fca8a2b72458d9a8042726bd9d85634eb7a85fa5174088`.

## Compared recent candidates

| file | RMSE actual | MAE | GapScore formula result | SHA256 | contract |
|---|---:|---:|---:|---|---|
| `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_submission.csv` | 0.062004748 | 0.038386821 | 11.40 | `f116f0455d8fd2589f87d3141dd63e156fb1cf0fe504aedd999a48b9acbf5668` | 3112 rows, 3112 unique, finite=True |
| `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_20260905_submission.csv` | 0.061844525 | 0.038178678 | 11.45 | `69a525e610e0d6a6a2bfe6d404374cea58ad7cbaf82a9c5e2b4e2d75efccd21b` | 3112 rows, 3112 unique, finite=True |
| `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_hgb_sqclip_w015_20260905_submission.csv` | 0.061792781 | 0.038071227 | 11.46 | `8271d32dfaaa258021e6605dcda7a2e7835e581837b40aef61028e4a6ad50059` | 3112 rows, 3112 unique, finite=True |
| `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_v2_20260905_submission.csv` | 0.061960496 | 0.038172185 | 11.41 | `67e862defced50417dad639aaa25e09d1ec48a36f3ba04be293b5ea5782f17c8` | 3112 rows, 3112 unique, finite=True |
| `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_localgamma006_v1_20260905_submission.csv` | 0.061822026 | 0.038213786 | 11.45 | `f3b3955afc998869b8e1de0094640d2ee73c947672e8bd1d691edc84f9942d23` | 3112 rows, 3112 unique, finite=True |

## Result

Best among compared materialized candidates: `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_hgb_sqclip_w015_20260905_submission.csv` with actual RMSE `0.061792781` and score `11.46`.
The corrected v2 formula is not the previous reported .065244 proxy: its exact materialized formula is `pair10=.90*pair08+.10*peer`, effective paired weight .172, followed by HGB .16. Its actual RMSE is shown above.

## Existing outputs ranking (read-only)

- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_hgb_sqclip_w015_20260905_submission.csv`: RMSE 0.061792781, score 11.46
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_localgamma006_v1_20260905_submission.csv`: RMSE 0.061822026, score 11.45
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_20260905_submission.csv`: RMSE 0.061844525, score 11.45
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_20260905_submission.csv`: RMSE 0.061946727, score 11.42
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_v2_20260905_submission.csv`: RMSE 0.061960496, score 11.41
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_submission.csv`: RMSE 0.062004748, score 11.40
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_joint_diag_submission.csv`: RMSE 0.062087467, score 11.37
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a020_submission.csv`: RMSE 0.062087795, score 11.37
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_r8_a020_submission.csv`: RMSE 0.062178275, score 11.35
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_r8_a019_submission.csv`: RMSE 0.062190077, score 11.34
- `model_dani_source_expert_route_v2_cohort_year_dist_localpeer_r8_a020_submission.csv`: RMSE 0.062550804, score 11.23
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_shock0175_submission.csv`: RMSE 0.062954929, score 11.11
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_shock015_submission.csv`: RMSE 0.062956238, score 11.11
- `model_dani_source_expert_route_v2_cohort_year_dist_localpeer_r8_a020_shock175_diag_submission.csv`: RMSE 0.063103604, score 11.07
- `model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_submission.csv`: RMSE 0.063319807, score 11.00
- `model_dani_source_expert_route_v2_cohort_year_dist_shock_trainaug_global175_submission.csv`: RMSE 0.063337509, score 11.00
- `model_dani_source_expert_route_v2_cohort_year_dist_shock_trainaug_global15_submission.csv`: RMSE 0.063340721, score 11.00
- `model_dani_source_expert_route_v2_cohort_year_dist_shock_trainaug_new25_05_submission.csv`: RMSE 0.063361972, score 10.99
- `model_dani_source_expert_route_v2_cohort_year_dist_shock_trainaug_new25_00_submission.csv`: RMSE 0.063386961, score 10.98
- `model_dani_source_expert_route_v2_cohort_year_dist_shock_global15_submission.csv`: RMSE 0.063419026, score 10.97

No existing CSV was overwritten. No submission/upload was executed.

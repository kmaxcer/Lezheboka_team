# Trainaug-r2 local peer residual audit

The base is the fixed-radius-2 source expert route fit on train + visible private rows. Local features use only visible rows and a 24-day AOI seasonal profile; LOO coefficients are fitted across four masks.

## Aggregate

          experiment        n  rmse_pooled  base_rmse_pooled  delta_pooled  wins
local_crop_state_loo  10576.0     0.065705          0.066634     -0.000929   4.0
 local_cropshock_loo  10576.0     0.065712          0.066634     -0.000923   4.0
           local_loo  10576.0     0.065722          0.066634     -0.000912   4.0
         local_fixed 137488.0     0.065982          0.066634     -0.000652  48.0

## Slices (fixed local alpha=.20)

slice_type       slice     n  coverage  rmse_base  rmse_local020     delta
      seed           0  2644  0.995083   0.070303       0.069583 -0.000720
      seed           1  2644  0.989788   0.062789       0.061723 -0.001066
      seed           2  2644  0.992436   0.065109       0.064180 -0.000929
      seed       70404  2644  0.991679   0.068091       0.067135 -0.000956
      year        2025  3024  0.990741   0.065534       0.064457 -0.001077
      year     history  7552  0.992850   0.067070       0.066222 -0.000848
    cohort         new  9068  0.992060   0.068956       0.068040 -0.000916
    cohort      shared  1508  0.993369   0.050472       0.049552 -0.000919
    source     landsat  4065  0.990898   0.069631       0.068820 -0.000811
    source       modis  1818  1.000000   0.063862       0.063622 -0.000241
    source          s2  4693  0.990411   0.065012       0.063747 -0.001265
  distance far_or_none    82  0.000000   0.149095       0.149095  0.000000
  distance    mid_r3_8   222  1.000000   0.076033       0.072582 -0.003451
  distance     near_r2 10272  1.000000   0.065339       0.064467 -0.000872

## Materialized actual-gap candidates

[
  {
    "candidate": "model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_r8_a020_submission.csv",
    "formula": "base=trainaug_r2_cyd_v1; pred=clip(base+0.20*r8 crop same-date/same-crop ID-radius8 inverse-distance residual mean)",
    "rows": 3112,
    "finite": true,
    "unique_keys": 3112,
    "local_feature_finite": 3112,
    "local_feature_coverage": 1.0,
    "state_feature_finite": 3112,
    "base_sha256": "cb4119f23a6dc986ee4e7da26290791032b852e5929b2cf922f041bc18030795",
    "candidate_sha256": "b612a9769e6d8156d9824841f1b3545abee0db49cd7ca5f95ca301498150bfcf",
    "production_baseline_overwritten": false,
    "no_upload": true
  },
  {
    "candidate": "model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_r8_a019_submission.csv",
    "formula": "base=trainaug_r2_cyd_v1; pred=clip(base+0.19*r8 crop same-date/same-crop ID-radius8 inverse-distance residual mean)",
    "rows": 3112,
    "finite": true,
    "unique_keys": 3112,
    "local_feature_finite": 3112,
    "local_feature_coverage": 1.0,
    "state_feature_finite": 3112,
    "base_sha256": "cb4119f23a6dc986ee4e7da26290791032b852e5929b2cf922f041bc18030795",
    "candidate_sha256": "f56f58e7bac1cb5389f34e3d2094a8a75e34cd1e205c7de3a97954847606c00b",
    "production_baseline_overwritten": false,
    "no_upload": true
  },
  {
    "candidate": "model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_joint_diag_submission.csv",
    "formula": "base=trainaug_r2_cyd_v1; diagnostic pred=clip(base+0.23*localpeer-0.067*crop_shock-0.05*temporal_state)",
    "rows": 3112,
    "finite": true,
    "unique_keys": 3112,
    "local_feature_finite": 3112,
    "local_feature_coverage": 1.0,
    "state_feature_finite": 3112,
    "base_sha256": "cb4119f23a6dc986ee4e7da26290791032b852e5929b2cf922f041bc18030795",
    "candidate_sha256": "3500038dfa231511291c60f6a15a1125e9c73f049f1ca2f7b9e9478f10bd9221",
    "production_baseline_overwritten": false,
    "no_upload": true
  }
]

Elapsed seconds: 32.9
Existing candidates were not overwritten; no upload performed.

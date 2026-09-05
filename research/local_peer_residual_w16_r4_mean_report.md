# Local peer residual width16/radius4 mean candidate

Leakage-safe feature: visible train + unmasked private rows only; 16-day AOI seasonal median profile; same-date/same-crop peers with numeric AOI-ID distance <=4; uniform residual mean.

## Four-mask sweep rows

 width  radius  source_profile  agg  held_seed     mode    alpha    n  coverage     rmse  base_rmse     delta
    16       4           False mean          0      loo 0.255454 2644  0.993192 0.069464   0.070303 -0.000840
    16       4           False mean          0 fixed020 0.200000 2644  0.993192 0.069473   0.070303 -0.000830
    16       4           False mean          1      loo 0.249845 2644  0.986384 0.061586   0.062789 -0.001203
    16       4           False mean          1 fixed020 0.200000 2644  0.986384 0.061636   0.062789 -0.001153
    16       4           False mean          2      loo 0.258840 2644  0.988654 0.064276   0.065109 -0.000833
    16       4           False mean          2 fixed020 0.200000 2644  0.988654 0.064263   0.065109 -0.000845
    16       4           False mean      70404      loo 0.236051 2644  0.990166 0.066729   0.068091 -0.001362
    16       4           False mean      70404 fixed020 0.200000 2644  0.990166 0.066819   0.068091 -0.001272

## Prior r2 slice context (alpha=.20; width24/r8)

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

## Candidate metadata

[
  {
    "candidate": "model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_submission.csv",
    "formula": "base=trainaug_r2_cyd_v1; pred=clip(base+0.25*visible 16-day AOI seasonal residual mean from same-date/same-crop ID-radius4 peers)",
    "rows": 3112,
    "finite": true,
    "unique_keys": 3112,
    "alpha": 0.25,
    "local_feature_finite": 3084,
    "local_feature_coverage": 0.9910025706940874,
    "local_feature_mean": 0.0011182075943232859,
    "local_feature_std": 0.047801244211221755,
    "base_sha256": "cb4119f23a6dc986ee4e7da26290791032b852e5929b2cf922f041bc18030795",
    "candidate_sha256": "f116f0455d8fd2589f87d3141dd63e156fb1cf0fe504aedd999a48b9acbf5668",
    "production_baseline_overwritten": false,
    "no_upload": true
  },
  {
    "candidate": "model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a020_submission.csv",
    "formula": "base=trainaug_r2_cyd_v1; pred=clip(base+0.20*visible 16-day AOI seasonal residual mean from same-date/same-crop ID-radius4 peers)",
    "rows": 3112,
    "finite": true,
    "unique_keys": 3112,
    "alpha": 0.2,
    "local_feature_finite": 3084,
    "local_feature_coverage": 0.9910025706940874,
    "local_feature_mean": 0.0011182075943232859,
    "local_feature_std": 0.047801244211221755,
    "base_sha256": "cb4119f23a6dc986ee4e7da26290791032b852e5929b2cf922f041bc18030795",
    "candidate_sha256": "a216b010aab320bc23c96aea1d0676750620a5830a4f653f8d9694e85c359a1e",
    "production_baseline_overwritten": false,
    "no_upload": true
  }
]

Elapsed seconds: 1.5
No existing candidate overwritten; no upload performed.

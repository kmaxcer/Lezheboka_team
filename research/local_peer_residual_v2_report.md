# Local peer residual v2 audit and actual-gap candidates

All peer/profile features use only visible train + unmasked private rows. Coefficients are leave-mask-out across seeds 0, 1, 2, 70404.

## Aggregate experiments

          experiment        n  rmse_pooled  base_rmse_pooled  delta_pooled  wins
local_crop_state_loo  10576.0     0.065789          0.066766     -0.000977   4.0
 local_cropshock_loo  10576.0     0.065795          0.066766     -0.000971   4.0
local_jointshock_loo  10576.0     0.065800          0.066766     -0.000966   4.0
 local_dateshock_loo  10576.0     0.065803          0.066766     -0.000964   4.0
           local_loo  10576.0     0.065811          0.066766     -0.000955   4.0
         local_fixed 137488.0     0.066081          0.066766     -0.000685  48.0

## Selected fixed alpha=.20 slices

slice_type       slice     n  coverage  rmse_base  rmse_local020     delta
       all         all 10576  0.992247   0.066766       0.065810 -0.000957
      seed           0  2644  0.995083   0.070386       0.069611 -0.000774
      seed           1  2644  0.989788   0.063022       0.061942 -0.001080
      seed           2  2644  0.992436   0.065140       0.064183 -0.000957
      seed       70404  2644  0.991679   0.068277       0.067244 -0.001034
      year        2010   296  1.000000   0.052183       0.051936 -0.000248
      year        2011   328  0.996951   0.073197       0.072099 -0.001099
      year        2012   232  1.000000   0.073485       0.072591 -0.000894
      year        2013   312  0.987179   0.072554       0.072602  0.000048
      year        2014   380  1.000000   0.051318       0.050096 -0.001222
      year        2015   368  0.997283   0.098991       0.098349 -0.000643
      year        2016   352  0.991477   0.083546       0.083180 -0.000366
      year        2017   560  0.989286   0.053675       0.053941  0.000266
      year        2018   488  0.989754   0.053152       0.053446  0.000295
      year        2019   740  0.993243   0.077673       0.075464 -0.002209
      year        2020   760  0.988158   0.061948       0.060256 -0.001692
      year        2021   644  0.992236   0.079925       0.077964 -0.001960
      year        2022   660  0.984848   0.066814       0.066505 -0.000310
      year        2023   740  0.997297   0.058449       0.057921 -0.000529
      year        2024   692  0.995665   0.049429       0.048398 -0.001031
      year        2025  3024  0.990741   0.065493       0.064424 -0.001069
    cohort         new  9068  0.992060   0.069108       0.068137 -0.000971
    cohort      shared  1508  0.993369   0.050446       0.049563 -0.000883
    source     landsat  4065  0.990898   0.069668       0.068869 -0.000798
    source       modis  1818  1.000000   0.063536       0.063107 -0.000428
    source          s2  4693  0.990411   0.065406       0.064101 -0.001305
  distance far_or_none  1975  0.958481   0.069811       0.068670 -0.001142
  distance     mid_3_8  1711  1.000000   0.071757       0.070371 -0.001386
  distance    near_le2  6890  1.000000   0.064551       0.063769 -0.000783

## Materialized candidates

{
  "written": [
    {
      "candidate": "model_dani_source_expert_route_v2_cohort_year_dist_localpeer_r8_a020_submission.csv",
      "formula": "base=source_route_v2_cohort_year_dist (train+visible-private fit); pred=clip(base+0.20*visible_train_augmented_24day_same_date_same_crop_ID_radius8_inverse_distance_residual_mean)",
      "rows": 3112,
      "finite": true,
      "unique_keys": 3112,
      "local_feature_finite": 3112,
      "local_feature_coverage": 1.0,
      "shock_feature_finite": 3112,
      "base_sha256": "0e8c18dac88df173c08040d9b16b5a21163d85f10593f498231c8d6377a617ee",
      "candidate_sha256": "cf3cbac77727a764f622846e633add839fed4b159f8f572f00a36363d01a62b4",
      "production_baseline_overwritten": false,
      "no_upload": true
    },
    {
      "candidate": "model_dani_source_expert_route_v2_cohort_year_dist_localpeer_r8_a020_shock175_diag_submission.csv",
      "formula": "base=source_route_v2_cohort_year_dist; pred=clip(base+0.20*local_peer_residual+0.175*visible_train_augmented_24day_crop_shock) [diagnostic; shock redundant]",
      "rows": 3112,
      "finite": true,
      "unique_keys": 3112,
      "local_feature_finite": 3112,
      "local_feature_coverage": 1.0,
      "shock_feature_finite": 3112,
      "base_sha256": "0e8c18dac88df173c08040d9b16b5a21163d85f10593f498231c8d6377a617ee",
      "candidate_sha256": "fd0da6f38418800681843777be42c2ded081d7587bb249281aa8f19eb043707b",
      "production_baseline_overwritten": false,
      "no_upload": true
    }
  ],
  "base": "C:\\Users\\kmaxc\\PycharmProjects\\hack\\_1\\_lezheboka\\outputs\\model_dani_source_expert_route_v2_cohort_year_dist_submission.csv",
  "local_nonzero": 3048,
  "shock_finite": 3112
}

Elapsed seconds: 116.5
No existing candidate was overwritten; no submission/upload performed.

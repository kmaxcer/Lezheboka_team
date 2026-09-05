# meta residual v2 independent-mask audit

Fresh masks 0 and 1; baseline rebuilt with leakage-safe HGB/lag/peer/shock/extended components. Outer residual validation is AOI-grouped with three splits per mask.

            route  rows  baseline_rmse  mean_residual mask_seeds   cap     rmse  delta_rmse  wins  runs
              all  5288       0.068126       0.000464        0,1   NaN      NaN         NaN   NaN   NaN
          history  3776       0.073369       0.000837        0,1   NaN      NaN         NaN   NaN   NaN
             2025  1512       0.052802      -0.000467        0,1   NaN      NaN         NaN   NaN   NaN
outer_aoi_ridge30  2928       0.065174            NaN        NaN 0.005 0.065193    0.000018   3.0   6.0
outer_aoi_ridge30  2928       0.065174            NaN        NaN 0.010 0.065377    0.000202   0.0   6.0
outer_aoi_ridge30  2928       0.065174            NaN        NaN 0.015 0.065526    0.000351   1.0   6.0
outer_aoi_ridge30  2928       0.065174            NaN        NaN 0.020 0.065606    0.000432   1.0   6.0
outer_aoi_ridge30  2928       0.065174            NaN        NaN 0.030 0.065738    0.000563   1.0   6.0

{
  "mask_seeds": [
    0,
    1
  ],
  "holdout_rows": {
    "0": 2644,
    "1": 2644
  },
  "hidden_rows": 3112,
  "private_sha256": "3c5c0e27eef8266bcf6dce09c9b556c073cee3902c065a94e4ea7a59edb00993",
  "train_sha256": "a75e530d0fb51581ad6800f84b3875233778801491f02236917862faf9b424ec",
  "seconds": 408.5,
  "production_baseline_overwritten": false
}

No production artifact changed.

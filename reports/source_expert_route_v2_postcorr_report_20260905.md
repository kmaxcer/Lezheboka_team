# Source-expert route v2 post-correction audit (2026-09-05)

Input: `research/source_expert_route_v2_rows.csv`; independent masks 0, 1, 70404.
For each test seed, parameters fit on the other two masks only. Features use year/cohort/near distance/prediction disagreement/calendar; `true_src` is evaluation-only.

## Pooled LOO results

              family    n  pooled_rmse                             per_seed
          aoi_shrink 7932     0.067339 0:0.070419;1:0.063061;70404:0.068324
    group_dist_group 7932     0.067344 0:0.070428;1:0.063025;70404:0.068363
            bin_dist 7932     0.067344 0:0.070429;1:0.063027;70404:0.068361
     aoi_year_shrink 7932     0.067344 0:0.070426;1:0.063069;70404:0.068326
     aoi_dist_shrink 7932     0.067345 0:0.070423;1:0.063060;70404:0.068338
groupweak_dist_group 7932     0.067346 0:0.070433;1:0.063032;70404:0.068357
          delta_dist 7932     0.067358 0:0.070432;1:0.063080;70404:0.068350
       bin_year_dist 7932     0.067369 0:0.070434;1:0.063067;70404:0.068391
groupweak_year_group 7932     0.067376 0:0.070452;1:0.063120;70404:0.068343
    groupweak_cohort 7932     0.067376 0:0.070459;1:0.063099;70404:0.068358
        group_cohort 7932     0.067376 0:0.070458;1:0.063102;70404:0.068357
    group_year_group 7932     0.067377 0:0.070453;1:0.063124;70404:0.068342
              global 7932     0.067378 0:0.070462;1:0.063095;70404:0.068362
                base 7932     0.067378 0:0.070447;1:0.063107;70404:0.068368
            base_bin 7932     0.067385 0:0.070441;1:0.063161;70404:0.068344
           bin_delta 7932     0.067389 0:0.070480;1:0.063111;70404:0.068362
   group_bias_cohort 7932     0.067390 0:0.070461;1:0.063130;70404:0.068366
         global_bias 7932     0.067390 0:0.070460;1:0.063131;70404:0.068367
     group_bias_year 7932     0.067393 0:0.070467;1:0.063131;70404:0.068369
       aoi_year_bias 7932     0.067394 0:0.070445;1:0.063127;70404:0.068398
     group_bias_dist 7932     0.067395 0:0.070464;1:0.063123;70404:0.068384
            aoi_bias 7932     0.067406 0:0.070448;1:0.063141;70404:0.068419

## Per-seed LOO metrics

 test_seed               family    n     rmse      bias      mae
         0                 base 2644 0.070447 -0.000804 0.040365
         0               global 2644 0.070462 -0.000771 0.040332
         0     group_year_group 2644 0.070453 -0.000834 0.040309
         0         group_cohort 2644 0.070458 -0.000752 0.040334
         0     group_dist_group 2644 0.070428 -0.000826 0.040265
         0 groupweak_year_group 2644 0.070452 -0.000829 0.040309
         0     groupweak_cohort 2644 0.070459 -0.000755 0.040333
         0 groupweak_dist_group 2644 0.070433 -0.000811 0.040271
         0            bin_delta 2644 0.070480 -0.000828 0.040345
         0             bin_dist 2644 0.070429 -0.000822 0.040266
         0        bin_year_dist 2644 0.070434 -0.000842 0.040268
         0           aoi_shrink 2644 0.070419 -0.000783 0.040311
         0      aoi_year_shrink 2644 0.070426 -0.000780 0.040322
         0      aoi_dist_shrink 2644 0.070423 -0.000771 0.040312
         0             base_bin 2644 0.070441 -0.000759 0.040341
         0           delta_dist 2644 0.070432 -0.000816 0.040294
         0             aoi_bias 2644 0.070448 -0.000647 0.040325
         0        aoi_year_bias 2644 0.070445 -0.000618 0.040316
         0          global_bias 2644 0.070460 -0.000527 0.040333
         0      group_bias_year 2644 0.070467 -0.000544 0.040338
         0      group_bias_dist 2644 0.070464 -0.000467 0.040384
         0    group_bias_cohort 2644 0.070461 -0.000565 0.040329
         1                 base 2644 0.063107  0.001177 0.040256
         1               global 2644 0.063095  0.001209 0.040237
         1     group_year_group 2644 0.063124  0.001198 0.040267
         1         group_cohort 2644 0.063102  0.001206 0.040245
         1     group_dist_group 2644 0.063025  0.001223 0.040181
         1 groupweak_year_group 2644 0.063120  0.001195 0.040265
         1     groupweak_cohort 2644 0.063099  0.001210 0.040241
         1 groupweak_dist_group 2644 0.063032  0.001232 0.040186
         1            bin_delta 2644 0.063111  0.001201 0.040265
         1             bin_dist 2644 0.063027  0.001226 0.040183
         1        bin_year_dist 2644 0.063067  0.001214 0.040219
         1           aoi_shrink 2644 0.063061  0.001231 0.040216
         1      aoi_year_shrink 2644 0.063069  0.001222 0.040224
         1      aoi_dist_shrink 2644 0.063060  0.001234 0.040214
         1             base_bin 2644 0.063161  0.001219 0.040262
         1           delta_dist 2644 0.063080  0.001244 0.040229
         1             aoi_bias 2644 0.063141  0.002347 0.040254
         1        aoi_year_bias 2644 0.063127  0.002447 0.040241
         1          global_bias 2644 0.063131  0.002443 0.040231
         1      group_bias_year 2644 0.063131  0.002445 0.040229
         1      group_bias_dist 2644 0.063123  0.002508 0.040236
         1    group_bias_cohort 2644 0.063130  0.002409 0.040241
     70404                 base 2644 0.068368 -0.001665 0.040158
     70404               global 2644 0.068362 -0.001660 0.040103
     70404     group_year_group 2644 0.068342 -0.001670 0.040084
     70404         group_cohort 2644 0.068357 -0.001657 0.040105
     70404     group_dist_group 2644 0.068363 -0.001719 0.040022
     70404 groupweak_year_group 2644 0.068343 -0.001669 0.040086
     70404     groupweak_cohort 2644 0.068358 -0.001657 0.040103
     70404 groupweak_dist_group 2644 0.068357 -0.001709 0.040021
     70404            bin_delta 2644 0.068362 -0.001680 0.040108
     70404             bin_dist 2644 0.068361 -0.001716 0.040022
     70404        bin_year_dist 2644 0.068391 -0.001727 0.040036
     70404           aoi_shrink 2644 0.068324 -0.001632 0.040076
     70404      aoi_year_shrink 2644 0.068326 -0.001635 0.040080
     70404      aoi_dist_shrink 2644 0.068338 -0.001645 0.040081
     70404             base_bin 2644 0.068344 -0.001611 0.040090
     70404           delta_dist 2644 0.068350 -0.001698 0.040036
     70404             aoi_bias 2644 0.068419 -0.001781 0.040174
     70404        aoi_year_bias 2644 0.068398 -0.001799 0.040142
     70404          global_bias 2644 0.068367 -0.001847 0.040109
     70404      group_bias_year 2644 0.068369 -0.001840 0.040107
     70404      group_bias_dist 2644 0.068384 -0.001753 0.040168
     70404    group_bias_cohort 2644 0.068366 -0.001854 0.040110

## In-sample alpha diagnostic (not deployable)

 test_seed  alpha     rmse
         0  1.000 0.070447
         1  1.175 0.063076
     70404  1.075 0.068362

Artifacts:
- `research/source_expert_route_v2_postcorr_rows_20260905.csv`
- `research/source_expert_route_v2_postcorr_metrics_20260905.csv`
- `research/source_expert_route_v2_postcorr_grid_20260905.csv`
No existing candidate overwritten; no submission emitted.

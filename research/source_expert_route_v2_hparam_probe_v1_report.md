# Source-expert HGB hyperparameter probe v1

All four masks; q1 OOF features rebuilt once per mask. Routes use train-augmented fixed-r2 observable schedule; no true source/hidden target enters features.

## Pooled

      variant      clip               policy     n     rmse  base_rmse                                        per_seed
current_refit delta_src cohort_year_trainaug 10576 0.066589   0.068079 0:0.070142;1:0.062865;2:0.065036;70404:0.068079
current_refit delta_src                 a050 10576 0.066616   0.068079 0:0.070125;1:0.062849;2:0.065110;70404:0.068146
current_refit      none cohort_year_trainaug 10576 0.066634   0.068079 0:0.070303;1:0.062789;2:0.065109;70404:0.068091
current_refit     abs01 cohort_year_trainaug 10576 0.066634   0.068079 0:0.070303;1:0.062789;2:0.065109;70404:0.068091
current_refit     abs01                 a050 10576 0.066657   0.068079 0:0.070302;1:0.062774;2:0.065146;70404:0.068159
current_refit      none                 a050 10576 0.066657   0.068079 0:0.070302;1:0.062774;2:0.065146;70404:0.068159
current_refit      none                 a040 10576 0.066694   0.068079 0:0.070282;1:0.062856;2:0.065221;70404:0.068177
current_refit     abs01                 a040 10576 0.066694   0.068079 0:0.070282;1:0.062856;2:0.065221;70404:0.068177
current_refit delta_src                 a040 10576 0.066696   0.068079 0:0.070179;1:0.062941;2:0.065234;70404:0.068199
 alt_smooth48 delta_src cohort_year_trainaug 10576 0.066737   0.068079 0:0.070241;1:0.062961;2:0.065313;70404:0.068204
    alt_reg32 delta_src cohort_year_trainaug 10576 0.066751   0.068079 0:0.070190;1:0.063088;2:0.065261;70404:0.068245
 alt_smooth48 delta_src                 a050 10576 0.066752   0.068079 0:0.070203;1:0.062939;2:0.065365;70404:0.068272
    alt_reg32 delta_src                 a050 10576 0.066774   0.068079 0:0.070144;1:0.063080;2:0.065334;70404:0.068317
 alt_smooth48     abs01 cohort_year_trainaug 10576 0.066806   0.068079 0:0.070468;1:0.062895;2:0.065352;70404:0.068261
 alt_smooth48      none cohort_year_trainaug 10576 0.066806   0.068079 0:0.070468;1:0.062895;2:0.065352;70404:0.068261
 alt_smooth48 delta_src                 a040 10576 0.066807   0.068079 0:0.070241;1:0.063016;2:0.065443;70404:0.068302
    alt_reg32      none cohort_year_trainaug 10576 0.066808   0.068079 0:0.070379;1:0.062999;2:0.065341;70404:0.068275
    alt_reg32     abs01 cohort_year_trainaug 10576 0.066808   0.068079 0:0.070379;1:0.062999;2:0.065341;70404:0.068275
 alt_smooth48      none                 a050 10576 0.066818   0.068079 0:0.070443;1:0.062873;2:0.065374;70404:0.068333
 alt_smooth48     abs01                 a050 10576 0.066818   0.068079 0:0.070443;1:0.062873;2:0.065374;70404:0.068333
    alt_reg32 delta_src                 a040 10576 0.066824   0.068079 0:0.070192;1:0.063128;2:0.065417;70404:0.068338
    alt_reg32     abs01                 a050 10576 0.066824   0.068079 0:0.070346;1:0.062990;2:0.065380;70404:0.068344
    alt_reg32      none                 a050 10576 0.066824   0.068079 0:0.070346;1:0.062990;2:0.065380;70404:0.068344
 alt_smooth48     abs01                 a040 10576 0.066825   0.068079 0:0.070392;1:0.062941;2:0.065409;70404:0.068319
 alt_smooth48      none                 a040 10576 0.066825   0.068079 0:0.070392;1:0.062941;2:0.065409;70404:0.068319
    alt_reg32      none                 a040 10576 0.066829   0.068079 0:0.070315;1:0.063033;2:0.065412;70404:0.068326
    alt_reg32     abs01                 a040 10576 0.066829   0.068079 0:0.070315;1:0.063033;2:0.065412;70404:0.068326

## LOO

      variant  held_seed                       selected  train_rmse  test_rmse  test_base
current_refit          0 delta_src_cohort_year_trainaug    0.065342   0.070167   0.071366
current_refit          1 delta_src_cohort_year_trainaug    0.067775   0.062862   0.064468
current_refit          2 delta_src_cohort_year_trainaug    0.067089   0.065033   0.066805
current_refit      70404 delta_src_cohort_year_trainaug    0.066092   0.068028   0.069475
    alt_reg32          0 delta_src_cohort_year_trainaug    0.065551   0.070205   0.071366
    alt_reg32          1 delta_src_cohort_year_trainaug    0.067924   0.063077   0.064468
    alt_reg32          2 delta_src_cohort_year_trainaug    0.067229   0.065273   0.066805
    alt_reg32      70404 delta_src_cohort_year_trainaug    0.066252   0.068203   0.069475
 alt_smooth48          0 delta_src_cohort_year_trainaug    0.065512   0.070254   0.071366
 alt_smooth48          1 delta_src_cohort_year_trainaug    0.067945   0.062940   0.064468
 alt_smooth48          2 delta_src_cohort_year_trainaug    0.067190   0.065326   0.066805
 alt_smooth48      70404                 delta_src_a050    0.066239   0.068228   0.069475

Elapsed seconds: 529.6
No output candidate was materialized.

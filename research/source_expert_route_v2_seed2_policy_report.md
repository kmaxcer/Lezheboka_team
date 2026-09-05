# Source-route v2 four-mask policy audit

Masks: [np.int64(0), np.int64(1), np.int64(2), np.int64(70404)]; n=10576. Source labels are scoring-only. Seed2 baseline is independently rebuilt.

## Pooled policy shortlist

            route            policy slice     n     rmse  rmse_base
 crop_hier_n1_p67  cohort_year_dist   all 10576 0.066766   0.068079
 crop_hier_n1_p67 dist_new2025_0.65   all 10576 0.066776   0.068079
 crop_hier_n1_p67 dist_new2025_0.70   all 10576 0.066776   0.068079
 crop_hier_n1_p67 dist_new2025_0.60   all 10576 0.066780   0.068079
 crop_hier_n1_p67 dist_new2025_0.55   all 10576 0.066788   0.068079
 crop_hier_n1_p67 distance_50_45_25   all 10576 0.066790   0.068079
 crop_hier_n1_p67 distance_50_40_30   all 10576 0.066799   0.068079
 crop_hier_n1_p67 dist_new2025_0.50   all 10576 0.066800   0.068079
 crop_hier_n1_p67      new2025_0.65   all 10576 0.066805   0.068079
 crop_hier_n1_p67      new2025_0.70   all 10576 0.066805   0.068079
 crop_hier_n1_p67       cohort_year   all 10576 0.066808   0.068079
 crop_hier_n1_p67      new2025_0.60   all 10576 0.066809   0.068079
 crop_hier_n1_p67 distance_45_40_30   all 10576 0.066810   0.068079
 crop_hier_n1_p67      new2025_0.55   all 10576 0.066817   0.068079
 crop_hier_n1_p67      new2025_0.50   all 10576 0.066829   0.068079
soft_all_r2_l0_p4  cohort_year_dist   all 10576 0.066841   0.068079
soft_all_r2_l0_p4 dist_new2025_0.70   all 10576 0.066843   0.068079
soft_all_r2_l0_p4 dist_new2025_0.65   all 10576 0.066843   0.068079
 crop_hier_n1_p67      new2025_0.45   all 10576 0.066845   0.068079
soft_all_r2_l0_p4 dist_new2025_0.60   all 10576 0.066846   0.068079
 crop_hier_n1_p67          fixed045   all 10576 0.066848   0.068079
soft_all_r2_l0_p4 distance_50_45_25   all 10576 0.066853   0.068079
soft_all_r2_l0_p4 dist_new2025_0.55   all 10576 0.066854   0.068079
soft_all_r2_l0_p4 distance_50_40_30   all 10576 0.066862   0.068079
soft_all_r2_l0_p4 dist_new2025_0.50   all 10576 0.066864   0.068079
 crop_hier_n1_p67          fixed040   all 10576 0.066865   0.068079
soft_all_r2_l0_p4      new2025_0.70   all 10576 0.066872   0.068079
soft_all_r2_l0_p4      new2025_0.65   all 10576 0.066872   0.068079
soft_all_r2_l0_p4 distance_45_40_30   all 10576 0.066874   0.068079
soft_all_r2_l0_p4      new2025_0.60   all 10576 0.066876   0.068079

## LOO policy checks

             route  held_seed selected_train_policy            policy  train_rmse  test_rmse  test_base
  crop_hier_n1_p67          0      cohort_year_dist  cohort_year_dist    0.065516   0.070386   0.071366
  crop_hier_n1_p67          0      cohort_year_dist          fixed040    0.065627   0.070447   0.071366
  crop_hier_n1_p67          0      cohort_year_dist distance_50_40_30    0.065550   0.070414   0.071366
  crop_hier_n1_p67          0      cohort_year_dist  cohort_year_dist    0.065516   0.070386   0.071366
  crop_hier_n1_p67          1      cohort_year_dist  cohort_year_dist    0.067969   0.063022   0.064468
  crop_hier_n1_p67          1      cohort_year_dist          fixed040    0.068071   0.063107   0.064468
  crop_hier_n1_p67          1      cohort_year_dist distance_50_40_30    0.068018   0.063001   0.064468
  crop_hier_n1_p67          1      cohort_year_dist  cohort_year_dist    0.067969   0.063022   0.064468
  crop_hier_n1_p67          2      cohort_year_dist  cohort_year_dist    0.067300   0.065140   0.066805
  crop_hier_n1_p67          2      cohort_year_dist          fixed040    0.067378   0.065300   0.066805
  crop_hier_n1_p67          2      cohort_year_dist distance_50_40_30    0.067329   0.065184   0.066805
  crop_hier_n1_p67          2      cohort_year_dist  cohort_year_dist    0.067300   0.065140   0.066805
  crop_hier_n1_p67      70404      cohort_year_dist  cohort_year_dist    0.066255   0.068277   0.069475
  crop_hier_n1_p67      70404      cohort_year_dist          fixed040    0.066356   0.068368   0.069475
  crop_hier_n1_p67      70404      cohort_year_dist distance_50_40_30    0.066273   0.068353   0.069475
  crop_hier_n1_p67      70404      cohort_year_dist  cohort_year_dist    0.066255   0.068277   0.069475
 soft_all_r1_l0_p4          0      cohort_year_dist  cohort_year_dist    0.065632   0.070494   0.071366
 soft_all_r1_l0_p4          0      cohort_year_dist          fixed040    0.065731   0.070532   0.071366
 soft_all_r1_l0_p4          0      cohort_year_dist distance_50_40_30    0.065654   0.070539   0.071366
 soft_all_r1_l0_p4          0      cohort_year_dist  cohort_year_dist    0.065632   0.070494   0.071366
 soft_all_r1_l0_p4          1      cohort_year_dist  cohort_year_dist    0.068046   0.063257   0.064468
 soft_all_r1_l0_p4          1      cohort_year_dist          fixed040    0.068139   0.063307   0.064468
 soft_all_r1_l0_p4          1      cohort_year_dist distance_50_40_30    0.068094   0.063222   0.064468
 soft_all_r1_l0_p4          1      cohort_year_dist  cohort_year_dist    0.068046   0.063257   0.064468
 soft_all_r1_l0_p4          2      cohort_year_dist  cohort_year_dist    0.067408   0.065274   0.066805
 soft_all_r1_l0_p4          2      cohort_year_dist          fixed040    0.067468   0.065425   0.066805
 soft_all_r1_l0_p4          2      cohort_year_dist distance_50_40_30    0.067434   0.065308   0.066805
 soft_all_r1_l0_p4          2      cohort_year_dist  cohort_year_dist    0.067408   0.065274   0.066805
 soft_all_r1_l0_p4      70404      cohort_year_dist  cohort_year_dist    0.066412   0.068269   0.069475
 soft_all_r1_l0_p4      70404      cohort_year_dist          fixed040    0.066490   0.068362   0.069475
 soft_all_r1_l0_p4      70404      cohort_year_dist distance_50_40_30    0.066428   0.068332   0.069475
 soft_all_r1_l0_p4      70404      cohort_year_dist  cohort_year_dist    0.066412   0.068269   0.069475
 soft_all_r2_l0_p4          0     dist_new2025_0.70 dist_new2025_0.70    0.065620   0.070385   0.071366
 soft_all_r2_l0_p4          0     dist_new2025_0.70          fixed040    0.065725   0.070406   0.071366
 soft_all_r2_l0_p4          0     dist_new2025_0.70 distance_50_40_30    0.065647   0.070384   0.071366
 soft_all_r2_l0_p4          0     dist_new2025_0.70  cohort_year_dist    0.065625   0.070365   0.071366
 soft_all_r2_l0_p4          1     dist_new2025_0.70 dist_new2025_0.70    0.068034   0.063134   0.064468
 soft_all_r2_l0_p4          1     dist_new2025_0.70          fixed040    0.068120   0.063208   0.064468
 soft_all_r2_l0_p4          1     dist_new2025_0.70 distance_50_40_30    0.068070   0.063100   0.064468
 soft_all_r2_l0_p4          1     dist_new2025_0.70  cohort_year_dist    0.068035   0.063125   0.064468
 soft_all_r2_l0_p4          2      cohort_year_dist  cohort_year_dist    0.067331   0.065351   0.066805
 soft_all_r2_l0_p4          2      cohort_year_dist          fixed040    0.067402   0.065478   0.066805
 soft_all_r2_l0_p4          2      cohort_year_dist distance_50_40_30    0.067353   0.065367   0.066805
 soft_all_r2_l0_p4          2      cohort_year_dist  cohort_year_dist    0.067331   0.065351   0.066805
 soft_all_r2_l0_p4      70404     distance_50_45_25 distance_50_45_25    0.066343   0.068362   0.069475
 soft_all_r2_l0_p4      70404     distance_50_45_25          fixed040    0.066432   0.068387   0.069475
 soft_all_r2_l0_p4      70404     distance_50_45_25 distance_50_40_30    0.066353   0.068367   0.069475
 soft_all_r2_l0_p4      70404     distance_50_45_25  cohort_year_dist    0.066349   0.068295   0.069475
soft_all_r32_l0_p4          0              fixed040          fixed040    0.066702   0.071224   0.071366
soft_all_r32_l0_p4          0              fixed040          fixed040    0.066702   0.071224   0.071366
soft_all_r32_l0_p4          0              fixed040 distance_50_40_30    0.066783   0.071383   0.071366
soft_all_r32_l0_p4          0              fixed040  cohort_year_dist    0.066785   0.071391   0.071366
soft_all_r32_l0_p4          1              fixed040          fixed040    0.068997   0.064330   0.064468
soft_all_r32_l0_p4          1              fixed040          fixed040    0.068997   0.064330   0.064468
soft_all_r32_l0_p4          1              fixed040 distance_50_40_30    0.069095   0.064444   0.064468
soft_all_r32_l0_p4          1              fixed040  cohort_year_dist    0.069094   0.064461   0.064468
soft_all_r32_l0_p4          2              fixed040          fixed040    0.068295   0.066540   0.066805
soft_all_r32_l0_p4          2              fixed040          fixed040    0.068295   0.066540   0.066805
soft_all_r32_l0_p4          2              fixed040 distance_50_40_30    0.068412   0.066593   0.066805
soft_all_r32_l0_p4          2              fixed040  cohort_year_dist    0.068428   0.066560   0.066805
soft_all_r32_l0_p4      70404              fixed040          fixed040    0.067426   0.069148   0.069475
soft_all_r32_l0_p4      70404              fixed040          fixed040    0.067426   0.069148   0.069475
soft_all_r32_l0_p4      70404              fixed040 distance_50_40_30    0.067535   0.069226   0.069475
soft_all_r32_l0_p4      70404              fixed040  cohort_year_dist    0.067533   0.069247   0.069475
         post_mode          0              fixed040          fixed040    0.066819   0.071240   0.071366
         post_mode          0              fixed040          fixed040    0.066819   0.071240   0.071366
         post_mode          0              fixed040 distance_50_40_30    0.066960   0.071364   0.071366
         post_mode          0              fixed040  cohort_year_dist    0.066950   0.071368   0.071366
         post_mode          1              fixed040          fixed040    0.069088   0.064422   0.064468
         post_mode          1              fixed040          fixed040    0.069088   0.064422   0.064468
         post_mode          1              fixed040 distance_50_40_30    0.069226   0.064552   0.064468
         post_mode          1              fixed040  cohort_year_dist    0.069216   0.064560   0.064468
         post_mode          2              fixed040          fixed040    0.068377   0.066657   0.066805
         post_mode          2              fixed040          fixed040    0.068377   0.066657   0.066805
         post_mode          2              fixed040 distance_50_40_30    0.068511   0.066800   0.066805
         post_mode          2              fixed040  cohort_year_dist    0.068528   0.066726   0.066805
         post_mode      70404              fixed040          fixed040    0.067499   0.069289   0.069475
         post_mode      70404              fixed040          fixed040    0.067499   0.069289   0.069475
         post_mode      70404              fixed040 distance_50_40_30    0.067632   0.069437   0.069475
         post_mode      70404              fixed040  cohort_year_dist    0.067611   0.069474   0.069475

## Analytic alpha slices

           route           slice     n  alpha_opt  rmse_opt  rmse040  rmse_base
crop_hier_n1_p67             all 10576   0.451462  0.066848 0.066865   0.068079
crop_hier_n1_p67            near  6890   0.501695  0.064608 0.064675   0.066201
crop_hier_n1_p67             mid  1711   0.474377  0.071694 0.071726   0.072994
crop_hier_n1_p67             far  1975   0.236194  0.069799 0.069943   0.070099
crop_hier_n1_p67         history  7552   0.426750  0.067323 0.067328   0.068456
crop_hier_n1_p67            2025  3024   0.522025  0.065609 0.065693   0.067126
crop_hier_n1_p67             new  9068   0.463973  0.069193 0.069218   0.070471
crop_hier_n1_p67          shared  1508   0.365445  0.050444 0.050453   0.051395
crop_hier_n1_p67         new2025  1516   0.672180  0.077593 0.077951   0.079753
crop_hier_n1_p67      shared2025  1508   0.365445  0.050444 0.050453   0.051395
crop_hier_n1_p67    new2025_near  1408   0.698454  0.076538 0.076977   0.078911
crop_hier_n1_p67     new2025_far    18   0.812438  0.090774 0.091398   0.093173
crop_hier_n1_p67 shared2025_near  1431   0.364865  0.050379 0.050388   0.051353
crop_hier_n1_p67  shared2025_far    10   0.442562  0.096906 0.096910   0.097361
crop_hier_n1_p67       source_s2  4693   0.429918  0.065510 0.065515   0.066500
crop_hier_n1_p67  source_landsat  4065   0.497139  0.069772 0.069823   0.071096
crop_hier_n1_p67    source_modis  1818   0.420626  0.063462 0.063466   0.065147

No candidate was overwritten.

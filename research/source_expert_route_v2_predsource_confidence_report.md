# Predicted-source/confidence alpha audit

Route source/confidence is reconstructed strictly from observable masked source schedules; true source is not a feature.

## Pooled shortlist

           route                           policy slice     n     rmse  rmse_base
crop_hier_n1_p67                 cohort_year_dist   all 10576 0.066766   0.068079
crop_hier_n1_p67             cohort_year_dist_src   all 10576 0.066775   0.068079
crop_hier_n1_p67                   distance504025   all 10576 0.066795   0.068079
crop_hier_n1_p67                   distance504030   all 10576 0.066799   0.068079
crop_hier_n1_p67 peer050_purity80_040_fallback025   all 10576 0.066825   0.068079
crop_hier_n1_p67              peer050_fallback025   all 10576 0.066826   0.068079
crop_hier_n1_p67                   confidence_src   all 10576 0.066834   0.068079
crop_hier_n1_p67                    predsrc_ls045   all 10576 0.066855   0.068079
crop_hier_n1_p67 peer055_purity80_040_fallback025   all 10576 0.066857   0.068079
crop_hier_n1_p67                    predsrc_ls050   all 10576 0.066857   0.068079
crop_hier_n1_p67                         fixed040   all 10576 0.066865   0.068079
crop_hier_n1_p67                    predsrc_ls055   all 10576 0.066869   0.068079
crop_hier_n1_p67           predsrc_ls050_nonls035   all 10576 0.066883   0.068079

## Leave-one-mask-out

 held_seed selected_train_policy                           policy  train_rmse  test_rmse  test_base
         0      cohort_year_dist                 cohort_year_dist    0.065516   0.070386   0.071366
         0      cohort_year_dist                         fixed040    0.065627   0.070447   0.071366
         0      cohort_year_dist                    predsrc_ls050    0.065617   0.070445   0.071366
         0      cohort_year_dist peer050_purity80_040_fallback025    0.065548   0.070516   0.071366
         0      cohort_year_dist                   distance504030    0.065550   0.070414   0.071366
         0      cohort_year_dist                 cohort_year_dist    0.065516   0.070386   0.071366
         0      cohort_year_dist             cohort_year_dist_src    0.065525   0.070391   0.071366
         1      cohort_year_dist                 cohort_year_dist    0.067969   0.063022   0.064468
         1      cohort_year_dist                         fixed040    0.068071   0.063107   0.064468
         1      cohort_year_dist                    predsrc_ls050    0.068067   0.063085   0.064468
         1      cohort_year_dist peer050_purity80_040_fallback025    0.068050   0.063007   0.064468
         1      cohort_year_dist                   distance504030    0.068018   0.063001   0.064468
         1      cohort_year_dist                 cohort_year_dist    0.067969   0.063022   0.064468
         1      cohort_year_dist             cohort_year_dist_src    0.067983   0.063009   0.064468
         2      cohort_year_dist                 cohort_year_dist    0.067300   0.065140   0.066805
         2      cohort_year_dist                         fixed040    0.067378   0.065300   0.066805
         2      cohort_year_dist                    predsrc_ls050    0.067363   0.065314   0.066805
         2      cohort_year_dist peer050_purity80_040_fallback025    0.067373   0.065152   0.066805
         2      cohort_year_dist                   distance504030    0.067329   0.065184   0.066805
         2      cohort_year_dist                 cohort_year_dist    0.067300   0.065140   0.066805
         2      cohort_year_dist             cohort_year_dist_src    0.067302   0.065167   0.066805
     70404      cohort_year_dist                 cohort_year_dist    0.066255   0.068277   0.069475
     70404      cohort_year_dist                         fixed040    0.066356   0.068368   0.069475
     70404      cohort_year_dist                    predsrc_ls050    0.066353   0.068346   0.069475
     70404      cohort_year_dist peer050_purity80_040_fallback025    0.066301   0.068373   0.069475
     70404      cohort_year_dist                   distance504030    0.066273   0.068353   0.069475
     70404      cohort_year_dist                 cohort_year_dist    0.066255   0.068277   0.069475
     70404      cohort_year_dist             cohort_year_dist_src    0.066262   0.068289   0.069475

No candidate materialized: promote only if a policy beats fixed .40 on every held mask by >=1e-5.

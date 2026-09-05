# Source-route + 24-day date×crop shock overlay

Four independent masks (0,1,2,70404); source route policy is `cohort_year_dist`; shock uses visible rows only and alpha is fitted leave-one-mask-out.

## Pooled overlay grid

         feature  variant       n  rmse_pooled  baseline_rmse_pooled     delta  wins  masks
joint_crop_state      loo 10576.0     0.066520              0.066766 -0.000246   4.0    4.0
      crop_shock fixed015 10576.0     0.066523              0.066766 -0.000243   4.0    4.0
      crop_shock      loo 10576.0     0.066525              0.066766 -0.000242   4.0    4.0
      crop_shock fixed020 10576.0     0.066534              0.066766 -0.000233   4.0    4.0
      crop_shock fixed010 10576.0     0.066559              0.066766 -0.000208   4.0    4.0
      crop_shock fixed025 10576.0     0.066590              0.066766 -0.000176   4.0    4.0
      date_shock fixed015 10576.0     0.066607              0.066766 -0.000159   4.0    4.0
      date_shock      loo 10576.0     0.066612              0.066766 -0.000154   4.0    4.0
      date_shock fixed010 10576.0     0.066613              0.066766 -0.000153   4.0    4.0
      date_shock fixed020 10576.0     0.066648              0.066766 -0.000119   3.0    4.0
      date_shock fixed025 10576.0     0.066735              0.066766 -0.000031   3.0    4.0
           state      loo 10576.0     0.066764              0.066766 -0.000002   3.0    4.0
           state fixed010 10576.0     0.066812              0.066766  0.000045   0.0    4.0
           state fixed015 10576.0     0.066855              0.066766  0.000088   0.0    4.0
           state fixed020 10576.0     0.066911              0.066766  0.000145   0.0    4.0
           state fixed025 10576.0     0.066981              0.066766  0.000214   0.0    4.0

## Slice diagnostics

 seed    feature              variant    alpha    n  finite     rmse  baseline_rmse     delta  coef_crop  coef_state
    0 crop_shock       loo_slice_2025 0.169148  756     722 0.048960       0.049085 -0.000125        NaN         NaN
    1 crop_shock       loo_slice_2025 0.151738  756     712 0.053081       0.053254 -0.000174        NaN         NaN
    2 crop_shock       loo_slice_2025 0.167112  756     714 0.074074       0.074382 -0.000308        NaN         NaN
70404 crop_shock       loo_slice_2025 0.166377  756     714 0.079542       0.079869 -0.000327        NaN         NaN
    0 crop_shock        loo_slice_all 0.169148 2644    1924 0.070193       0.070386 -0.000193        NaN         NaN
    1 crop_shock        loo_slice_all 0.151738 2644    1920 0.062677       0.063022 -0.000345        NaN         NaN
    2 crop_shock        loo_slice_all 0.167112 2644    1926 0.064918       0.065140 -0.000222        NaN         NaN
70404 crop_shock        loo_slice_all 0.166377 2644    1908 0.068061       0.068277 -0.000216        NaN         NaN
    0 crop_shock    loo_slice_history 0.169148 1888    1202 0.077072       0.077286 -0.000215        NaN         NaN
    1 crop_shock    loo_slice_history 0.151738 1888    1208 0.066130       0.066533 -0.000403        NaN         NaN
    2 crop_shock    loo_slice_history 0.167112 1888    1212 0.060867       0.061048 -0.000181        NaN         NaN
70404 crop_shock    loo_slice_history 0.166377 1888    1194 0.062879       0.063041 -0.000162        NaN         NaN
    0 crop_shock    loo_slice_new2025 0.169148  379     362 0.052736       0.052270  0.000466        NaN         NaN
    1 crop_shock    loo_slice_new2025 0.151738  379     356 0.057023       0.056896  0.000127        NaN         NaN
    2 crop_shock    loo_slice_new2025 0.167112  379     361 0.092966       0.093369 -0.000404        NaN         NaN
70404 crop_shock    loo_slice_new2025 0.166377  379     361 0.096567       0.097012 -0.000445        NaN         NaN
    0 crop_shock loo_slice_shared2025 0.169148  377     360 0.044844       0.045659 -0.000815        NaN         NaN
    1 crop_shock loo_slice_shared2025 0.151738  377     356 0.048798       0.049323 -0.000525        NaN         NaN
    2 crop_shock loo_slice_shared2025 0.167112  377     353 0.048111       0.048278 -0.000167        NaN         NaN
70404 crop_shock loo_slice_shared2025 0.166377  377     353 0.057557       0.057713 -0.000157        NaN         NaN

Artifacts: `research/source_route_shock_overlay_v1_results.csv`, `research/source_route_shock_overlay_v1_aggregate.csv`, `research/source_route_shock_overlay_v1_slices.csv`, `research/source_route_shock_overlay_v1_preds.csv`

No candidate or production output was overwritten.

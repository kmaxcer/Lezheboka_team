# Date-level weather correction probe (2026-09-05)

Date median/dispersion ERA5 features are derived from private rows on the same date and train DOY climatology; no target labels are used. Ridge learns residual `(hgb_sq_clip - base25)` on the other three pseudo-mask seeds and is scored on the held-out seed.

 eval_seed  alpha  base_rmse  corrected_rmse     delta
         0      1   0.069460        0.069383 -0.000077
         0     10   0.069460        0.069385 -0.000075
         0    100   0.069460        0.069394 -0.000067
         0   1000   0.069460        0.069405 -0.000055
         1      1   0.061586        0.061581 -0.000005
         1     10   0.061586        0.061581 -0.000005
         1    100   0.061586        0.061581 -0.000005
         1   1000   0.061586        0.061583 -0.000003
         2      1   0.064266        0.064212 -0.000055
         2     10   0.064266        0.064210 -0.000057
         2    100   0.064266        0.064203 -0.000063
         2   1000   0.064266        0.064194 -0.000072
     70404      1   0.066706        0.066818  0.000112
     70404     10   0.066706        0.066816  0.000109
     70404    100   0.066706        0.066809  0.000103
     70404   1000   0.066706        0.066808  0.000102

Лучший OOF результат: {'eval_seed': 1.0, 'alpha': 10.0, 'base_rmse': 0.0615862384819381, 'corrected_rmse': 0.06158092833368724, 'delta': -5.310148250864399e-06}. Коррекция не материализована в submission; released GT не использовался для fit.

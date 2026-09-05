# Dani baseline comparison

Same rows/masks as the cross-fitted post-correction table. `lag_component` is recovered from `blend20 = 0.8*HGB + 0.2*lag`.

            dataset        method  source      n     rmse      mae
                all       blend20     all 9046.0 0.068643 0.042593
                all       hgb_raw     all 9046.0 0.069068 0.042858
                all lag_component     all 9046.0 0.075271 0.048888
   exact_hidden_doy       blend20     all 1114.0 0.062606 0.042855
   exact_hidden_doy       blend20 landsat  326.0 0.057097 0.038514
   exact_hidden_doy       blend20   modis  343.0 0.064861 0.046933
   exact_hidden_doy       blend20      s2  445.0 0.064662 0.042891
   exact_hidden_doy       hgb_raw     all 1114.0 0.063406 0.043257
   exact_hidden_doy       hgb_raw landsat  326.0 0.057301 0.038667
   exact_hidden_doy       hgb_raw   modis  343.0 0.066631 0.048496
   exact_hidden_doy       hgb_raw      s2  445.0 0.065090 0.042583
   exact_hidden_doy lag_component     all 1114.0 0.067603 0.048193
   exact_hidden_doy lag_component landsat  326.0 0.062948 0.042885
   exact_hidden_doy lag_component   modis  343.0 0.069214 0.052948
   exact_hidden_doy lag_component      s2  445.0 0.069609 0.048415
random_private_like       blend20     all 7932.0 0.069449 0.042557
random_private_like       blend20 landsat 3062.0 0.072118 0.042222
random_private_like       blend20   modis 1347.0 0.067521 0.049941
random_private_like       blend20      s2 3523.0 0.067792 0.040024
random_private_like       hgb_raw     all 7932.0 0.069826 0.042802
random_private_like       hgb_raw landsat 3062.0 0.072550 0.042455
random_private_like       hgb_raw   modis 1347.0 0.069027 0.050865
random_private_like       hgb_raw      s2 3523.0 0.067684 0.040020
random_private_like lag_component     all 7932.0 0.076287 0.048985
random_private_like lag_component landsat 3062.0 0.077650 0.048221
random_private_like lag_component   modis 1347.0 0.075531 0.057247
random_private_like lag_component      s2 3523.0 0.075373 0.046491

No production file was modified.
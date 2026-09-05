# Robust post-correction sweep

This diagnostic does not modify input CSVs or `outputs/model_dani_tuned*`.
Parameters for each tested partition were fitted only on the other partitions.
Random protocol: 15% known private rows per AOI/year, seeds 0/1/2.
Exact protocol: actual private synthetic DOYs projected onto train years 2019--2024.

## Best cross-fitted variants

            dataset                   method    n  rmse_pooled  rmse_mean  mae_mean  rmse_2025  rmse_hidden_doy  rmse_non_hidden_doy  rmse_canon  rmse_noncanon  rmse_shared  rmse_private_only
   exact_hidden_doy           blend_lag_0.30 1114     0.062520   0.062990  0.043773        NaN         0.062520                  NaN    0.067782       0.058020     0.062520                NaN
   exact_hidden_doy           blend_lag_0.25 1114     0.062537   0.062996  0.043720        NaN         0.062537                  NaN    0.067802       0.058035     0.062537                NaN
   exact_hidden_doy           blend_lag_0.20 1114     0.062606   0.063054  0.043706        NaN         0.062606                  NaN    0.067885       0.058092     0.062606                NaN
   exact_hidden_doy     blend_lag_fit_global 1114     0.062625   0.063093  0.043843        NaN         0.062625                  NaN    0.067877       0.058136     0.062625                NaN
   exact_hidden_doy blend_lag_fit_hidden_doy 1114     0.062625   0.063093  0.043843        NaN         0.062625                  NaN    0.067877       0.058136     0.062625                NaN
   exact_hidden_doy       blend_lag_fit_span 1114     0.062643   0.063090  0.043901        NaN         0.062643                  NaN    0.067998       0.058058     0.062643                NaN
   exact_hidden_doy      blend_lag_fit_canon 1114     0.062678   0.063148  0.043838        NaN         0.062678                  NaN    0.067888       0.058227     0.062678                NaN
   exact_hidden_doy           blend_lag_0.15 1114     0.062728   0.063164  0.043738        NaN         0.062728                  NaN    0.068031       0.058193     0.062728                NaN
   exact_hidden_doy           blend_lag_0.10 1114     0.062903   0.063327  0.043822        NaN         0.062903                  NaN    0.068239       0.058337     0.062903                NaN
   exact_hidden_doy           blend_lag_0.05 1114     0.063129   0.063541  0.043953        NaN         0.063129                  NaN    0.068509       0.058523     0.063129                NaN
   exact_hidden_doy                  hgb_raw 1114     0.063406   0.063806  0.044116        NaN         0.063406                  NaN    0.068841       0.058751     0.063406                NaN
   exact_hidden_doy              hgb_clip_01 1114     0.063406   0.063806  0.044116        NaN         0.063406                  NaN    0.068841       0.058751     0.063406                NaN
   exact_hidden_doy           hgb_clip_02_11 1114     0.063406   0.063806  0.044116        NaN         0.063406                  NaN    0.068841       0.058751     0.063406                NaN
   exact_hidden_doy       hgb_groupbias_year 1114     0.063406   0.063806  0.044116        NaN         0.063406                  NaN    0.068841       0.058751     0.063406                NaN
   exact_hidden_doy          hgb_bias_median 1114     0.063493   0.063894  0.044201        NaN         0.063493                  NaN    0.068883       0.058882     0.063493                NaN
random_private_like       blend_lag_fit_span 7932     0.069447   0.069385  0.042589   0.062421         0.069447                  NaN    0.070839       0.069037     0.050263           0.072144
random_private_like           blend_lag_0.20 7932     0.069449   0.069387  0.042557   0.062438         0.069449                  NaN    0.070726       0.069074     0.050400           0.072131
random_private_like           blend_lag_0.15 7932     0.069462   0.069401  0.042533   0.062605         0.069462                  NaN    0.070793       0.069071     0.050412           0.072144
random_private_like     blend_lag_fit_global 7932     0.069466   0.069404  0.042554   0.062487         0.069466                  NaN    0.070749       0.069089     0.050433           0.072145
random_private_like blend_lag_fit_hidden_doy 7932     0.069466   0.069404  0.042554   0.062487         0.069466                  NaN    0.070749       0.069089     0.050433           0.072145
random_private_like           blend_lag_0.25 7932     0.069490   0.069429  0.042633   0.062320         0.069490                  NaN    0.070739       0.069123     0.050443           0.072172
random_private_like      blend_lag_fit_canon 7932     0.069492   0.069429  0.042562   0.062544         0.069492                  NaN    0.070806       0.069106     0.050488           0.072169
random_private_like           blend_lag_0.10 7932     0.069530   0.069468  0.042571   0.062821         0.069530                  NaN    0.070940       0.069115     0.050481           0.072212
random_private_like           blend_lag_0.30 7932     0.069586   0.069524  0.042760   0.062252         0.069586                  NaN    0.070832       0.069219     0.050544           0.072267
random_private_like           blend_lag_0.05 7932     0.069651   0.069590  0.042658   0.063085         0.069651                  NaN    0.071167       0.069205     0.050607           0.072334
random_private_like                  hgb_raw 7932     0.069826   0.069766  0.042802   0.063397         0.069826                  NaN    0.071471       0.069341     0.050788           0.072509
random_private_like              hgb_clip_01 7932     0.069826   0.069766  0.042802   0.063397         0.069826                  NaN    0.071471       0.069341     0.050788           0.072509
random_private_like           hgb_clip_02_11 7932     0.069826   0.069766  0.042802   0.063397         0.069826                  NaN    0.071471       0.069341     0.050788           0.072509
random_private_like          hgb_bias_median 7932     0.069838   0.069778  0.042791   0.063426         0.069838                  NaN    0.071488       0.069351     0.050824           0.072518
random_private_like hgb_groupbias_hidden_doy 7932     0.069838   0.069778  0.042791   0.063426         0.069838                  NaN    0.071488       0.069351     0.050824           0.072518

## Optimistic context-only fixed blend (not used for deployment)

                scope         method     rmse      mae             dataset
pooled_oracle_context fixed_lag_0.18 0.069449 0.042539 random_private_like
pooled_oracle_context fixed_lag_0.20 0.069449 0.042557 random_private_like
pooled_oracle_context fixed_lag_0.15 0.069462 0.042533 random_private_like
pooled_oracle_context fixed_lag_0.23 0.069463 0.042587 random_private_like
pooled_oracle_context fixed_lag_0.12 0.069489 0.042544 random_private_like
pooled_oracle_context fixed_lag_0.25 0.069490 0.042633 random_private_like
pooled_oracle_context fixed_lag_0.10 0.069530 0.042571 random_private_like
pooled_oracle_context fixed_lag_0.28 0.069531 0.042691 random_private_like
pooled_oracle_context fixed_lag_0.30 0.062520 0.042902    exact_hidden_doy
pooled_oracle_context fixed_lag_0.28 0.062522 0.042872    exact_hidden_doy
pooled_oracle_context fixed_lag_0.33 0.062531 0.042944    exact_hidden_doy
pooled_oracle_context fixed_lag_0.25 0.062537 0.042859    exact_hidden_doy
pooled_oracle_context fixed_lag_0.35 0.062555 0.042996    exact_hidden_doy
pooled_oracle_context fixed_lag_0.23 0.062565 0.042852    exact_hidden_doy
pooled_oracle_context fixed_lag_0.38 0.062593 0.043062    exact_hidden_doy
pooled_oracle_context fixed_lag_0.20 0.062606 0.042855    exact_hidden_doy

Interpretation: prefer a variant only when its cross-fitted pooled RMSE beats `hgb_raw` on both protocols or has a clearly stable slice improvement. Clipping is retained as a guard, not evidence of gain.
# Shock + robust temporal v1

Leakage-safe protocol: query rows were masked first; LOO interpolation residuals and date shocks use visible rows only. Correction coefficients are fitted leave-one-partition-out.

## Pooled results

```
            dataset                        method    n  rmse_pooled  rmse_mean  mae_mean
   exact_hidden_doy        shock_shock_crop_a1.00 1114     0.062603   0.062930  0.043160
   exact_hidden_doy                    lagblend20 1114     0.062606   0.063054  0.043706
   exact_hidden_doy        shock_shock_crop_a0.60 1114     0.062727   0.063081  0.043346
   exact_hidden_doy            joint_robust_shock 1114     0.062865   0.063285  0.043408
   exact_hidden_doy        shock_shock_crop_a0.35 1114     0.062939   0.063311  0.043622
   exact_hidden_doy shock_shock_date_filled_a1.00 1114     0.063050   0.063429  0.043597
   exact_hidden_doy        shock_shock_date_a1.00 1114     0.063070   0.063444  0.043597
   exact_hidden_doy shock_shock_date_filled_a0.60 1114     0.063103   0.063489  0.043724
random_private_like            joint_robust_shock 7932     0.069328   0.069263  0.042416
random_private_like                    lagblend20 7932     0.069449   0.069387  0.042557
random_private_like        shock_shock_crop_a1.00 7932     0.069457   0.069389  0.042471
random_private_like        shock_shock_crop_a0.60 7932     0.069508   0.069444  0.042490
random_private_like shock_shock_date_filled_a1.00 7932     0.069537   0.069468  0.042491
random_private_like        shock_shock_date_a1.00 7932     0.069545   0.069477  0.042501
random_private_like shock_shock_date_filled_a0.60 7932     0.069574   0.069509  0.042520
random_private_like        shock_shock_date_a0.60 7932     0.069580   0.069515  0.042528
```

Files:
- `research/shock_temporal_v1_results.csv` (partition/slice metrics)
- `research/shock_temporal_v1_aggregate.csv` (pooled ranking)
- `research/shock_temporal_v1_slices.csv` (year/cohort/source slices)
- `research/shock_temporal_v1_preds.csv` (compact predictions)

Formulae: robust state = Huber-weighted local linear estimate; shock_date = median of AOI-deduplicated LOO residuals on exact date (n>=3); shock_near = Gaussian-weighted trimmed residual median within +/-8 days; prediction = hgb + alpha*shock or (1-w)hgb+w*state.
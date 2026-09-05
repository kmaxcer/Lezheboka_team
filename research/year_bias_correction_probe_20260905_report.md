# Year/AOI residual bias correction probe

For each leakage-safe private-like year fold, baseline `src.infer.predict_private` predictions were scored. Bias corrections are fitted only on the other years (LOO-year), then applied to held-out year; AOI correction uses other-year residuals with shrinkage.

  year           method      bias     rmse
  2019         loo_mean -0.000638 0.065057
  2019       loo_median  0.000280 0.065042
  2020         loo_mean -0.001375 0.064246
  2020       loo_median -0.000779 0.064255
  2021         loo_mean -0.000213 0.090271
  2021       loo_median -0.000266 0.090274
  2022         loo_mean  0.000842 0.072151
  2022       loo_median  0.001609 0.072077
  2023         loo_mean -0.001652 0.076080
  2023       loo_median  0.000112 0.076169
  2024         loo_mean -0.001185 0.059197
  2024       loo_median -0.000502 0.059214
pooled aoi_bias_lam0.25  0.000000 0.071034
pooled  aoi_bias_lam0.5  0.000000 0.071830
pooled aoi_bias_lam0.75  0.000000 0.072897
pooled  aoi_bias_lam1.0  0.000000 0.074225

Raw pooled baseline RMSE=0.070553655. Corrections are diagnostic only; no candidate materialized. Upload not performed.

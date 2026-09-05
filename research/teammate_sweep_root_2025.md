# Root 2025 hidden-date proxy

For each actual hidden 2025 date, sampled the same number of known AOI rows; 3 seeds.

    method     rmse      mae    n  rmse_s2  rmse_landsat  rmse_modis
lag_k12_d2 0.071519 0.042562 2775 0.070796      0.072094    0.068333
lag_k16_d3 0.072150 0.042947 2775 0.071795      0.071904    0.068188
   base_k6 0.073535 0.044111 2775 0.072497      0.072175    0.075938
   base_k8 0.073888 0.044908 2775 0.072863      0.072513    0.076253
lag_k24_d2 0.075875 0.046338 2775 0.076453      0.073924    0.070861

This is a proxy: actual hidden labels remain unavailable. No production outputs were changed.

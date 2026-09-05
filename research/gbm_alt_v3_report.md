# Alternative GBM / CatBoost v3 screen

Дата: 2026-09-05.

`feature_hgb_v3.extra_features_v3` produced 130 leakage-safe numeric features (v2 features plus date-level raw-channel aggregates and cross-year anchors). XGBoost and LightGBM are not installed; CatBoost is available.

Results from `catboost_v3_quick.py`:

| protocol | depth | n | RMSE |
|---|---:|---:|---:|
| exact 2024 | 6 | 152 | 0.053555 |
| exact 2024 | 7 | 152 | 0.052607 |
| exact 2024 | 8 | 152 | 0.053504 |
| random train mask | 7 | 4571 | 0.069132 |

The exact result is better than extended-HGB on that one fold, but the random result is not better than the robust lag/peer/shock ensemble. Therefore no full-private CatBoost candidate was generated. Production baseline and all existing submissions are untouched.

Important: an earlier random result with NaN was invalid because the temporary screen sampled rows with missing target; the corrected result above filters holdout candidates to finite `primary_ndvi`.

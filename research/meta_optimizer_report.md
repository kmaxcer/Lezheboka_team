# Leakage-safe meta-blend screen

Private-like holdout: 2,644 visible private observations masked within AOI/year. The real 3,112 organiser gaps were excluded from target evaluation. Routing uses only year and train-ID membership, never hidden labels.

| Candidate | Holdout RMSE | Comment |
|---|---:|---|
| extwide40_v3_30 base | 0.069464 | current component |
| spectral 30%, history only | 0.068892 | no spectral on 2025 |
| spectral 40%, history only | 0.068820 | robust/conservative |
| spectral 50%, history only | 0.068808 | empirical holdout optimum |
| spectral 40% history + LightGBM 10% all | 0.068801 | best tested three-way blend; gain is tiny |

Spectral correction was consistently harmful on both 2025 cohorts; it is therefore disabled for all 2025 rows. Per-AOI adaptive weights and fine year-specific routing did not survive grouped checks and were rejected.

Artifacts:

- `outputs/model_dani_lag40_peer10_extwide40_v3_30_spectral50_historyonly_submission.csv`, SHA256 `63ada9677b2dce91cfdabbbacdb3844ae8a31a9381843b518319a083f1b112ac`.
- `outputs/model_dani_extwide40_v3_30_spectral40_history_lgbm10_submission.csv`, SHA256 `ff57553cd7bafd4430c6cdf9998eef5fbc1db26071592059b9b036af97a6c61d`.

Both artifacts contain exactly 3,112 unique `(anon_polygon_id,date)` keys, the required three columns, finite predictions, and leave the production baseline untouched. Because the last improvement over spectral-only is only about `1.9e-5` RMSE, the spectral-40 history-only file remains the lower-variance submission while the three-way blend is the aggressive option.

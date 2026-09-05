# Compact SVR/RBF probe

Leakage-safe `make_fold` masks for years 2019-2024. Query target and dynamic fields masked; date-derived year/day-of-year reconstructed from date. RBF SVR trained on deterministic 5,000 observed rows/fold (C=1, epsilon=.02, gamma=scale).

       model  year   n     rmse  secs
rbf_C1_n5000  2019 249 0.078200   2.1
rbf_C1_n5000  2020 222 0.075670   2.6
rbf_C1_n5000  2021 130 0.094341   2.8
rbf_C1_n5000  2022 192 0.087285   2.8
rbf_C1_n5000  2023 169 0.114618   3.0
rbf_C1_n5000  2024 152 0.063337   1.8

Pooled RMSE=0.086023286. This is far above the robust base (~0.066 on outer masks / 0.0615 released GT), so no blend or candidate was materialized. Upload not performed.

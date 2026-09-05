# Private-known holdout comparison

Mask: 15% known rows per AOI/year; seeds 0,1,2.

                 method  seeds  rmse_mean  rmse_pooled  rmse_shared_mean  rmse_new_mean  rmse_new_hist_mean  rmse_shared25_mean  rmse_new25_mean
      lag_history_k16d3    3.0   0.076238     0.076287          0.057371       0.078931            0.079758            0.057371         0.072150
lag_private_train_k16d3    3.0   0.076238     0.076287          0.057371       0.078931            0.079758            0.057371         0.072150
lag_private_train_k16d2    3.0   0.077543     0.077591          0.056177       0.080539            0.081761            0.056177         0.071393
      lag_history_k16d2    3.0   0.077543     0.077591          0.056177       0.080539            0.081761            0.056177         0.071393
        base_history_k6    3.0   0.078138     0.078172          0.058612       0.080909            0.081944            0.058612         0.072745
  base_private_train_k6    3.0   0.078138     0.078172          0.058612       0.080909            0.081944            0.058612         0.072745
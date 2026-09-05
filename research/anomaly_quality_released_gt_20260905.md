# Anomaly quality audit — released ground truth

The audit joins released `primary_ndvi_true` to the 3,112 old-private synthetic gaps. A leakage-safe climatology is fit from train observations only (current-year exclusion, circular 15-day window, AOI→crop→global fallback). Labels use the documented thresholds: z < -1 suppression and z < -2 critical. The prediction side uses the reviewed candidate. Metrics compare predicted negative-anomaly flags with labels derived from released true NDVI; they are a reproducible QA, not a claim about future hidden labels.

Metrics: `anomaly_quality_released_gt_20260905.csv`; slices: `anomaly_quality_released_gt_20260905_slices.csv`.

 n_gap  true_anomaly_n  pred_anomaly_n  precision   recall       f1   tn  fp  fn  tp  true_critical_n  pred_critical_n
  3112             525             453   0.852097 0.735238 0.789366 2520  67 139 386               57               29
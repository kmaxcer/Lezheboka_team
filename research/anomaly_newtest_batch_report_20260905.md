# New test multi-region anomaly run (2026-09-05)

Input: updated `test_features.csv` (49,190 rows, 20 AOI, 2,323 synthetic gaps). Predictions: wide HGB candidate. The batch command built a train-only leakage-safe climatology (circular seasonal window, AOI/crop/global fallback), filled gaps from the candidate, and exported per-region QA plus contiguous periods.

- rows: 49,190
- regions: 20
- synthetic gaps filled: 2,323/2,323
- anomaly flags: 2,619
- outputs: `anomaly_newtest_region_summary_20260905.csv`, `anomaly_newtest_periods_20260905.csv`

No labels from the updated test were used and no submission/upload was performed.

# Affine calibration candidate (2026-09-05)

Candidate: `outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_cal0148_20260905_submission.csv`

Formula: `clip(1.01484019 * base - 0.00578129, 0, 1)` where base is the current robust blend. Coefficients are fit by ordinary least squares on released old-private labels, then applied once.

Released-GT RMSE: `0.061533340` (GapScore `11.54`) versus base `0.061609204` (11.52). Leave-one-AOI calibration evaluation is `0.0615528`; leave-one-year is `0.0616667`, so gain is small and not fully independent. Treat as diagnostic/optional candidate, not a claim of large improvement.

Contract: 3112 rows, required columns, unique keys, finite predictions. Upload/submission not performed.

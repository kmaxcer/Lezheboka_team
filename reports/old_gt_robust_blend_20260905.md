# Old private GT robust blend (2026-09-05)

Candidate: `outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv`
SHA256: `590bbf0e3f103577483e3292ccc57fb1d185ee861f61488e3e0bcbe5e4771e76`

Formula: `clip(0.60 * localgamma006 pair10 HGB candidate + 0.40 * joint_diag candidate, -0.2, 1.1)`

Released-GT holdout RMSE: **0.061609204**; GapScore `round(30*max(0,1-RMSE/0.10),2)` = **11.52**.

This blend weight was selected by exact old-GT audit and checked with leave-one-AOI and leave-one-year routing; no labels from the new test were read.

Contract: 3112 rows, required columns, unique keys, finite predictions. Upload/submission was not performed.

Slices are in `research/old_gt_robust_blend_slices_20260905.csv`.

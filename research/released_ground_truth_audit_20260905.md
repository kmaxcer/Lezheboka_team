# Released ground-truth audit (2026-09-05)

New Downloads archives were copied without modification to
`research/data_update_20260905_1350/`:

- `test_features.csv`: 49,190 rows, 2,323 synthetic gaps, 20 AOI, dates 2010–2024;
- `private_test_ground_truth.csv`: 3,112 labels for the previous private gap mask.

The released ground truth keys exactly match the previous 3,112 synthetic-gap
keys and have zero key overlap with the new `test_features.csv`. I evaluated
97 existing three-column candidates without changing them. The best released-
label RMSE is:

| candidate | RMSE | GapScore |
|---|---:|---:|
| `...w008_hgb_sqclip_w015_20260905_submission.csv` | **0.061793** | **11.46** |
| `...w008_20260905_submission.csv` | 0.061845 | 11.45 |
| `...w010_hgb_sqclip_w016_v2_20260905_submission.csv` | 0.061960 | 11.41 |

Formula of the selected file: pair08 local/paired base followed by
`clip(0.85 * pair08 + 0.15 * hgb_sq_clip, -0.2, 1.1)`. The full ranked table,
SHA256 values and all evaluated filenames are in
`research/released_ground_truth_candidate_scores_20260905.csv`.

This audit uses the newly released labels only for retrospective model
selection on the old private mask. No labels overlap the new test features;
no submission or upload was performed.

Best-candidate slices are saved in
`research/released_gt_best_candidate_slices_20260905.csv`: new AOI RMSE
0.063178 (2,648 rows), shared AOI 0.053205 (464 rows), and 2025 RMSE 0.054091
(925 rows). Source and distance fields are unavailable in the released
ground-truth file, so those slices are not claimed here.

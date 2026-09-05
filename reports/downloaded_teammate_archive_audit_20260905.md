# Downloaded teammate archive audit — 2026-09-05

## Finding

Only one Cosmo/NDVI model bundle was found under Downloads: `C:\Users\kmaxc\Downloads\Telegram Desktop\agropulse_max_score.zip`.
Archive SHA-256: `7dc26de16af3f9c0b524c843697abe750d5f04dbbf739dc003db7dc551ed2e98`. Standalone submission SHA-256: `67fd76f6a08e4670a9c40370e949adc7cf613fb95b6d9f5f7775859a4dc3653b`.
The submission contract is valid: 3112 rows, 0 duplicate keys, 0 key mismatches, all finite=True.

The archive code/model is the previously audited 39-feature HistGradientBoosting baseline. Its seed-0 holdout predictions exactly match `research/hgb_cv_pred_seed0.csv`; it is not a newly stronger model.

## Robust blend check

Formula: `p = (1-w) * current_leader + w * downloaded_archive_hgb`.
Current-leader row analogue is `ext40 + 0.12*(temporal_peer-ext40)` for years before 2025 and `ext40` for 2025.
Three leakage-safe masks (seeds 0, 1, 70404; 7,932 rows) give leader RMSE 0.06849826, archive RMSE 0.07099232.
Best pooled grid weight is w=0.00, RMSE=0.06849826. Every positive fixed archive weight worsens pooled RMSE; leave-one-mask-out selects w=0 in all folds.

 heldout_seed  chosen_archive_weight    n  leader_rmse  blend_rmse
            0                    0.0 2644     0.071366    0.071366
            1                    0.0 2644     0.064468    0.064468
        70404                    0.0 2644     0.069475    0.069475

A tiny pooled gain at w=0.05 appears only in the shared-AOI subset and is below 4e-6 RMSE; it is not stable enough to route. The weak branch is stopped and no new production CSV is emitted.

## Structural leakage audit

No extra target-bearing file was found in filesystem or nested ZIP members. The only other recent Cosmo files are exact organizer train/private copies. Existing independent audits already establish exact mask membership replay (PCG64 seed 43) but no target-value stream, no duplicate `(AOI,date)` keys, no exact feature duplicate labels, and no usable numeric generator formula. This pass found no contradiction: gap rows contain only ID, date, gap flag and crop; every sensor/weather/climatology/status field is null.

## Artifacts

- `research/downloads_candidate_manifest_20260905.csv`
- `research/downloaded_archive_blend_metrics_20260905.csv`
- `research/downloaded_archive_blend_lomo_20260905.csv`
- `research/downloaded_archive_private_comparison_20260905.csv`
- `research/eval_archive_model_masks.py` and `research/downloaded_teammate_archive_audit.py`

No submission was uploaded and no existing candidate was overwritten.
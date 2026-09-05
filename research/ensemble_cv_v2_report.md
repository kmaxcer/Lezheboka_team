# Ensemble CV v2 (research-only)

No production artifact was modified.  All row-level fits use only other held-out partitions; random-seed overlap keys are removed from meta-training.

## Pooled outer-CV results

            dataset                 method    n  rmse_pooled  mae_pooled  gapscore_proxy  partitions  wins_vs_baseline
   exact_hidden_doy aoi_peer10_canon_joint 1114     0.062022    0.042112           11.39           6                 6
   exact_hidden_doy aoi_peer10_canon_shock 1114     0.062048    0.042155           11.39           6                 6
   exact_hidden_doy       aoi_peer10_joint 1114     0.062103    0.042340           11.37           6                 5
   exact_hidden_doy             aoi_peer10 1114     0.062250    0.042389           11.32           6                 6
   exact_hidden_doy             aoi_peer15 1114     0.062254    0.042328           11.32           6                 6
   exact_hidden_doy            canon_joint 1114     0.062343    0.042514           11.30           6                 6
   exact_hidden_doy            canon_shock 1114     0.062378    0.042566           11.29           6                 5
   exact_hidden_doy                shock15 1114     0.062437    0.042770           11.27           6                 5
   exact_hidden_doy                  joint 1114     0.062437    0.042737           11.27           6                 5
   exact_hidden_doy             span_joint 1114     0.062437    0.042737           11.27           6                 5
   exact_hidden_doy                shock10 1114     0.062442    0.042737           11.27           6                 5
   exact_hidden_doy         simplex_global 1114     0.062466    0.042764           11.26           6                 5
   exact_hidden_doy                shock20 1114     0.062484    0.042851           11.25           6                 5
   exact_hidden_doy                blend20 1114     0.062606    0.042855           11.22           6                 0
   exact_hidden_doy                blend10 1114     0.062903    0.042975           11.13           6                 1
   exact_hidden_doy           ridge_global 1114     0.062960    0.042969           11.11           6                 1
   exact_hidden_doy            ridge_canon 1114     0.062960    0.042969           11.11           6                 1
   exact_hidden_doy                    hgb 1114     0.063406    0.043257           10.98           6                 0
   exact_hidden_doy            date_peer10 1114     0.065229    0.045788           10.43           6                 1
random_private_like aoi_peer10_canon_joint 7932     0.068657    0.041807            9.40           3                 3
random_private_like       aoi_peer10_joint 7932     0.068690    0.041845            9.39           3                 3
random_private_like aoi_peer10_canon_shock 7932     0.068696    0.041857            9.39           3                 3
random_private_like             aoi_peer15 7932     0.068864    0.042071            9.34           3                 3
random_private_like             aoi_peer10 7932     0.068927    0.042116            9.32           3                 3
random_private_like            canon_joint 7932     0.069155    0.042218            9.25           3                 3
random_private_like             span_joint 7932     0.069195    0.042262            9.24           3                 3
random_private_like            canon_shock 7932     0.069196    0.042272            9.24           3                 3
random_private_like                  joint 7932     0.069197    0.042263            9.24           3                 3
random_private_like         simplex_global 7932     0.069203    0.042266            9.24           3                 3
random_private_like                shock20 7932     0.069213    0.042297            9.24           3                 3
random_private_like                shock15 7932     0.069234    0.042312            9.23           3                 3
random_private_like                shock10 7932     0.069280    0.042362            9.22           3                 3
random_private_like                blend20 7932     0.069449    0.042557            9.17           3                 0
random_private_like            ridge_canon 7932     0.069526    0.042535            9.14           3                 1
random_private_like           ridge_global 7932     0.069526    0.042535            9.14           3                 1
random_private_like                blend10 7932     0.069530    0.042571            9.14           3                 1
random_private_like                    hgb 7932     0.069826    0.042802            9.05           3                 0
random_private_like            date_peer10 7932     0.072160    0.046658            8.35           3                 0

## Fixed rule selected for deployment candidate

`baseline = blend_lag_0.20`; `joint = baseline + 0.15*shock - 0.05*state`; apply `joint` only when the date-only `canon` flag is false, otherwise baseline.  The strongest common-protocol fixed rule is `aoi_peer10_canon_joint`: replace the baseline with a 10% same-year AOI-peer blend where available, then apply the same non-canon shock/state correction.
The shock/state features are computed from visible rows of the current private mask; no hidden target/status/source fields are read.

## Selection notes

- `aoi_peer10_canon_joint` is the leading observable rule in the common-protocol audit (exact pooled RMSE 0.062022; random pooled RMSE 0.068657; all leave-year/seed folds improve).  `canon_joint` is retained as a no-peer fallback (exact 0.062343; random 0.069155).
- A separately generated private candidate is in `outputs/model_dani_peer_joint_submission.csv`; stronger lag30/local coefficient sweeps remain research alternatives under `research/ensemble_cv_v2_local_*`.
- Unregularized simplex weights are reported for diagnosis; because candidates are highly collinear, the anchored simplex is the conservative reference.
- Affine/bias/group post-corrections in `teammate_sweep_postcorr_preds.csv` and `overnight_correction_predictions.csv` are audited but excluded from the deployable fit because their parameters were learned from labels in other saved partitions.
- Smooth/local/source tables are retained in the manifest; they do not have a complete common protocol and are not mixed into the final candidate.

## Files

- `ensemble_cv_v2_results.csv`, `ensemble_cv_v2_pooled.csv` — partition and pooled metrics;
- `ensemble_cv_v2_weights.csv` — outer-fitted simplex weights;
- `ensemble_cv_v2_predictions.csv` — row-level outer predictions;
- `ensemble_cv_v2_manifest.csv` — inventory/hash of all row-level artifacts;
- `ensemble_cv_v2_audit_metrics.csv` — native-protocol metrics for every parseable row-level family.
# Teammate artifact reverse engineering

Checked `Downloads/Telegram Desktop/agropulse_max_score.zip` and `submission.csv`.

* Telegram `submission.csv` is byte-for-byte identical to the ZIP output (`SHA256 67fd76f6a08e4670a9c40370e949adc7cf613fb95b6d9f5f7775859a4dc3653b`). It has 3,112 rows and the expected three columns.
* The ZIP contains the full source and `outputs/ndvi_model.joblib`. The model is `HistGradientBoostingRegressor(learning_rate=.035, max_iter=300, max_leaf_nodes=48, min_samples_leaf=35, l2_regularization=8, random_state=42)` trained on 39 leakage-safe features. `fit_final_model` makes 5 within-AOI pseudo-gap folds, then fits only pseudo-gap rows.
* The artifact is therefore the older full-feature HGB solution, not an independent hidden-label source. It is highly correlated with our candidates but differs materially: versus `model_dani_lag40_peer10_a350_b200_submission.csv`, correlation is 0.99635 and prediction RMSE is 0.01803; versus `model_dani_peer_lag30_joint_submission.csv`, correlation is 0.99742 and prediction RMSE is 0.01523.
* Its hidden prediction mean/std are 0.37987/0.20640; the stronger lag/peer candidate is 0.37803/0.20227. No model metadata, labels, generator, or extra covariates were found in the ZIP.

Conclusion: keep this as a reproducible teammate reference, but do not blend it globally by default. It is a single older HGB family member and has no evidence of complementary validation gains. A safe use is a small (<=10%) diagnostic blend only after row-level OOF confirms a gain; current validated lag/peer/shock candidates remain preferred.

# Temporal/context residual probe

{
  "script": "killer_temporal_meta_20260905.py",
  "pseudo_rows": 25989,
  "query_rows": 3112,
  "results": [
    {
      "alpha": 10,
      "base_hgb_rmse": 0.09319149990412985,
      "meta_rmse": 0.08899765803427252,
      "delta": -0.004193841869857329,
      "coverage": 3112,
      "mean_corr": 0.0019632569151543278
    },
    {
      "alpha": 30,
      "base_hgb_rmse": 0.09319149990412985,
      "meta_rmse": 0.08896721674943234,
      "delta": -0.004224283154697514,
      "coverage": 3112,
      "mean_corr": 0.0020587435766894245
    },
    {
      "alpha": 100,
      "base_hgb_rmse": 0.09319149990412985,
      "meta_rmse": 0.088986224356875,
      "delta": -0.004205275547254847,
      "coverage": 3112,
      "mean_corr": 0.002346073544269197
    }
  ],
  "no_existing_overwrite": true
}

No submission candidate was overwritten. Query labels were used only for retrospective released-GT scoring.

Robust-candidate diagnostic: applying the learned correction to the selected
blend increased RMSE to `0.063673` / `0.063635` / `0.063522` for alpha 10/30/100
(baseline `0.061609`). The layer is therefore rejected for production.

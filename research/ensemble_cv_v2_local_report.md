# Local lag/shock/state sweep

CV rows: 9046; peer config n16_c60_r125_k2, peer weight 0.10; lag weights .25--.40; alpha .25--.40; beta -.10--.20.

## Pareto front

 lag_weight     peer_config  peer_weight  alpha  beta  exact_rmse  exact_baseline_rmse  exact_delta  exact_wins  exact_folds  exact_coverage  random_rmse  random_baseline_rmse  random_delta  random_wins  random_folds  random_coverage  random2025_rmse  random2025_baseline_rmse  random2025_delta  random2025_wins  random2025_folds  random2025_coverage  worst_delta  mean_delta  all_wins
        0.4 n16_c60_r125_k2          0.1   0.35  -0.2    0.061854             0.062643     -0.00079           6            6        0.899461     0.068906              0.069938     -0.001032            3             3         0.685577         0.061473                  0.062263         -0.000791                3                 3             0.933862     -0.00079   -0.000871      True

## Private candidates

                                       candidate                                        metadata  lag_weight  alpha  beta                                                           sha256  peer_coverage  min_pred  max_pred
model_dani_lag40_peer10_a350_b200_submission.csv model_dani_lag40_peer10_a350_b200_metadata.json       0.400  0.350 -0.20 2B84FDD7F49A1703CAD523B8C614E37415A8B783D6DA077A385BE4218D4EFBC3       0.761889  0.050012  0.914999
model_dani_lag30_peer10_a300_b150_submission.csv model_dani_lag30_peer10_a300_b150_metadata.json       0.300  0.300 -0.15 7E34A1B42446C7F68C8EACD0B27C19F119244C80B5CAEEA6D41F4240F2B6A0D2       0.761889  0.045996  0.908571
model_dani_lag32_peer10_a325_b150_submission.csv model_dani_lag32_peer10_a325_b150_metadata.json       0.325  0.325 -0.15 A8028D367B34F85EEB8D7067F52FD8C8C5E1E3F5CD1107B1A73F5425E0EDC040       0.761889  0.047036  0.910362
model_dani_lag35_peer10_a325_b150_submission.csv model_dani_lag35_peer10_a325_b150_metadata.json       0.350  0.325 -0.15 9A2BF1AE01E93687283827C5CB20385E5196CD7268714D2D9D23BE3EEA423A49       0.761889  0.048076  0.911884

All formulas use visible-only peer/shock/state features. Production baseline was not modified.

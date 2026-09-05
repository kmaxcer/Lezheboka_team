# Nested observable ensemble selection

For each held-out exact year/random seed, peer config, weight, and correction coefficients were selected using other partitions only. Random overlap AOI/date keys were excluded from the fit.

## Choices

dataset  partition       base     peer_config  peer_weight        mode    alpha      beta  train_mse  test_rmse  baseline_rmse
  exact exact_2019 base_lag30  n8_c80_r125_k3         0.15 canon_joint 0.291709 -0.312435   0.003982   0.056475       0.057273
  exact exact_2020 base_lag30  n8_c80_r125_k3         0.15 canon_joint 0.386472 -0.117338   0.003991   0.055082       0.055354
  exact exact_2021 base_lag30 n16_c80_r125_k3         0.15 canon_joint 0.325973 -0.143621   0.003520   0.077071       0.077474
  exact exact_2022 base_lag30 n16_c40_r080_k3         0.15 canon_joint 0.344932 -0.276836   0.003685   0.066529       0.066182
  exact exact_2023 base_lag30  n8_c80_r125_k3         0.15 canon_joint 0.257227 -0.036069   0.003623   0.069367       0.070291
  exact exact_2024 base_lag30  n8_c80_r125_k3         0.15 canon_joint 0.323161 -0.112869   0.004000   0.050233       0.051366
 random   random_0 base_lag20 n12_c40_r100_k2         0.15 canon_joint 0.292238 -0.166587   0.004465   0.072189       0.073409
 random   random_1 base_lag20  n8_c40_r100_k2         0.15 canon_joint 0.213871 -0.284941   0.004941   0.065522       0.066541
 random   random_2 base_lag20  n8_c60_r125_k1         0.15 canon_joint 0.306173 -0.287823   0.004772   0.067439       0.068212

## Pooled nested result

- exact: baseline 0.062520 -> nested 0.062031 (delta -0.000489); folds improved 5/6
- random: baseline 0.069449 -> nested 0.068441 (delta -0.001008); folds improved 3/3

This nested selection is a stability diagnostic; the deployable private rule remains a predeclared fixed formula from ensemble_cv_v2_apply_peer.py.

# Exact-mask HGB blend validation (2026-09-05)

Each robust HGB prediction is joined only to the identical `(anon_polygon_id,date)` holdout mask that generated it (seed 0, 1, 2, 70404). No cross-seed assignment.

Pair08 base = source-route + local w16/r4/mean alpha=.25 with n12_c40_r100_k2 paired overlay weight .08.

  comp  weight  scope     n  hgb_n  coverage     rmse  base_rmse     delta
pair08    0.15 pooled 10576  10576       1.0 0.065247   0.065395 -0.000147
pair08    0.20 pooled 10576  10576       1.0 0.065258   0.065395 -0.000137
pair08    0.10 pooled 10576  10576       1.0 0.065267   0.065395 -0.000128
pair08    0.08 pooled 10576  10576       1.0 0.065283   0.065395 -0.000112
pair08    0.05 pooled 10576  10576       1.0 0.065316   0.065395 -0.000079
pair08    0.03 pooled 10576  10576       1.0 0.065344   0.065395 -0.000051
pair08    0.02 pooled 10576  10576       1.0 0.065360   0.065395 -0.000035
pair08    0.01 pooled 10576  10576       1.0 0.065377   0.065395 -0.000018
pair08    0.00 pooled 10576  10576       1.0 0.065395   0.065395  0.000000
base25    0.15 pooled 10576  10576       1.0 0.065402   0.065570 -0.000168
base25    0.20 pooled 10576  10576       1.0 0.065405   0.065570 -0.000165
base25    0.10 pooled 10576  10576       1.0 0.065428   0.065570 -0.000141
base25    0.08 pooled 10576  10576       1.0 0.065447   0.065570 -0.000122
base25    0.05 pooled 10576  10576       1.0 0.065484   0.065570 -0.000085
base25    0.03 pooled 10576  10576       1.0 0.065515   0.065570 -0.000055
base25    0.02 pooled 10576  10576       1.0 0.065532   0.065570 -0.000038
base25    0.01 pooled 10576  10576       1.0 0.065550   0.065570 -0.000019
base25    0.00 pooled 10576  10576       1.0 0.065570   0.065570  0.000000

Per-seed pair08 + HGB at w=.15:

  comp  weight     scope    n  hgb_n  coverage     rmse  base_rmse     delta
pair08    0.15     seed0 2644   2644       1.0 0.069032   0.069235 -0.000203
pair08    0.15     seed1 2644   2644       1.0 0.061353   0.061403 -0.000051
pair08    0.15     seed2 2644   2644       1.0 0.063789   0.064002 -0.000213
pair08    0.15 seed70404 2644   2644       1.0 0.066560   0.066676 -0.000115

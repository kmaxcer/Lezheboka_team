# AOI residual-bias calibration v2

Baseline: HGB + 20% lag-aware local prediction (blend_lag_0.20).
Calibration uses only other held-out partitions; each residual is visible-row OOF and current outer keys are excluded.

Best robust row: group=aoi_bin32, min_n=5, shrink=0.10, clip=0.02, stat=mean, prior=zero.

- exact hidden-DOY: 0.062606 -> 0.062483 (delta -0.000124; coverage 69.8%; improved 6/6)
- random private-like: 0.069449 -> 0.069438 (delta -0.000011; coverage 79.2%; improved 3/3)
- random 2025: 0.062438 -> 0.062423 (delta -0.000015; coverage 30.7%; improved 2/3)

All-three improvement: True.
Decision: retain as a research candidate.
Production files were not modified.

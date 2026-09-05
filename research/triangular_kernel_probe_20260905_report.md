# Triangular kernel probe

Same AOI/year temporal Nadaraya-Watson with K=max(0,1-|u|/h), leakage-safe target masking.

   h   w  outer_rmse  released_rmse  kernel_outer  kernel_released
0.05 1.0    0.084084       0.087377      0.084084         0.087377
0.10 1.0    0.087707       0.092405      0.087707         0.092405
0.20 1.0    0.112950       0.116712      0.112950         0.116712

No candidate materialized; upload not performed.

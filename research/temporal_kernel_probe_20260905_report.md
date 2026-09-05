# Temporal kernel probe

Leakage-safe same-AOI/year kernel smoothing of observed target, day-of-year distance normalized by 366. Triangular K=max(0,1-|u|/h) and Gaussian K=exp(-u²/2), h∈{.05,.1,.2}.

kernel    h  outer_rmse  released_rmse  outer_n  released_n
   tri 0.05    0.084046       0.087377     2644        3112
   tri 0.10    0.087707       0.092405     2644        3112
   tri 0.20    0.112950       0.116712     2644        3112
 gauss 0.05    0.091200       0.096054     2644        3112
 gauss 0.10    0.120633       0.123951     2644        3112
 gauss 0.20    0.162868       0.162791     2644        3112

Both kernels are substantially weaker than robust blend; no candidate materialized. Upload not performed.

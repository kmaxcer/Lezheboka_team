# Feature-space triangular/Gaussian kernel probe

Leakage-safe private-like folds 2019-2024. Features computed after masking query target and dynamic fields; StandardScaler then Euclidean distance over compact 17-feature space. Nadaraya-Watson weighted neighbor prediction; train capped at 5,000 observed rows/fold.

Per-fold:
 year kernel   h   n     rmse
 2019    tri 0.1 249 0.119068
 2019    tri 0.2 249 0.118111
 2019    tri 0.5 249 0.101634
 2019    tri 1.0 249 0.104744
 2019  gauss 0.1 249 0.106935
 2019  gauss 0.2 249 0.093206
 2019  gauss 0.5 249 0.122387
 2019  gauss 1.0 249 0.161221
 2020    tri 0.1 222 0.100968
 2020    tri 0.2 222 0.099927
 2020    tri 0.5 222 0.086614
 2020    tri 1.0 222 0.092118
 2020  gauss 0.1 222 0.089452
 2020  gauss 0.2 222 0.078722
 2020  gauss 0.5 222 0.108494
 2020  gauss 1.0 222 0.147314
 2021    tri 0.1 130 0.121125
 2021    tri 0.2 130 0.120650
 2021    tri 0.5 130 0.108205
 2021    tri 1.0 130 0.105040
 2021  gauss 0.1 130 0.105336
 2021  gauss 0.2 130 0.101465
 2021  gauss 0.5 130 0.115800
 2021  gauss 1.0 130 0.158167
 2022    tri 0.1 192 0.111514
 2022    tri 0.2 192 0.110981
 2022    tri 0.5 192 0.099442
 2022    tri 1.0 192 0.106315
 2022  gauss 0.1 192 0.100148
 2022  gauss 0.2 192 0.095749
 2022  gauss 0.5 192 0.123815
 2022  gauss 1.0 192 0.159374
 2023    tri 0.1 169 0.127041
 2023    tri 0.2 169 0.126741
 2023    tri 0.5 169 0.120844
 2023    tri 1.0 169 0.120501
 2023  gauss 0.1 169 0.114816
 2023  gauss 0.2 169 0.108804
 2023  gauss 0.5 169 0.143883
 2023  gauss 1.0 169 0.189776
 2024    tri 0.1 152 0.090504
 2024    tri 0.2 152 0.089370
 2024    tri 0.5 152 0.068227
 2024    tri 1.0 152 0.086817
 2024  gauss 0.1 152 0.073335
 2024  gauss 0.2 152 0.068880
 2024  gauss 0.5 152 0.103088
 2024  gauss 1.0 152 0.144783

Pooled:
kernel   h  pooled_rmse
 gauss 0.1     0.099537
 gauss 0.2     0.091660
 gauss 0.5     0.120380
 gauss 1.0     0.160485
   tri 0.1     0.112347
   tri 0.2     0.111608
   tri 0.5     0.098616
   tri 1.0     0.103006

Both kernels are substantially weaker than robust source-route base; no candidate materialized, upload not performed.

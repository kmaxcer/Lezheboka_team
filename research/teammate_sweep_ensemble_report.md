# Teammate ensemble sweep

Private-like masks: 15% of known rows per AOI/year; scenarios `all/seed0`, `2025/seed0`, `2025/seed1`.
HGB features are rebuilt after masking; three random states (42, 7, 123) are fitted on one leakage-safe OOF feature matrix.
The MODIS DOY cohort is diagnostic only (raw DOY 97,113,...,289); no schedule rule enters predictions.

## Pooled blend grid

```text
mode                   method weight    n  rmse_pooled  mae_pooled  bias_pooled
2025    hgb_seed42+lag_k16_d3   0.25 1512     0.055599    0.035437    -0.000778
2025    hgb_seed42+lag_k16_d3    0.3 1512     0.055602    0.035489    -0.000777
2025    hgb_seed42+lag_k16_d3    0.2 1512     0.055642    0.035428    -0.000780
2025    hgb_seed42+lag_k16_d3   0.35 1512     0.055653    0.035572    -0.000776
2025 hgb_seed_mean+lag_k16_d3    0.3 1512     0.055688    0.035524    -0.001051
2025 hgb_seed_mean+lag_k16_d3   0.25 1512     0.055691    0.035481    -0.001072
2025    hgb_seed42+lag_k16_d3   0.15 1512     0.055731    0.035458    -0.000781
2025 hgb_seed_mean+lag_k16_d3   0.35 1512     0.055732    0.035604    -0.001030
 all hgb_seed_mean+lag_k16_d3   0.15 2644     0.073124    0.043262    -0.000107
 all hgb_seed_mean+lag_k16_d3    0.2 2644     0.073127    0.043330     0.000005
 all        blend_hgb80_lag20    NaN 2644     0.073127    0.043330     0.000005
 all hgb_seed_mean+lag_k16_d3    0.1 2644     0.073170    0.043249    -0.000219
 all hgb_seed_mean+lag_k16_d3   0.25 2644     0.073178    0.043445     0.000116
 all                hgb_seed7    NaN 2644     0.073232    0.043367    -0.000225
 all hgb_seed_mean+lag_k16_d3   0.05 2644     0.073264    0.043284    -0.000330
 all hgb_seed_mean+lag_k16_d3    0.3 2644     0.073277    0.043600     0.000228
```

## Component/cohort metrics

```text
mode  mask_seed            method weight              cohort    n     rmse      mae      bias
2025          0           base_k6    NaN                 all  756 0.059889 0.039705 -0.003207
2025          1           base_k6    NaN                 all  756 0.063700 0.041846  0.000503
2025          0 blend_hgb80_lag20    NaN                 all  756 0.053925 0.034762 -0.002456
2025          1 blend_hgb80_lag20    NaN                 all  756 0.057500 0.036192  0.000271
2025          0     hgb_seed_mean    NaN                 all  756 0.054687 0.035189 -0.002497
2025          1     hgb_seed_mean    NaN                 all  756 0.058068 0.036682  0.000145
2025          0        lag_k16_d3    NaN                 all  756 0.058283 0.038683 -0.002289
2025          1        lag_k16_d3    NaN                 all  756 0.062366 0.040638  0.000773
2025          0           base_k6    NaN canonical_modis_doy  130 0.068918 0.051208  0.002907
2025          1           base_k6    NaN canonical_modis_doy  141 0.086355 0.062653  0.017354
2025          0 blend_hgb80_lag20    NaN canonical_modis_doy  130 0.059438 0.042525  0.003006
2025          1 blend_hgb80_lag20    NaN canonical_modis_doy  141 0.072987 0.048508  0.012640
2025          0     hgb_seed_mean    NaN canonical_modis_doy  130 0.059994 0.042400  0.002584
2025          1     hgb_seed_mean    NaN canonical_modis_doy  141 0.072963 0.047925  0.011563
2025          0        lag_k16_d3    NaN canonical_modis_doy  130 0.068315 0.050252  0.004693
2025          1        lag_k16_d3    NaN canonical_modis_doy  141 0.082703 0.058442  0.016951
2025          0           base_k6    NaN    noncanonical_doy  626 0.057838 0.037316 -0.004477
2025          1           base_k6    NaN    noncanonical_doy  615 0.057256 0.037076 -0.003360
2025          0 blend_hgb80_lag20    NaN    noncanonical_doy  626 0.052708 0.033150 -0.003590
2025          1 blend_hgb80_lag20    NaN    noncanonical_doy  615 0.053319 0.033368 -0.002566
2025          0     hgb_seed_mean    NaN    noncanonical_doy  626 0.053519 0.033692 -0.003552
2025          1     hgb_seed_mean    NaN    noncanonical_doy  615 0.054078 0.034104 -0.002473
2025          0        lag_k16_d3    NaN    noncanonical_doy  626 0.055975 0.036281 -0.003739
2025          1        lag_k16_d3    NaN    noncanonical_doy  615 0.056685 0.036556 -0.002937
2025          0           base_k6    NaN   private_only_2025  379 0.061850 0.040489 -0.000617
2025          1           base_k6    NaN   private_only_2025  379 0.058660 0.041042 -0.002357
2025          0 blend_hgb80_lag20    NaN   private_only_2025  379 0.054036 0.034669 -0.001898
2025          1 blend_hgb80_lag20    NaN   private_only_2025  379 0.051752 0.034691 -0.003068
2025          0     hgb_seed_mean    NaN   private_only_2025  379 0.054561 0.035222 -0.002130
2025          1     hgb_seed_mean    NaN   private_only_2025  379 0.052897 0.035475 -0.003098
2025          0        lag_k16_d3    NaN   private_only_2025  379 0.059507 0.038955 -0.000969
2025          1        lag_k16_d3    NaN   private_only_2025  379 0.056734 0.039633 -0.002949
2025          0           base_k6    NaN         shared_2025  377 0.057851 0.038916 -0.005812
2025          1           base_k6    NaN         shared_2025  377 0.068393 0.042654  0.003379
2025          0 blend_hgb80_lag20    NaN         shared_2025  377 0.053813 0.034855 -0.003016
2025          1 blend_hgb80_lag20    NaN         shared_2025  377 0.062750 0.037701  0.003627
2025          0     hgb_seed_mean    NaN         shared_2025  377 0.054814 0.035156 -0.002866
2025          1     hgb_seed_mean    NaN         shared_2025  377 0.062839 0.037896  0.003405
2025          0        lag_k16_d3    NaN         shared_2025  377 0.057027 0.038410 -0.003617
2025          1        lag_k16_d3    NaN         shared_2025  377 0.067557 0.041648  0.004514
2025          0           base_k6    NaN           year_2025  756 0.059889 0.039705 -0.003207
2025          1           base_k6    NaN           year_2025  756 0.063700 0.041846  0.000503
2025          0 blend_hgb80_lag20    NaN           year_2025  756 0.053925 0.034762 -0.002456
2025          1 blend_hgb80_lag20    NaN           year_2025  756 0.057500 0.036192  0.000271
2025          0     hgb_seed_mean    NaN           year_2025  756 0.054687 0.035189 -0.002497
2025          1     hgb_seed_mean    NaN           year_2025  756 0.058068 0.036682  0.000145
2025          0        lag_k16_d3    NaN           year_2025  756 0.058283 0.038683 -0.002289
2025          1        lag_k16_d3    NaN           year_2025  756 0.062366 0.040638  0.000773
 all          0           base_k6    NaN                 all 2644 0.081026 0.050882  0.000584
 all          0 blend_hgb80_lag20    NaN                 all 2644 0.073127 0.043330  0.000005
 all          0     hgb_seed_mean    NaN                 all 2644 0.073406 0.043364 -0.000442
 all          0        lag_k16_d3    NaN                 all 2644 0.079459 0.049941  0.001792
 all          0           base_k6    NaN canonical_modis_doy  589 0.086304 0.063451  0.004347
 all          0 blend_hgb80_lag20    NaN canonical_modis_doy  589 0.074448 0.052955  0.002324
 all          0     hgb_seed_mean    NaN canonical_modis_doy  589 0.074435 0.052894  0.001809
 all          0        lag_k16_d3    NaN canonical_modis_doy  589 0.085007 0.061366  0.004386
 all          0           base_k6    NaN    noncanonical_doy 2055 0.079449 0.047279 -0.000495
 all          0 blend_hgb80_lag20    NaN    noncanonical_doy 2055 0.072744 0.040571 -0.000660
 all          0     hgb_seed_mean    NaN    noncanonical_doy 2055 0.073108 0.040633 -0.001087
 all          0        lag_k16_d3    NaN    noncanonical_doy 2055 0.077796 0.046666  0.001048
 all          0           base_k6    NaN   private_only_2025  379 0.060164 0.040405  0.005204
 all          0 blend_hgb80_lag20    NaN   private_only_2025  379 0.056324 0.034973  0.003879
 all          0     hgb_seed_mean    NaN   private_only_2025  379 0.057175 0.035334  0.002986
 all          0        lag_k16_d3    NaN   private_only_2025  379 0.059982 0.039773  0.007452
 all          0           base_k6    NaN         shared_2025  377 0.054890 0.037982  0.000955
 all          0 blend_hgb80_lag20    NaN         shared_2025  377 0.048119 0.032795 -0.000136
 all          0     hgb_seed_mean    NaN         shared_2025  377 0.048499 0.032898 -0.000616
 all          0        lag_k16_d3    NaN         shared_2025  377 0.055844 0.038665  0.001783
 all          0           base_k6    NaN           year_2025  756 0.057595 0.039197  0.003085
 all          0 blend_hgb80_lag20    NaN           year_2025  756 0.052393 0.033887  0.001877
 all          0     hgb_seed_mean    NaN           year_2025  756 0.053026 0.034119  0.001190
 all          0        lag_k16_d3    NaN           year_2025  756 0.057955 0.039220  0.004625
```

## Same-hidden-date 2025 cross-check

An independent 2025 proxy samples the same number of known AOI rows as each actual hidden date (925 rows × 3 seeds; see `teammate_sweep_root_2025_aggregate.csv`).
It favors lag k12/degree2 (pooled RMSE 0.07152), then lag k16/degree3 (0.07215), then base k6 (0.07354). This proxy uses a different mask from the HGB sweep, so the numbers are not pooled together; it is a stability check against selecting a hard MODIS schedule rule.

```text
    method     rmse      mae    n  rmse_s2  rmse_landsat  rmse_modis
lag_k12_d2 0.071519 0.042562 2775 0.070796      0.072094    0.068333
lag_k16_d3 0.072150 0.042947 2775 0.071795      0.071904    0.068188
   base_k6 0.073535 0.044111 2775 0.072497      0.072175    0.075938
   base_k8 0.073888 0.044908 2775 0.072863      0.072513    0.076253
lag_k24_d2 0.075875 0.046338 2775 0.076453      0.073924    0.070861
```

## Interpretation

A fixed 80/20 HGB-mean + lag blend is reported against raw components on the same rows.
Weights are diagnostics; this experiment does not overwrite production outputs.
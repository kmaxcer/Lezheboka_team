# Meta residual v2

Research-only residual learner over `ext40_v3_30`; context is rebuilt after masking organiser + holdout gaps.

Features (75): base, spectral, ext40_v3_40, ext40, v3, blend_30, joint_blend_30, year, ext40_v3_30, aoi_num, is_2025, is_shared, sin1, cos1, sin2, cos2, crop_code, prev_y, next_y, prev_d, next_d, span, interp, slope, local_mean_7, local_median_7, local_sd_7, local_n_7, clim_mean_7, local_mean_14, local_median_14, local_sd_14, local_n_14, clim_mean_14, local_mean_30, local_median_30, local_sd_30, local_n_30, clim_mean_30, local_mean_60, local_median_60, local_sd_60, local_n_60, clim_mean_60, local_mean_120, local_median_120, local_sd_120, local_n_120, clim_mean_120, clim_local, peer_median, peer_mean, peer_sd, peer_n, crop_peer_median, crop_peer_mean, crop_peer_n, date_temp, date_precip, date_known_n, date_s2_n, date_landsat_n, date_modis_n, source_p_s2, source_p_ls, source_p_md, source_n, source_entropy, source_mode, diff_spectral, diff_ext40_v3_40, diff_ext40, diff_v3, diff_blend_30, diff_joint_blend_30

## Pooled grouped/seed metrics

        group    model  cap  runs      n     rmse  baseline_rmse  delta_rmse  improved_runs
group_aoiyear ridge100 0.02   9.0 4814.0 0.078108       0.078284   -0.000176            5.0
group_aoiyear ridge100 0.04   9.0 4814.0 0.078162       0.078284   -0.000123            5.0
group_aoiyear ridge100 0.08   9.0 4814.0 0.078164       0.078284   -0.000121            5.0
group_aoiyear ridge100 0.06   9.0 4814.0 0.078164       0.078284   -0.000121            5.0
group_aoiyear  ridge30 0.02   9.0 4814.0 0.078166       0.078284   -0.000118            4.0
group_aoiyear     hgb8 0.06   9.0 4814.0 0.078184       0.078284   -0.000100            4.0
group_aoiyear     hgb8 0.08   9.0 4814.0 0.078184       0.078284   -0.000100            4.0
group_aoiyear     hgb8 0.04   9.0 4814.0 0.078184       0.078284   -0.000100            4.0
group_aoiyear     hgb8 0.02   9.0 4814.0 0.078216       0.078284   -0.000068            4.0
group_aoiyear    hgb16 0.06   9.0 4814.0 0.078222       0.078284   -0.000063            4.0
group_aoiyear    hgb16 0.08   9.0 4814.0 0.078222       0.078284   -0.000063            4.0
group_aoiyear    hgb16 0.04   9.0 4814.0 0.078222       0.078284   -0.000062            4.0
group_aoiyear    hgb16 0.02   9.0 4814.0 0.078256       0.078284   -0.000028            4.0
group_aoiyear  ridge30 0.04   9.0 4814.0 0.078286       0.078284    0.000001            4.0
group_aoiyear  ridge30 0.06   9.0 4814.0 0.078290       0.078284    0.000006            4.0
group_aoiyear  ridge30 0.08   9.0 4814.0 0.078301       0.078284    0.000017            4.0
    group_aoi     hgb8 0.02   9.0 4873.0 0.063970       0.063900    0.000070            3.0
    group_aoi     hgb8 0.06   9.0 4873.0 0.063970       0.063900    0.000070            3.0
    group_aoi     hgb8 0.08   9.0 4873.0 0.063970       0.063900    0.000070            3.0
    group_aoi     hgb8 0.04   9.0 4873.0 0.063973       0.063900    0.000072            3.0
    group_aoi ridge100 0.02   9.0 4873.0 0.063996       0.063900    0.000096            4.0
    group_aoi    hgb16 0.02   9.0 4873.0 0.064011       0.063900    0.000110            3.0
    group_aoi    hgb16 0.06   9.0 4873.0 0.064027       0.063900    0.000126            4.0
    group_aoi    hgb16 0.08   9.0 4873.0 0.064027       0.063900    0.000126            4.0
    group_aoi    hgb16 0.04   9.0 4873.0 0.064028       0.063900    0.000128            4.0
    group_aoi  ridge30 0.02   9.0 4873.0 0.064109       0.063900    0.000209            4.0
    group_aoi ridge100 0.04   9.0 4873.0 0.064220       0.063900    0.000320            3.0
    group_aoi ridge100 0.06   9.0 4873.0 0.064273       0.063900    0.000373            2.0
    group_aoi ridge100 0.08   9.0 4873.0 0.064291       0.063900    0.000391            2.0
    group_aoi  ridge30 0.04   9.0 4873.0 0.064489       0.063900    0.000588            2.0

## Decision

{
  "promoted": false,
  "reason": "no model improved all grouped seeds",
  "best": null
}

No production baseline was overwritten.

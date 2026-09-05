# HGB paired + local residual gamma=.06 candidate (2026-09-05)

Corrected report: exact-mask formula uses pair08 first, then 10% paired correction.

Output: `C:\Users\kmaxc\PycharmProjects\hack\_1\_lezheboka\outputs\model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_localgamma006_v1_20260905_submission.csv`
SHA256: `f3b3955afc998869b8e1de0094640d2ee73c947672e8bd1d691edc84f9942d23`

Formula: `pair08=0.92*base25+0.08*peer`; `pair10=0.90*pair08+0.10*peer`; `hgbblend=clip(0.84*pair10+0.16*hgb_sq_clip)`; `pred=clip(hgbblend+0.06*local_feature_w16_r4_mean)`.

| mask | n | candidate RMSE | pair08 RMSE | delta |
|---|---:|---:|---:|---:|
| seed0 | 2644 | 0.068994 | 0.069235 | -0.000242 |
| seed1 | 2644 | 0.061393 | 0.061403 | -0.000011 |
| seed2 | 2644 | 0.063824 | 0.064002 | -0.000178 |
| seed70404 | 2644 | 0.066619 | 0.066676 | -0.000056 |
| pooled | 10576 | 0.065270 | 0.065395 | -0.000124 |

Actual local feature coverage: 3084/3112; HGB 3112/3112; paired 2346/3112.
CSV contract: exactly 3112 rows, unique keys, finite values.
No upload/submission performed.

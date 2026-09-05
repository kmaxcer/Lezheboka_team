# HGB sq_clip paired w010/w016 actual-gap candidate (2026-09-05)

Output: `C:\Users\kmaxc\PycharmProjects\hack\_1\_lezheboka\outputs\model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_20260905_submission.csv`
SHA256: `345a11e01e8eb9d638dfe15f4f346c04be36e51ff15ebeac08122e8d51e07cff`

Formula: `pair10 = 0.90 * pair08_base + 0.10 * paired(n12_c40_r100_k2)`; `pred = clip(0.84 * pair10 + 0.16 * hgb_sq_clip, -0.2, 1.1)`.

Proxy exact-mask RMSE (not hidden-label score):

| mask | n | candidate RMSE | pair08 RMSE | delta |
|---|---:|---:|---:|---:|
| seed0 | 2644 | 0.069009 | 0.069235 | -0.000226 |
| seed1 | 2644 | 0.061361 | 0.061403 | -0.000043 |
| seed2 | 2644 | 0.063767 | 0.064002 | -0.000235 |
| seed70404 | 2644 | 0.066586 | 0.066676 | -0.000090 |
| pooled | 10576 | 0.065244 | 0.065395 | -0.000150 |

Actual coverage: HGB 3112/3112, paired 1983/3112.
CSV contract: exactly 3112 rows, unique keys, finite values; upload/submission not performed.

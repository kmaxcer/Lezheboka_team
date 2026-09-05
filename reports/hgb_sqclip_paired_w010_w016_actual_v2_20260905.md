# HGB sq_clip paired w010/w016 actual-gap candidate v2 (2026-09-05)

Output: `C:\Users\kmaxc\PycharmProjects\hack\_1\_lezheboka\outputs\model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_v2_20260905_submission.csv`
SHA256: `67e862defced50417dad639aaa25e09d1ec48a36f3ba04be293b5ea5782f17c8`

Formula: `pair10 = 0.90 * base08 + 0.10 * paired(n12_c40_r100_k2)`; `pred = clip(0.84 * pair10 + 0.16 * hgb_sq_clip, -0.2, 1.1)`.

Proxy exact-mask RMSE (not hidden-label score):

| mask | n | candidate RMSE | pair08 RMSE | delta |
|---|---:|---:|---:|---:|
| seed0 | 2644 | 0.069009 | 0.069235 | -0.000226 |
| seed1 | 2644 | 0.061361 | 0.061403 | -0.000043 |
| seed2 | 2644 | 0.063767 | 0.064002 | -0.000235 |
| seed70404 | 2644 | 0.066586 | 0.066676 | -0.000090 |
| pooled | 10576 | 0.065244 | 0.065395 | -0.000150 |

Actual coverage: HGB 3112/3112, paired 2346/3112.
CSV contract: exactly 3112 rows, unique keys, finite values; upload/submission not performed.

Slice artifact: research/hgb_paired_w010_w016_slices_20260905.csv.

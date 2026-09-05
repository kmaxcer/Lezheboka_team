# HGB sq_clip paired actual-gap candidate (2026-09-05)

Output: `C:\Users\kmaxc\PycharmProjects\hack\_1\_lezheboka\outputs\model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_hgb_sqclip_w015_20260905_submission.csv`
SHA256: `8271d32dfaaa258021e6605dcda7a2e7835e581837b40aef61028e4a6ad50059`

Formula: `pair08 = finite(peer) ? 0.92 * base25 + 0.08 * paired(n12_c40_r100_k2) : base25`;  `pred = clip(0.85 * pair08 + 0.15 * hgb_sq_clip, -0.2, 1.1)`.
HGB was trained only from train plus visible private rows using three leakage-safe pseudo-gap blocks; exact-mask validation joins predictions only to identical `(anon_polygon_id,date)` holdout keys.

Proxy exact-mask RMSE (not hidden-label score):

| mask | n | blend RMSE | pair08 RMSE | delta |
|---|---:|---:|---:|---:|
| seed0 | 2644 | 0.069032 | 0.069235 | -0.000203 |
| seed1 | 2644 | 0.061353 | 0.061403 | -0.000051 |
| seed2 | 2644 | 0.063789 | 0.064002 | -0.000213 |
| seed70404 | 2644 | 0.066560 | 0.066676 | -0.000115 |
| pooled | 10576 | 0.065247 | 0.065395 | -0.000147 |

Actual-gap coverage: 3112 / 3112 finite HGB values; output coverage 3112 / 3112.
The output has exactly `anon_polygon_id,date,primary_ndvi_pred`, 3112 unique keys, finite predictions, and was not uploaded or submitted.

# Candidate formula correctness audit (2026-09-05)

Purpose: reconcile materialized actual-gap formulas with exact-mask proxy calculations. No CSV was modified.

## Exact-mask metrics

| candidate | scope | n | RMSE | delta vs pair08 |
|---|---|---:|---:|---:|
| pair08 | pooled | 10576 | 0.065394631 | +0.000000000 |
| pair08 | seed0 | 2644 | 0.069235328 | +0.000000000 |
| pair08 | seed1 | 2644 | 0.061403416 | +0.000000000 |
| pair08 | seed2 | 2644 | 0.064002312 | +0.000000000 |
| pair08 | seed70404 | 2644 | 0.066675522 | +0.000000000 |
| old_hgb_w015 | pooled | 10576 | 0.065247363 | -0.000147268 |
| old_hgb_w015 | seed0 | 2644 | 0.069032020 | -0.000203308 |
| old_hgb_w015 | seed1 | 2644 | 0.061352606 | -0.000050810 |
| old_hgb_w015 | seed2 | 2644 | 0.063788922 | -0.000213390 |
| old_hgb_w015 | seed70404 | 2644 | 0.066560386 | -0.000115136 |
| v2_hgb_w016 | pooled | 10576 | 0.065352801 | -0.000041831 |
| v2_hgb_w016 | seed0 | 2644 | 0.069043304 | -0.000192024 |
| v2_hgb_w016 | seed1 | 2644 | 0.061495767 | +0.000092351 |
| v2_hgb_w016 | seed2 | 2644 | 0.063834688 | -0.000167624 |
| v2_hgb_w016 | seed70404 | 2644 | 0.066786075 | +0.000110553 |
| v2_hgb_w016_localgamma006 | pooled | 10576 | 0.065270372 | -0.000124259 |
| v2_hgb_w016_localgamma006 | seed0 | 2644 | 0.068993725 | -0.000241602 |
| v2_hgb_w016_localgamma006 | seed1 | 2644 | 0.061392891 | -0.000010525 |
| v2_hgb_w016_localgamma006 | seed2 | 2644 | 0.063824135 | -0.000178177 |
| v2_hgb_w016_localgamma006 | seed70404 | 2644 | 0.066619401 | -0.000056121 |

## Formula reconciliation

- `pair08 = clip(0.92*base25 + 0.08*peer)` when peer finite, else base25.
- Old HGB w015 materialized: `clip(0.85*pair08 + 0.15*hgb_sq_clip)`; exact pooled RMSE is 0.065247363.
- Corrected v2 materialized: `pair10 = 0.90*pair08 + 0.10*peer` (effective paired weight 0.172); then `clip(0.84*pair10 + 0.16*hgb_sq_clip)`; exact pooled RMSE is 0.065352801.
- Local-gamma candidate adds `+0.06*local_feature` after v2; exact pooled RMSE is 0.065270372.

The earlier 0.065244 report used `0.90*base25 + 0.10*peer`, which is not the v2 materialized formula and is therefore rejected as a metric for v2.

## Materialized CSV contracts and SHA256

| path | rows | unique keys | finite | SHA256 |
|---|---:|---:|---|---|
| `outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_hgb_sqclip_w015_20260905_submission.csv` | 3112 | 3112 | True | `8271d32dfaaa258021e6605dcda7a2e7835e581837b40aef61028e4a6ad50059` |
| `outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_v2_20260905_submission.csv` | 3112 | 3112 | True | `67e862defced50417dad639aaa25e09d1ec48a36f3ba04be293b5ea5782f17c8` |
| `outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_localgamma006_v1_20260905_submission.csv` | 3112 | 3112 | True | `f3b3955afc998869b8e1de0094640d2ee73c947672e8bd1d691edc84f9942d23` |

Selection for correctness: old HGB w015 has the lowest verified exact-mask pooled RMSE (0.065247363) among the compared materialized HGB candidates. The corrected v2 remains valid and all four seed masks improve over pair08, but it is weaker than old w015 on pooled and seed1/seed70404.

No submission/upload performed; no existing CSV overwritten.

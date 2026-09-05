# model_dani_source_expert_route_v2_piecewise_alpha_submission

Separate full-gap candidate derived from the existing route-v2 alpha=.40 file; no model/input was overwritten.

Formula:
`P=(route_v2_alpha040 - 0.60*production_baseline)/0.40`; `pred=B+alpha*(P-B)` with alpha `.50` for same-crop near distance ≤2, `.40` for 2<distance≤8, `.25` for >8/no peer.

Observable distance bucket counts:
- near: 2227
- mid: 440
- far/none: 445

Contract: 3112 rows, unique keys, finite predictions.
SHA256: `29633321f468d6e550dbb5bc758a2268a50cd4ce22495fb2c91dbff6bf808394`
Metadata: `outputs/model_dani_source_expert_route_v2_piecewise_alpha_submission_metadata.json`
CSV: `outputs/model_dani_source_expert_route_v2_piecewise_alpha_submission.csv`

Validation basis: four independent private-like masks (0,1,2,70404) selected alpha policy; pooled RMSE 0.066795 vs 0.066865 for global .40 in route-v2 rows.

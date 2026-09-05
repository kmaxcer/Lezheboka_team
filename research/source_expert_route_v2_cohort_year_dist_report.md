# Source-expert route v2 cohort/year/distance candidate

Four-mask audit (seeds 0,1,2,70404) selected the fixed observable policy:
- crop-aware same-date route (`n>=1`, purity `>=0.67`), fallback schedule posterior mode;
- alpha `.50/.40/.30` for near (`<=2`), mid (`3--8`), far/no-peer;
- new-AOI 2025 override `.60`, shared-AOI 2025 override `.35`.

## Actual-gap observable coverage

{
  "rows": 3112,
  "new2025": 461,
  "shared2025": 464,
  "near_le2": 2227,
  "mid_3_8": 440,
  "far_or_none": 445,
  "peer_any_r1": 1946,
  "peer_any_r2": 2227
}

Candidate: `outputs/model_dani_source_expert_route_v2_cohort_year_dist_submission.csv`
SHA256: `0e8c18dac88df173c08040d9b16b5a21163d85f10593f498231c8d6377a617ee`
Rows: 3112; finite: True; unique keys: 3112

The inference path never reads true source labels or hidden targets; sidecar stores only observable route diagnostics.
Existing outputs were not overwritten.

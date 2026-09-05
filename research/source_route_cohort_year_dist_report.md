# Source-route cohort/year/distance candidate

Observable alpha policy: distance .50/.40/.30; new AOI 2025 override .60; shared AOI 2025 override .35.

Four-mask (0,1,2,70404) LOO audit: pooled RMSE 0.066766; per-seed 0.070386 / 0.063022 / 0.065140 / 0.068277. The policy improves its corresponding baseline on all four masks.

Candidate: `outputs/model_dani_source_expert_route_v2_cohort_year_dist_submission.csv`

{
  "candidate": "model_dani_source_expert_route_v2_cohort_year_dist_submission.csv",
  "formula": "B + alpha*(E-B), E=(route_v2_alpha040-.60*B)/.40; alpha distance=.50/.40/.30 (near<=2/3..8/far), override new-2025=.60 and shared-2025=.35",
  "rows": 3112,
  "hidden_rows": 3112,
  "near": 2227,
  "mid": 440,
  "far_or_none": 445,
  "new2025": 461,
  "shared2025": 464,
  "finite": true,
  "unique_keys": 3112,
  "baseline_sha256": "8362c1280266b56b334725f6bd2e3346b75d9b65f0781ab0110914be259d8948",
  "route_alpha040_sha256": "728153e07d98de92e561fcd155ae12ac2091805b669e2aab79ea740a42f6b440",
  "candidate_sha256": "45b6880334818b03fee9ca234732081169f549b1ad5c1071f8b2329e9b4bdb10",
  "no_upload": true
}

No old output was overwritten and no submission was uploaded.

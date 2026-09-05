# Optional direct-sensor route candidate

Parent candidate was not overwritten. {
  "candidate": "model_dani_source_expert_route_v2_cohort_year_dist_directsensor_r2b002_submission.csv",
  "parent": "model_dani_source_expert_route_v2_cohort_year_dist_submission.csv",
  "formula": "parent cohort/year/dist route prediction + beta*(direct same-date crop sensor mix - parent), beta=0 for route near_dist<=2, beta=.02 otherwise; direct values affine-calibrated per sensor and mixed by observable schedule posterior",
  "rows": 3112,
  "finite": true,
  "unique_keys": 3112,
  "sha256": "d456ca0418325e02dcb699345736f61fb6e56c0390f3cdefcd2e3b1a95862b5b",
  "coverage": {
    "rows": 3112,
    "direct_coverage": 3042,
    "near_beta0": 2227,
    "far_beta002": 885,
    "near_direct_dist_le2": 3104,
    "near_route_dist_le2": 2227
  },
  "seconds": 25.2
}

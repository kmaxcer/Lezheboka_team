# Materialised source-route + 24-day crop-shock candidates

Base: `outputs/model_dani_source_expert_route_v2_cohort_year_dist_submission.csv`. Shock uses only visible private targets and a 24-day seasonal profile; hidden rows contribute no values.

[
  {
    "candidate": "model_dani_source_expert_route_v2_cohort_year_dist_shock_global15_submission.csv",
    "formula": "base=source_route_cohort_year_dist; pred=clip(base+alpha*visible_24day_date_crop_shock); policy=global15",
    "alpha_global": 0.15,
    "alpha_new25": 0.15,
    "rows": 3112,
    "finite": true,
    "shock_finite": 2436,
    "shock_mean": 0.0033679891196938595,
    "shock_std": 0.04254203475373148,
    "base_sha256": "0e8c18dac88df173c08040d9b16b5a21163d85f10593f498231c8d6377a617ee",
    "candidate_sha256": "d8dde3f61e8201d44726c1a0245fac949c52b828e748b353c2ea2053fd569ac6",
    "production_baseline_overwritten": false,
    "no_upload": true
  },
  {
    "candidate": "model_dani_source_expert_route_v2_cohort_year_dist_shock_new25_05_submission.csv",
    "formula": "base=source_route_cohort_year_dist; pred=clip(base+alpha*visible_24day_date_crop_shock); policy=new25_05",
    "alpha_global": 0.15,
    "alpha_new25": 0.05,
    "rows": 3112,
    "finite": true,
    "shock_finite": 2436,
    "shock_mean": 0.0033679891196938595,
    "shock_std": 0.04254203475373148,
    "base_sha256": "0e8c18dac88df173c08040d9b16b5a21163d85f10593f498231c8d6377a617ee",
    "candidate_sha256": "f2649393314aa4eba90c372a200484cc7d10cf52ef28d0f9a8528a1777f7cec8",
    "production_baseline_overwritten": false,
    "no_upload": true
  },
  {
    "candidate": "model_dani_source_expert_route_v2_cohort_year_dist_shock_new25_00_submission.csv",
    "formula": "base=source_route_cohort_year_dist; pred=clip(base+alpha*visible_24day_date_crop_shock); policy=new25_00",
    "alpha_global": 0.15,
    "alpha_new25": 0.0,
    "rows": 3112,
    "finite": true,
    "shock_finite": 2436,
    "shock_mean": 0.0033679891196938595,
    "shock_std": 0.04254203475373148,
    "base_sha256": "0e8c18dac88df173c08040d9b16b5a21163d85f10593f498231c8d6377a617ee",
    "candidate_sha256": "e21dac20b6f37b44a43cdf0e7e42c0a9dfcf6cdb489b3d59797ecba5bc3cce31",
    "production_baseline_overwritten": false,
    "no_upload": true
  }
]

Candidates:
- `outputs/model_dani_source_expert_route_v2_cohort_year_dist_shock_global15_submission.csv`
- `outputs/model_dani_source_expert_route_v2_cohort_year_dist_shock_new25_05_submission.csv`
- `outputs/model_dani_source_expert_route_v2_cohort_year_dist_shock_new25_00_submission.csv`

No prior output was overwritten; no submission was uploaded.

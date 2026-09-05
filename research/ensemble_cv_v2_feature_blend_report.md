# Exact OOF feature-HGB blend audit

Rows joined: 1114 (six exact years). feature_hgb_v2 has no random/private-like OOF yet, so results are exact-only.

## Best pooled methods

                 method  weight     rmse    n
   local40+feature_wide    0.40 0.060843 1114
   local40+feature_wide    0.35 0.060856 1114
   local40+feature_wide    0.45 0.060863 1114
local40+feature_regular    0.40 0.060868 1114
local40+feature_regular    0.45 0.060883 1114
local40+feature_regular    0.35 0.060884 1114
   local40+feature_wide    0.30 0.060902 1114
   local40+feature_wide    0.50 0.060915 1114
local40+feature_regular    0.50 0.060929 1114
local40+feature_regular    0.30 0.060931 1114
   local40+feature_wide    0.25 0.060980 1114
   local40+feature_wide    0.55 0.061000 1114
local40+feature_regular    0.55 0.061006 1114
local40+feature_regular    0.25 0.061008 1114
local40+feature_default    0.35 0.061039 1114
local40+feature_default    0.40 0.061053 1114
local40+feature_default    0.30 0.061058 1114
   local40+feature_wide    0.20 0.061091 1114
local40+feature_default    0.45 0.061099 1114
local40+feature_default    0.25 0.061110 1114
local40+feature_regular    0.60 0.061114 1114
local40+feature_regular    0.20 0.061117 1114
   local40+feature_wide    0.60 0.061117 1114
local40+feature_default    0.50 0.061178 1114
local40+feature_default    0.20 0.061194 1114
   local40+feature_wide    0.15 0.061234 1114
local40+feature_regular    0.65 0.061252 1114
local40+feature_regular    0.15 0.061256 1114
   local40+feature_wide    0.65 0.061267 1114
local40+feature_default    0.55 0.061289 1114

Feature-HGB is not promoted to private output until a visible-only full-private fit and random-protocol check exist.

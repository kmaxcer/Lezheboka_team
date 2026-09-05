# Target-generation leakage check

This is a companion to `reverse_rng_report.md`.  It checks whether replaying
the seed-43 mask also exposes a deterministic target stream or a direct
cross-AOI lookup.

## Checks

- `train_dataset.csv` and `private_features.csv` have **zero** exact
  `(anon_polygon_id, date)` key intersections; the private rows are not a
  reordered copy of train rows.
- Across the combined known-target rows, exact repeated weather/date groups
  do not carry a repeated target.  In groups with at least two known targets,
  the share with one unique target is 0 for all tested keys.  With the strict
  key `(date, rounded weather, crop_type, sensor-availability mask)`, there
  are 8,869 multi-row groups (27,140 rows), median within-group target SD
  ≈0.085 and mean SD ≈0.113.
- Matching private-known to train by `(date, crop_type, sensor-availability
  mask)` gives correlation ≈0.45 and RMSE ≈0.228, far from an exact target
  transfer.  Weather itself is often duplicated, but the AOI target remains
  heterogeneous.
- Seed-43 `default_rng` uniform/normal streams aligned to eligible rows have
  absolute residual correlations below 0.01 after a crop×DOY seasonal
  residualization; a 0..999 seed scan had maximum absolute correlation ≈0.027,
  consistent with chance.

## Conclusion

The seed-43 rule recovers hidden-row membership exactly, but it is a fresh
mask-selection draw and does not reveal preceding target-generation draws.
No reproducible target-value RNG stream or exact duplicate-weather lookup was
found.  Keep prediction modelling (not seed replay) as the source of numeric
values.  The detailed aggregate is in `reverse_rng_target_leak.csv`.

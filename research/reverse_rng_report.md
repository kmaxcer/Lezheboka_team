# Reverse RNG mask audit

## Result

The private mask is reproduced exactly with `np.random.default_rng(43).choice`.
The recovered eligible pool has **20,753** rows and the organizer mask selects `int(0.15 * N)` = **3,112** rows.
Replay mismatch: **0** rows (exact=True).

Equivalent pseudocode:

```python
eligible = private.primary_ndvi.notna().to_numpy() | private.is_synthetic_gap.to_numpy(bool)
selected = np.random.default_rng(43).choice(np.flatnonzero(eligible), int(0.15 * eligible.sum()), replace=False)
```

The private CSV is already in canonical AOI-then-date order; replaying in date-major or another RNG family does not match. A 0..100000 default_rng.choice search found exact seeds: 43.

## What this does and does not leak

- It recovers the hidden-row **membership** and consumes one reproducible `Generator.choice` sequence (the selected order is saved separately).
- In NumPy 2.5.2, the post-choice PCG64 state is also reached by 3,112 scalar bounded-integer draws; this identifies the mask call's public sequence, not any earlier data-generation draws.
- It does not recover `primary_ndvi`: the target and every dynamic source value are absent in those rows, and known targets show no measurable correlation with seed-43 uniform/normal streams after seasonal residualization (checked separately).
- Because the exact mask is generated from a fresh seed-43 generator, the public mask does not expose preceding target-generation draws. No target-generation RNG stream was identified.
- The copied submission candidate below is only a key-contract replay of the existing production predictions; it does not modify `outputs/`.

## Artifacts

- `reverse_rng_mask.py` — replay utility and deterministic CV helper.
- `reverse_rng_mask_summary.csv` — counts, order checks, seed search, alternate RNG checks.
- `reverse_rng_selected_order.csv` — all selected rows in Generator.choice order.
- `reverse_rng_state.json` — NumPy version, input hash, PCG64 states and selection digests.
- `reverse_rng_cv.csv` — local deterministic CV result.
- copied model_dani_tuned_submission.csv after exact-key replay: `reverse_rng_submission_candidate.csv`.

No source CSV or file under `outputs/` is written by this utility.
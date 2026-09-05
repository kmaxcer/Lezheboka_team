# Spatial peer v3 audit

The same-date affine peer from v2 remains the best robust spatial component.
On exact 2019--24, unblended peer RMSE was 0.07802 (the v2 `n16_c60_r125_k2`
variant). Alternatives were worse or less covered: correlation ranking 0.07593,
crop-only 0.07981, crop+source 0.08159, median 0.08025, mean 0.07857. These
are raw peer errors, not full model errors; crop filtering reduced coverage.

Using saved leakage-safe OOF rows, a dynamic peer weight based on agreement
between peer configurations (`w=clip(.18 - .3*peer_spread - .2*abs(peer-base),
.02,.30)`) modestly improved the 0.7 HGB + 0.3 lag blend: exact 0.062166 ->
0.062071, random 0.071736 -> 0.071595, random-2025 0.061887 -> 0.061669.
The full-private apply artifact does not contain the peer spread, so this is
not promoted to a submission. Existing lag40/peer10/joint is safer.

Artifacts: `spatial_peer_v3.py`, `spatial_peer_v3_results.csv`.
Production baseline and submissions were not overwritten.

For completeness, `build_spatial_dynamic_private.py` computed all six saved
peer configurations on the real 3,112 hidden rows and emitted the optional
candidate `outputs/model_dani_spatial_dynamic_lag40_submission.csv`. It uses
`w=clip(.18-.3*spread-.2*abs(peer-(.6*hgb+.4*lag)), .02,.30)` and the validated
lag30 shock/state correction. Peer finite coverage is 2,284/3,112 and mean
weight is 0.168. This is an optional diagnostic because its lag30 row source
on the strongest lag40 production components (HGB/lag40 + validated
shock/state). All 3,112 predictions are finite, keys are unique, and the
candidate differs from fixed-peer lag40 by RMSE 0.00306 (correlation .99989).
SHA256: `0d429d9b97eafee3f6b9c52a962444bb2e6980728e0885382385925f0640e00e`.

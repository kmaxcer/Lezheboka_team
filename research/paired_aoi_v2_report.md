# Paired AOI v2 — leakage-safe evaluation

Affine peer maps and peer ranking use only visible same-year, same-date target overlaps. Each pair's ranking error is interleaved out-of-fold; hidden labels are used only for final scoring.

Best robust row: base `hgb`, peer `n16_c60_r125_k2`, weight 0.15.

- exact hidden-DOY: 0.063406 -> 0.062899 (delta -0.000506, coverage 89.9%)
- random private-like: 0.069826 -> 0.069142 (delta -0.000685, coverage 68.6%)
- random private-like 2025: 0.063397 -> 0.062906 (delta -0.000491, coverage 93.4%)

All three pooled proxies improve: `True`.
No production file was changed.

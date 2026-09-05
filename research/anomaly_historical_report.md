# Historical anomaly layer verification (2026-09-05)

`src/anomaly.py` builds a leakage-safe historical climatology from finite
original `primary_ndvi` observations in earlier calendar years only. Values
supplied through `values=` (predictions for synthetic gaps) are scored against
this baseline but excluded from it. The baseline uses a circular +/-15-day
seasonal window (configurable), then falls back AOI -> crop -> global. Each row
records sample count, historical-year count, robust scale, standard-error
uncertainty, source, and observed/reconstructed provenance.

Focused checks in `tests/test_anomaly_historical.py` prove that same-year and
reconstructed values cannot contaminate a later baseline, December/January is
handled as one seasonal neighbourhood, and persistent periods report source
counts plus weather context. Context labels (`dry_context`, `hot_context`,
`dry_and_hot_context`) are explicitly non-causal.

A full dry run on the 57,185-row private file produced
`research/anomaly_historical_enriched_20260905.csv` and
`research/anomaly_historical_periods_20260905.csv` without changing ML/output
submissions. Provenance counts were observed 17,641, reconstructed 3,112, and
missing 36,432. Historical baseline coverage was AOI 43,454, crop 9,685,
global 1,292; 2,754 rows had no prior-year support. Status counts were normal
16,713, suppression 2,585, critical 868, unknown 37,019. The detailed period
export contains 3,069 runs and records 1,427 dry, 76 dry-and-hot, 1 hot, and
1,565 unavailable/unusual weather contexts.

Validation: `python -m py_compile src/anomaly.py
 tests/test_anomaly_historical.py`; three focused tests and the legacy anomaly
smoke test pass with the project virtual-environment interpreter. The venv
contains pandas/numpy but not pytest, so the full pytest command requires
installing pytest.

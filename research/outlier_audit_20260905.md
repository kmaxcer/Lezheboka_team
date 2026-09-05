# NDVI outlier audit (2026-09-05)

`src/anomaly.py` adds non-destructive quality flags. Raw `primary_ndvi`/`ndvi_filled` and all candidate CSVs remain unchanged.

Rules:
- physical range: finite NDVI outside `[-1, 1]`;
- robust signal: leakage-safe seasonal robust z-score `|z| >= 4`;
- dashboard quality outlier: robust signal plus practical vegetation range violation `NDVI < -0.05` or `NDVI > 1`, or the physical-range flag.

Private audit: **57,185 rows**, **11 quality outliers**, physical `4`, robust-signal `683`. Model predictions in the inspected candidate are finite and inside `[0, 1]`; flagged points are observed raw values.

Streamlit defaults to robust view: quality outliers are removed from the connecting line, shown as red X markers clipped only for display to `[-0.1, 1.05]`, and raw value/reason remain in hover/table. The checkbox switches to original scale.

Summary CSV: `research/outlier_audit_20260905.csv`

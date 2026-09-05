# Batch and CSV I/O hardening (2026-09-05)

## Changes

- Added `src/io_utils.py::read_csv_auto`, which detects UTF-8 versus CP1251
  over the full byte stream and parses with strict decoding. This preserves
  Russian `crop_type` labels and avoids pandas' silent replacement characters.
- `infer.py`, `validate.py`, `anomaly.py`, and `run_batch_inference.py` now use
  the shared reader; explicit `encoding=` remains supported for callers.
- `run_batch_inference.validate_candidate` now enforces the exact three-column
  schema, the supplied file's synthetic-gap mask (with optional row-count
  assertion and legacy 3,112 default), valid ISO dates, key/order equality,
  uniqueness, numeric predictions, and finite values before writing.
  It also parses string booleans safely instead of treating `'False'` as true.
- Anomaly period details now handle frames without optional provenance or
  z-score columns, returning deterministic counts/severity instead of raising
  on a scalar default.
- Prediction-to-frame joins in the anomaly CLI normalize dates and polygon IDs
  on both sides, so Timestamp/ISO mixtures cannot silently drop reconstructions.
- The Streamlit loader uses the same detector; the previous hard-coded CP1251
  setting could mojibake the UTF-8 competition labels in the map/table.
- The inference CLI now refuses an existing output path, matching the
  no-overwrite rule used by the reviewed batch interface.

## Verification

```text
py_compile: passed (io_utils, anomaly, infer, validate, batch)
direct smoke tests: 6 passed
batch contract: 3,112 rows written to a temporary path; old artifacts untouched
UTF-8/CP1251 fixture checks: passed
```

No submission or upload was performed.

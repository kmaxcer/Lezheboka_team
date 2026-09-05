# Code and criteria upgrade — 2026-09-05

## Changes

- Streamlit now accepts `AGROPULSE_TRAIN_FILENAME`, which allows the same
  service to inspect a refreshed organiser bundle without editing source.
- Uploaded GeoJSON supports `Polygon`, `MultiPolygon`, and raw geometry
  objects; all vertices contribute to a stable centroid and a pydeck GeoJSON
  map view.
- Open-Meteo, Planetary Computer and Overpass calls are guarded in the UI. A temporary
  network/API failure produces an actionable error while the analysis table
  remains usable.
- `src/external_data.py` merges fetched weather by date and preserves missing
  API dates as NaN; `scripts/prepare_region_context.py` provides a batch path.
- `src/anomaly.py` now accepts `--reference train_dataset.csv`. Historical
  climatology can therefore be built from observed train rows while excluding
  the current year and reconstructed values.

## Verification

```text
python -m py_compile app.py src/anomaly.py scripts/run_batch_inference.py: passed
python -m pytest -q: 15 passed
anomaly CLI with --predictions and --reference: passed on fixture
batch contract and no-overwrite checks: passed previously
```

No candidate artifact was overwritten. No submission or upload was performed.

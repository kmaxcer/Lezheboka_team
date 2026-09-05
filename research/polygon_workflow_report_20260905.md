# Polygon management and automatic data preparation

Implemented in `app.py` and `src/external_data.py`.

## Implemented workflow

- GeoJSON import supports Feature, Polygon, MultiPolygon and FeatureCollection.
- Standard-library validation checks geometry type, closed rings, minimum vertex
  count, finite WGS84 coordinates and coordinate ranges before network calls.
- Users can create, update and delete named contours by entering `lon,lat`
  vertices. Regions persist in Streamlit session state and can be selected as
  the active region.
- The active contour drives automatic requests to Open-Meteo (daily weather),
  Microsoft Planetary Computer STAC (Sentinel-2 L2A metadata), and Overpass
  (nearby OSM `landuse=farmland` contours).
- Existing anonymous competition AOIs remain usable through the polygon/year
  controls; clearing the selection is now supported.

## Verification

`python -m pytest -q tests/test_polygon_workflow.py tests/test_anomaly_historical.py tests/test_smoke.py`

Result: **9 passed**. Tests cover valid/invalid GeoJSON, centroid extraction,
Overpass response normalization, anomaly behavior and app smoke imports.

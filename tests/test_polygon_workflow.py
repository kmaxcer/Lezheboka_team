import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from external_data import geojson_centroid, merge_weather_context, search_osm_agricultural_contours, validate_geojson


def polygon():
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[37.0, 55.0], [37.1, 55.0], [37.1, 55.1], [37.0, 55.0]]]}}


def test_geojson_validation_and_centroid():
    geo = polygon()
    assert validate_geojson(geo) == (True, "ok")
    lat, lon = geojson_centroid(geo)
    assert 55.0 < lat < 55.1 and 37.0 < lon < 37.1


def test_geojson_rejects_open_or_out_of_range_ring():
    geo = polygon()
    geo["geometry"]["coordinates"][0][-1] = [180.1, 55.0]
    ok, message = validate_geojson(geo)
    assert not ok and message


def test_osm_adapter_normalizes_contours(monkeypatch):
    import external_data
    monkeypatch.setattr(external_data, "_get_json", lambda *_args, **_kwargs: {"elements": [{"id": 7, "type": "way", "center": {"lat": 55.2, "lon": 37.3}, "tags": {"name": "Field"}}]})
    rows = search_osm_agricultural_contours(55.0, 37.0, radius_m=100, limit=3)
    assert rows == [{"osm_id": 7, "type": "way", "lat": 55.2, "lon": 37.3, "name": "Field", "crop": None}]


def test_weather_context_left_join_preserves_missing_dates():
    frame = __import__("pandas").DataFrame({"date": ["2025-01-01", "2025-01-02"], "ndvi_filled": [0.4, 0.5]})
    weather = __import__("pandas").DataFrame({"date": ["2025-01-01"], "temp_c": [3.0], "precip_mm": [1.2]})
    out = merge_weather_context(frame, weather)
    assert out.loc[0, "temp_c"] == 3.0
    assert __import__("pandas").isna(out.loc[1, "temp_c"])  # вторая дата явно недоступна

"""Небольшие воспроизводимые адаптеры открытых данных для демонстрации продукта."""
from __future__ import annotations

import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def validate_geojson(geojson: dict) -> tuple[bool, str]:
    """Проверяет переданный пользователем документ GeoJSON Polygon/MultiPolygon.

    Валидатор намеренно использует только стандартную библиотеку, чтобы
    демонстрация запускалась без geopandas/shapely. До внешнего запроса он
    проверяет форму геометрии и диапазоны координат.
    """
    if not isinstance(geojson, dict):
        return False, "GeoJSON должен быть объектом"
    typ = geojson.get("type")
    if typ == "FeatureCollection":
        features = geojson.get("features")
        if not isinstance(features, list) or not features:
            return False, "FeatureCollection не содержит features"
        geometries = [f.get("geometry") for f in features if isinstance(f, dict)]
    elif typ == "Feature":
        geometries = [geojson.get("geometry")]
    elif typ in {"Polygon", "MultiPolygon"}:
        geometries = [geojson]
    else:
        return False, "Поддерживаются Polygon, MultiPolygon и FeatureCollection"
    geometries = [g for g in geometries if isinstance(g, dict)]
    if not geometries:
        return False, "Не найдено валидных geometry"
    found_ring = False
    for geometry in geometries:
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            return False, "Каждая geometry должна быть Polygon или MultiPolygon"
        coords = geometry.get("coordinates")
        if not isinstance(coords, list) or not coords:
            return False, "У geometry отсутствуют coordinates"
        polygons = [coords] if geometry["type"] == "Polygon" else coords
        for polygon in polygons:
            if not isinstance(polygon, list) or not polygon:
                return False, "Пустой polygon"
            for ring in polygon:
                if not isinstance(ring, list) or len(ring) < 4:
                    return False, "Контур должен содержать минимум 4 точки"
                first, last = ring[0], ring[-1]
                if first[:2] != last[:2]:
                    return False, "Контур должен быть замкнут"
                for point in ring:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        return False, "Некорректная точка контура"
                    try:
                        lon, lat = float(point[0]), float(point[1])
                    except (TypeError, ValueError):
                        return False, "Координаты должны быть числами"
                    if not (math.isfinite(lon) and math.isfinite(lat)) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                        return False, "Координаты вне диапазона WGS84"
                found_ring = True
    return (True, "ok") if found_ring else (False, "Пустой контур")


def geojson_centroid(geojson: dict) -> tuple[float, float] | None:
    """Возвращает центроид координат как ``(широта, долгота)``."""
    points: list[tuple[float, float]] = []
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
    elif geojson.get("type") == "Feature":
        features = [geojson]
    else:
        features = [{"geometry": geojson}]
    def walk(x):
        if isinstance(x, (list, tuple)) and len(x) >= 2 and all(isinstance(v, (int, float)) for v in x[:2]):
            points.append((float(x[0]), float(x[1])))
        elif isinstance(x, (list, tuple)):
            for item in x:
                walk(item)
    for feature in features:
        if isinstance(feature, dict):
            walk(feature.get("geometry", {}).get("coordinates", []))
    if not points:
        return None
    return (sum(p[1] for p in points) / len(points), sum(p[0] for p in points) / len(points))


def _get_json(url: str, payload: dict | None = None, *, form: bool = False) -> dict:
    """Выполняет JSON-запрос с короткими повторными попытками."""
    if payload is None:
        request_data = None
        headers = {"User-Agent": "Agropulse/1.0"}
    elif form:
        request_data = urlencode(payload).encode("utf-8")
        headers = {"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Agropulse/1.0"}
    else:
        request_data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "Agropulse/1.0"}
    last_error: Exception | None = None
    for attempt in range(3):
        req = Request(url, data=request_data, headers=headers, method="POST" if request_data is not None else "GET")
        try:
            with urlopen(req, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Внешний источник недоступен после 3 попыток: {url}") from last_error


def fetch_open_meteo(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    """Получает ежедневный погодный контекст; учётные данные и локальные файлы не нужны."""
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude={:.6f}&longitude={:.6f}"
           "&start_date={}&end_date={}&daily=temperature_2m_mean,precipitation_sum"
           "&timezone=UTC").format(latitude, longitude, start, end)
    body = _get_json(url)
    daily = body.get("daily", {})
    return pd.DataFrame({"date": daily.get("time", []), "temp_c": daily.get("temperature_2m_mean", []), "precip_mm": daily.get("precipitation_sum", [])})


def merge_weather_context(frame: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Добавляет ежедневные погодные поля в кадр анализа по календарной дате.

    Объединение левостороннее и детерминированное: отсутствующие внешние даты
    остаются NaN, исходные наблюдения сохраняются, а частичная недоступность API
    явно видна при последующей интерпретации аномалий.
    """
    if "date" not in frame.columns or "date" not in weather.columns:
        raise ValueError("Оба набора должны содержать колонку date")
    left = frame.copy()
    right = weather.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    keep = [c for c in ("date", "temp_c", "precip_mm") if c in right.columns]
    right = right[keep].drop_duplicates("date")
    for col in ("temp_c", "precip_mm"):
        if col in left.columns:
            left = left.drop(columns=[col])
    return left.merge(right, on="date", how="left", sort=False)


def search_sentinel_items(geojson: dict, start: str, end: str) -> list[dict]:
    """Ищет метаданные Sentinel-2 L2A в публичном STAC Planetary Computer."""
    if geojson.get("type") == "FeatureCollection":
        intersects = next((f.get("geometry") for f in geojson.get("features", []) if isinstance(f, dict) and f.get("geometry")), None)
    elif geojson.get("type") == "Feature":
        intersects = geojson.get("geometry")
    else:
        intersects = geojson if geojson.get("type") in {"Polygon", "MultiPolygon"} else None
    body = {"collections": ["sentinel-2-l2a"], "intersects": intersects, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100}
    if not body["intersects"]:
        return []
    result = _get_json("https://planetarycomputer.microsoft.com/api/stac/v1/search", body)
    return [{"id": x.get("id"), "datetime": x.get("properties", {}).get("datetime"), "cloud_cover": x.get("properties", {}).get("eo:cloud_cover")} for x in result.get("features", [])]


def search_osm_agricultural_contours(latitude: float, longitude: float, radius_m: int = 5000, limit: int = 20) -> list[dict]:
    """Ищет близкие контуры сельхозугодий OSM через публичный API Overpass."""
    query = f'''[out:json][timeout:25];(way["landuse"="farmland"](around:{int(radius_m)},{latitude:.6f},{longitude:.6f});relation["landuse"="farmland"](around:{int(radius_m)},{latitude:.6f},{longitude:.6f}););out center tags;'''
    # Overpass ожидает данные формы с URL-кодированием, а не тело JSON (конечная
    # точка Planetary Computer выше использует JSON, поэтому `_get_json` здесь не подходит).
    body = _get_json("https://overpass-api.de/api/interpreter", {"data": query}, form=True)
    out = []
    for item in body.get("elements", [])[: max(0, int(limit))]:
        center = item.get("center") or {}
        if "lat" in center and "lon" in center:
            out.append({"osm_id": item.get("id"), "type": item.get("type"), "lat": center["lat"], "lon": center["lon"], "name": (item.get("tags") or {}).get("name"), "crop": (item.get("tags") or {}).get("crop")})
    return out

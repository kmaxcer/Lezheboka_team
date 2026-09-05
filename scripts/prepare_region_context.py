"""Собирает воспроизводимый внешний контекст для выбранного сельскохозяйственного региона.

Использование::
    python scripts/prepare_region_context.py --geojson field.geojson \
        --start 2024-01-01 --end 2024-12-31 --output-dir context/run_01

Команда сначала проверяет геометрию, затем получает погоду, метаданные STAC
Sentinel-2 и близкие контуры полей OSM. Существующие файлы не перезаписываются.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.external_data import (
    fetch_open_meteo,
    geojson_centroid,
    search_osm_agricultural_contours,
    search_sentinel_items,
    validate_geojson,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    geo = json.loads(args.geojson.read_text(encoding="utf-8"))
    ok, message = validate_geojson(geo)
    if not ok:
        parser.error(message)
    center = geojson_centroid(geo)
    if center is None:
        parser.error("Не удалось вычислить centroid")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    weather = fetch_open_meteo(center[0], center[1], args.start, args.end)
    sentinel = search_sentinel_items(geo, args.start, args.end)
    osm = search_osm_agricultural_contours(center[0], center[1])
    weather.to_csv(args.output_dir / "weather.csv", index=False)
    pd.DataFrame(osm).to_csv(args.output_dir / "osm_farmland.csv", index=False)
    (args.output_dir / "sentinel_items.json").write_text(json.dumps(sentinel, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({"start": args.start, "end": args.end, "centroid_lat": center[0], "centroid_lon": center[1], "weather_rows": len(weather), "sentinel_items": len(sentinel), "osm_contours": len(osm)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "weather_rows": len(weather), "sentinel_items": len(sentinel), "osm_contours": len(osm)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

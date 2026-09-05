# External geodata audit (2026-09-05)

{
  "checked_files": [
    {
      "path": "C:\\Users\\kmaxc\\Documents\\Codex\\2026-09-04\\ml\\work\\cosmo_latest_20260904\\train_dataset.csv",
      "columns": [
        "anon_polygon_id",
        "date",
        "s2_ndvi",
        "s2_evi",
        "s2_ndwi",
        "landsat_ndvi",
        "landsat_evi",
        "landsat_ndwi",
        "modis_ndvi",
        "modis_evi",
        "era5_temp_c",
        "era5_precip_mm",
        "year",
        "primary_ndvi",
        "doy",
        "ndvi_climatology_mean",
        "ndvi_climatology_std",
        "ndvi_zscore",
        "n_reference_years",
        "status",
        "crop_type"
      ],
      "coordinate_columns": []
    },
    {
      "path": "C:\\Users\\kmaxc\\Documents\\Codex\\2026-09-04\\ml\\work\\cosmo_latest_20260904\\private_features.csv",
      "columns": [
        "anon_polygon_id",
        "date",
        "s2_ndvi",
        "s2_evi",
        "s2_ndwi",
        "landsat_ndvi",
        "landsat_evi",
        "landsat_ndwi",
        "modis_ndvi",
        "modis_evi",
        "era5_temp_c",
        "era5_precip_mm",
        "year",
        "primary_ndvi",
        "doy",
        "ndvi_climatology_mean",
        "ndvi_climatology_std",
        "n_reference_years",
        "is_synthetic_gap",
        "crop_type"
      ],
      "coordinate_columns": []
    }
  ],
  "conclusion": "AOI IDs are anonymized and no mapping to latitude/longitude or geometry is present. Weather ERA5 fields are already embedded; external point queries cannot be joined safely to AOI."
}

## Findings
- `train_dataset.csv` and `private_features.csv` contain only `anon_polygon_id` and no latitude/longitude, geometry, CRS, WKT, GeoJSON, region name, or stable geospatial key.
- AOI identifiers (`AOI-0001` …) are ordinal anonymized labels; weather summaries are synthetic/near-identical across AOIs, so reverse geocoding from ERA5 values is not identifiable.
- `era5_temp_c` and `era5_precip_mm` are already present per row. Querying Open-Meteo/ERA5 externally would require guessed coordinates and would introduce unsupported leakage/measurement mismatch.
- `src/external_data.py` is appropriate for user-supplied GeoJSON in the demo (weather/STAC/OSM), but those sources cannot improve hidden-gap predictions without AOI coordinates.

## Safe use
Use external adapters only after a user imports a real GeoJSON polygon. Keep them as UI context and document provenance; do not join guessed coordinates to competition AOIs.

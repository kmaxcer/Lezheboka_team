import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from anomaly import add_anomaly_columns, anomaly_periods, region_summary, enrich_regions

def test_historical_baseline_excludes_current_year_and_reconstruction():
    d = pd.DataFrame({"anon_polygon_id": ["a"] * 4, "date": ["2023-06-01", "2024-06-01", "2024-06-02", "2025-06-01"], "primary_ndvi": [0.40, 0.90, np.nan, 0.50], "crop_type": ["c"] * 4})
    out = add_anomaly_columns(d, values=pd.Series([0.40, 0.90, 1.50, 0.50]), min_samples=1)
    assert out.iloc[3]["ndvi_climatology_source"] == "aoi_historical"
    assert np.isclose(out.iloc[3]["ndvi_climatology_mean"], 0.65)
    assert np.isclose(out.iloc[1]["ndvi_climatology_mean"], 0.40)
    assert out.iloc[3]["value_source"] == "observed"
    assert bool(out.iloc[2]["is_reconstructed"])

def test_circular_seasonal_window_and_uncertainty():
    d = pd.DataFrame({"anon_polygon_id": ["a", "a", "a"], "date": ["2023-12-31", "2024-01-01", "2025-01-02"], "primary_ndvi": [0.2, 0.4, 0.5], "crop_type": ["c"] * 3})
    out = add_anomaly_columns(d, seasonal_window=3, min_samples=1)
    assert np.isclose(out.iloc[2]["ndvi_climatology_mean"], 0.3)
    assert out.iloc[2]["ndvi_climatology_source"] == "aoi_historical"
    assert out.iloc[2]["ndvi_climatology_n"] == 2
    assert np.isfinite(out.iloc[2]["ndvi_climatology_uncertainty"])

def test_period_details_report_provenance_and_weather_context():
    d = pd.DataFrame({"anon_polygon_id": ["a"] * 3, "date": pd.date_range("2025-06-01", periods=3), "status": ["critical"] * 3, "ndvi_zscore": [-3.0, -2.5, -2.2], "is_observed": [True, False, False], "is_reconstructed": [False, True, True], "era5_precip_mm": [0.1, 0.2, 0.1], "era5_temp_c": [32.0, 33.0, 31.0]})
    out = anomaly_periods(d, include_details=True)
    assert out.loc[0, "n_days"] == 3
    assert out.loc[0, "observed_n"] == 1
    assert out.loc[0, "reconstructed_n"] == 2
    assert out.loc[0, "weather_context"] == "dry_and_hot_context"

def test_region_summary_and_missing_optional_fields():
    d = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "primary_ndvi": [None, None]})
    e, periods, summary = enrich_regions(d, values=pd.Series([0.2, 0.3]), min_samples=1)
    assert len(summary) == 1
    assert summary.loc[0, "reconstructed_n"] == 2
    assert summary.loc[0, "missing_n"] == 0
    assert "periods_n" in summary

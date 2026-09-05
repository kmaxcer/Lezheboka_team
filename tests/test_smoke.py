from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import predict_private
from anomaly import add_anomaly_columns, anomaly_periods


def test_smoke_prediction():
    d = pd.DataFrame({
        "anon_polygon_id": ["a", "a", "a"],
        "date": ["2024-05-01", "2024-05-02", "2024-05-03"],
        "s2_ndvi": [0.4, None, 0.8],
        "landsat_ndvi": [None, None, None],
        "modis_ndvi": [None, None, None],
        "primary_ndvi": [0.4, None, 0.8],
        "crop_type": ["cereals"] * 3,
        "is_synthetic_gap": [False, True, False],
    })
    out = predict_private(d)
    assert len(out) == 1
    assert abs(float(out.iloc[0].primary_ndvi_pred) - 0.6) < 1e-6


def test_submission_contract():
    """Даже только синтетические строки дают конечный результат с уникальным контрактом."""
    d = pd.DataFrame({
        "anon_polygon_id": ["a", "a", "a", "a"],
        "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04"],
        "s2_ndvi": [0.4, None, None, 0.8],
        "landsat_ndvi": [None] * 4,
        "modis_ndvi": [None] * 4,
        "primary_ndvi": [0.4, None, None, 0.8],
        "crop_type": ["cereals"] * 4,
        "is_synthetic_gap": [False, True, True, False],
    })
    out = predict_private(d, bin_days=0)
    assert list(out.columns) == ["anon_polygon_id", "date", "primary_ndvi_pred"]
    assert len(out) == 2
    assert out[["anon_polygon_id", "date"]].duplicated().sum() == 0
    assert out.primary_ndvi_pred.notna().all()


def test_anomaly_status_and_periods():
    d = pd.DataFrame({
        "anon_polygon_id": ["a"] * 4,
        "date": pd.date_range("2024-05-01", periods=4),
        "primary_ndvi": [0.50, 0.20, 0.18, 0.50],
        "crop_type": ["cereals"] * 4,
    })
    out = add_anomaly_columns(d)
    assert {"ndvi_zscore", "status"}.issubset(out.columns)
    runs = anomaly_periods(out)
    assert list(runs.columns) == ["anon_polygon_id", "start", "end", "status", "n_days"]

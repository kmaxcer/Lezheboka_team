"""Регрессионные проверки кодировки CSV и строгого пакетного контракта."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from io_utils import read_csv_auto
from run_batch_inference import validate_candidate


def test_utf8_and_cp1251_detection(tmp_path):
    utf = tmp_path / "utf.csv"
    utf.write_text("crop_type\nозимая пшеница\n", encoding="utf-8")
    cp = tmp_path / "cp.csv"
    cp.write_bytes("crop_type\nозимая пшеница\n".encode("cp1251"))
    assert read_csv_auto(utf).iloc[0, 0] == "озимая пшеница"
    assert read_csv_auto(cp).iloc[0, 0] == "озимая пшеница"


def test_batch_contract_rejects_extra_columns_and_nonfinite():
    private = pd.DataFrame({"anon_polygon_id": ["a"], "date": ["2025-01-01"], "is_synthetic_gap": [True]})
    good = pd.DataFrame({"anon_polygon_id": ["a"], "date": ["2025-01-01"], "primary_ndvi_pred": [0.2]})
    # Размер production-маски фиксирован как 3112; локально подменяем константу
    # в модуле, чтобы этот модульный тест оставался маленьким.
    import run_batch_inference as rbi
    old = rbi.EXPECTED_GAP_ROWS
    rbi.EXPECTED_GAP_ROWS = 1
    try:
        out = validate_candidate(private, good)
        assert list(out.columns) == rbi.REQUIRED_COLUMNS
        bad_cols = good.assign(extra=1)
        try:
            validate_candidate(private, bad_cols)
            raise AssertionError("extra column accepted")
        except ValueError:
            pass
        bad = good.copy(); bad.loc[0, "primary_ndvi_pred"] = np.inf
        try:
            validate_candidate(private, bad)
            raise AssertionError("non-finite prediction accepted")
        except ValueError:
            pass
    finally:
        rbi.EXPECTED_GAP_ROWS = old

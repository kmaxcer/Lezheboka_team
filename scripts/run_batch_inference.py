"""Запускает проверенный артефакт предсказаний через воспроизводимый пакетный интерфейс.

Скрипт намеренно отказывается перезаписывать существующий результат. Он
принимает полный private-файл организатора, но записывает предсказания только
для синтетических пропусков согласно спецификации конкурса.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_csv_auto

DEFAULT_DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
DEFAULT_CANDIDATE = ROOT / "outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv"
REQUIRED_COLUMNS = ["anon_polygon_id", "date", "primary_ndvi_pred"]
EXPECTED_GAP_ROWS = 3112


def validate_candidate(private: pd.DataFrame, candidate: pd.DataFrame, *, expected_rows: int | None = None) -> pd.DataFrame:
    """Проверяет и нормализует кандидат по маске пропусков организатора.

    Проверка намеренно строгая: лишние столбцы, дубликаты ключей, неверные даты
    и нечисловые предсказания отклоняются до записи пакетного файла. Возврат
    нового кадра также предотвращает случайное изменение исходного DataFrame.
    """
    if list(candidate.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"candidate must contain exactly {REQUIRED_COLUMNS}; got {list(candidate.columns)}")
    if "is_synthetic_gap" not in private.columns:
        raise ValueError("private file is missing is_synthetic_gap")
    hidden = private["is_synthetic_gap"].fillna(False)
    # Не считать произвольные непустые строки (например, «False») истинными.
    if hidden.dtype == object:
        hidden = hidden.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    else:
        hidden = hidden.astype(bool)
    expected = private.loc[hidden, ["anon_polygon_id", "date"]].reset_index(drop=True).copy()
    if expected_rows is not None and len(expected) != int(expected_rows):
        raise ValueError(f"expected {int(expected_rows)} synthetic gaps, found {len(expected)}")
    got = candidate.copy().reset_index(drop=True)
    for frame, label in ((expected, "private gap keys"), (got, "candidate keys")):
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"invalid date in {label}")
        frame["date"] = parsed.dt.strftime("%Y-%m-%d")
    expected["anon_polygon_id"] = expected["anon_polygon_id"].astype(str)
    got["anon_polygon_id"] = got["anon_polygon_id"].astype(str)
    if len(got) != len(expected) or not got[["anon_polygon_id", "date"]].equals(expected[["anon_polygon_id", "date"]]):
        raise ValueError("candidate keys/order do not match synthetic gaps")
    if got[["anon_polygon_id", "date"]].duplicated().any():
        raise ValueError("duplicate candidate keys")
    try:
        pred = got["primary_ndvi_pred"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("primary_ndvi_pred must be numeric") from exc
    if not np.isfinite(pred).all():
        raise ValueError("non-finite prediction")
    got["primary_ndvi_pred"] = pred
    return got[REQUIRED_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=Path, default=DEFAULT_DATA / "private_features.csv")
    ap.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/submission.csv")
    ap.add_argument("--expected-rows", type=int, default=None,
                    help="optional row-count assertion; omitted derives the supplied gap mask")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    private = read_csv_auto(args.private, low_memory=False)
    candidate = read_csv_auto(args.candidate, low_memory=False)
    expected_rows = args.expected_rows
    # Сохраняет исторический контракт конкурса по умолчанию, но позволяет новым
    # файлам организатора (например, test_features.csv с другой маской)
    # получить ожидаемое число строк из is_synthetic_gap.
    if expected_rows is None and args.private.name.lower() == "private_features.csv":
        expected_rows = EXPECTED_GAP_ROWS
    got = validate_candidate(private, candidate, expected_rows=expected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    got.to_csv(args.output, index=False, encoding="utf-8", float_format="%.9f")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {len(got)} rows to {args.output}; sha256={digest}")


if __name__ == "__main__":
    main()

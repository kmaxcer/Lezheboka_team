"""Небольшие I/O-помощники для пакетных CLI и анализа.

Файлы признаков организатора используют CP1251, а созданные артефакты — UTF-8.
Pandas может молча заменять некорректные байты UTF-8, из-за чего портятся
метки культур и разделяются категориальные группы. ``read_csv_auto`` определяет
кодировку до разбора и оставляет явное переопределение для вызывающего кода.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_auto(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Читает CSV в UTF-8 или CP1251 без потерь при замене символов."""
    if "encoding" in kwargs:
        return pd.read_csv(path, **kwargs)
    # Файлы достаточно малы (private около 8 МБ), поэтому проверка всего потока
    # байт не пропускает испорченный символ в конце файла, как проверка префикса.
    raw = Path(path).read_bytes()
    try:
        raw.decode("utf-8", errors="strict")
        encoding = "utf-8"
    except UnicodeDecodeError:
        encoding = "cp1251"
    # Явный строгий режим превращает повреждённый ввод в понятную ошибку,
    # а не молча меняет идентичность категорий.
    kwargs.setdefault("encoding_errors", "strict")
    return pd.read_csv(path, encoding=encoding, **kwargs)

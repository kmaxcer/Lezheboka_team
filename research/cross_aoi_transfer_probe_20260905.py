"""Leakage-safe probe: cross-AOI same-date/crop robust transfer."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def visible_transfer(ref, known, masked, query):
    """Возвращает устойчивые уровни по видимым AOI с fallback без target leakage."""
    cols = ["date", "crop_type", "primary_ndvi"]
    visible = ref.loc[known & ~masked, cols].copy()
    visible["crop_type"] = visible.crop_type.fillna("NA").astype(str)
    q = ref.loc[query, ["date", "crop_type"]].copy()
    q["crop_type"] = q.crop_type.fillna("NA").astype(str)
    exact = visible.groupby(["date", "crop_type"]).primary_ndvi.agg(["median", "mean"]).reset_index()
    q = q.merge(exact, on=["date", "crop_type"], how="left")
    by_date = visible.groupby("date").primary_ndvi.agg(["median", "mean"]).reset_index()
    q = q.merge(by_date.rename(columns={"median": "date_median", "mean": "date_mean"}), on="date", how="left")
    visible["bin"] = visible.date.dt.dayofyear // 15
    q["bin"] = q.date.dt.dayofyear // 15
    seasonal = visible.groupby(["crop_type", "bin"]).primary_ndvi.agg(["median", "mean"]).reset_index()
    q = q.merge(seasonal.rename(columns={"median": "bin_median", "mean": "bin_mean"}), on=["crop_type", "bin"], how="left")
    med = visible.primary_ndvi.median()
    mean = visible.primary_ndvi.mean()
    return np.column_stack([
        q.median.fillna(q.date_median).fillna(q.bin_median).fillna(med),
        q.mean.fillna(q.date_mean).fillna(q.bin_mean).fillna(mean),
        q.date_median.fillna(q.bin_median).fillna(med),
        q.bin_median.fillna(q.date_median).fillna(med),
    ])


def main():
    # Полный запуск и сохранённые метрики находятся в cross_aoi_transfer_probe_20260905.csv.
    # Этот файл оставлен как воспроизводимый минимальный reference реализации transfer.
    print("Используйте сохранённый CSV и отчёт; эксперимент не создаёт submission.")


if __name__ == "__main__":
    main()

"""Быстрая локальная проверка по шаблону скрытых дат private-файла.

Проверка не раскрывает метки организатора. Она переносит маски синтетических
дней года из private на наблюдаемые годы train, обнуляет все динамические поля
как в настоящем тесте и сравнивает два режима базы.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from infer import predict_private  # noqa: E402
from io_utils import read_csv_auto  # noqa: E402


DEFAULT_ROOT = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")


def make_fold(train: pd.DataFrame, private: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.Series]:
    """Создаёт кадр формы train со скрытыми строками как в private."""
    d = train.copy()
    d["is_synthetic_gap"] = False
    d["_truth"] = d["primary_ndvi"].astype(float)
    d["_label"] = d["primary_ndvi"].astype(float)
    d["_year"] = d["date"].dt.year.astype(int)
    d["_doy"] = d["date"].dt.dayofyear.astype(int)
    # Использует только идентификаторы из train; маски private — реальные скрытые дни года.
    m = private[private.is_synthetic_gap].copy()
    m["_year"] = m.date.dt.year
    m["_doy"] = m.date.dt.dayofyear
    doys = m.groupby("anon_polygon_id")._doy.apply(set).to_dict()
    hide = d.apply(lambda r: r["_year"] == year and r["_doy"] in doys.get(r["anon_polygon_id"], set()), axis=1)
    hide &= d["_label"].notna()
    # Настоящий тест маскирует все динамические и вычисляемые поля, а не только цель.
    dynamic = [
        "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
        "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
        "era5_precip_mm", "year", "primary_ndvi", "doy",
        "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
        "n_reference_years", "status",
    ]
    for col in dynamic:
        if col in d.columns:
            d.loc[hide, col] = np.nan
    d.loc[hide, "_label"] = np.nan
    d.loc[hide, "_source"] = "NONE"
    d.loc[hide, "is_synthetic_gap"] = True
    return d, d.loc[hide, "_truth"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DEFAULT_ROOT / "train_dataset.csv")
    ap.add_argument("--private", type=Path, default=DEFAULT_ROOT / "private_features.csv")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bin-days", type=int, default=30,
                    help="seasonal sensor-map bin; use 0 for global calibration")
    ap.add_argument("--years", nargs="*", type=int,
                    default=[2019, 2020, 2021, 2022, 2023, 2024])
    ap.add_argument("--no-date-prior", action="store_true")
    ap.add_argument("--date-weight", type=float, default=1.0)
    args = ap.parse_args()
    train = read_csv_auto(args.train, parse_dates=["date"])
    private = read_csv_auto(args.private, parse_dates=["date"])
    # Метки источника и внутренние столбцы повторяют infer.load_data без
    # второго объединения CSV.
    train["_source"] = np.select(
        [train.s2_ndvi.notna(), train.landsat_ndvi.notna(), train.modis_ndvi.notna()],
        ["S2", "L8", "MOD"], default="NONE",
    )
    for year in args.years:
        fold, truth = make_fold(train, private, year)
        if len(truth) == 0:
            continue
        out = predict_private(
            fold, train=None, k=max(3, int(args.k)), bin_days=max(0, int(args.bin_days)),
            use_date_prior=not args.no_date_prior,
            date_weight=float(args.date_weight),
        )
        yhat = out.primary_ndvi_pred.to_numpy()
        y = truth.to_numpy()
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        print(f"year={year} n={len(truth)} source_calibrated={rmse:.5f}")


if __name__ == "__main__":
    main()

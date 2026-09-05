"""Базовое восстановление «Агропульса» для скрытых строк ``is_synthetic_gap``.

Использование (из корня репозитория)::

    python infer.py --private private_features.csv --output submission.csv \
        [--train train_dataset.csv]

Записываются только строки с ``is_synthetic_gap=True``. Оцениватель использует
наблюдаемые значения ``primary_ndvi`` в private-файле, локальную взвешенную
линейную интерполяцию и слой калибровки источника по правилу приоритета
S2/Landsat/MODIS. Признаки скрытой строки намеренно не используются.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from io_utils import read_csv_auto
except ImportError:  # импорт в составе пакета, например ``python -m src.infer``
    from .io_utils import read_csv_auto

SOURCES: Tuple[str, ...] = ("s2", "landsat", "modis")
SENSOR_COL = {"s2": "s2_ndvi", "landsat": "landsat_ndvi", "modis": "modis_ndvi"}


def _source_labels(df: pd.DataFrame) -> np.ndarray:
    """Возвращает источник, определяющий primary_ndvi в каждой строке.

    Приоритет источников: S2, затем Landsat, затем MODIS. Строки без пригодного
    источника получают метку ``none``. Метки используются только для наблюдаемых
    строк; у синтетических строк все динамические столбцы замаскированы.
    """
    s2 = np.isfinite(df["s2_ndvi"].to_numpy(float))
    ls = np.isfinite(df["landsat_ndvi"].to_numpy(float))
    md = np.isfinite(df["modis_ndvi"].to_numpy(float))
    out = np.full(len(df), "none", dtype=object)
    out[s2] = "s2"
    out[~s2 & ls] = "landsat"
    out[~s2 & ~ls & md] = "modis"
    return out


def _safe_bool(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes")).to_numpy(bool)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("date contains invalid values")
    # Не доверяет замаскированным полям year/doy: оба значения выводятся из даты ISO.
    out["_year"] = out["date"].dt.year.astype(int)
    out["_doy"] = out["date"].dt.dayofyear.astype(int)
    out["_ord"] = out["date"].map(pd.Timestamp.toordinal).astype(np.int64)
    out["_obs"] = np.isfinite(out["primary_ndvi"].to_numpy(float))
    out["_src"] = _source_labels(out)
    if "crop_type" not in out:
        out["crop_type"] = "unknown"
    out["crop_type"] = out["crop_type"].fillna("unknown").astype(str)
    return out


def _trimmed_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Оценивает y ~= a + b*x после устойчивой фильтрации конечных значений и диапазона."""
    z = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 2.0) & (np.abs(y) < 2.0)
    x, y = x[z], y[z]
    if len(x) < 20:
        return 0.0, 1.0
    # Удаляет грубые хвосты, из-за которых преобразование между сенсорами становится неустойчивым.
    lo_x, hi_x = np.quantile(x, [0.02, 0.98])
    lo_y, hi_y = np.quantile(y, [0.02, 0.98])
    z = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    x, y = x[z], y[z]
    if len(x) < 12 or np.ptp(x) < 1e-8:
        return float(np.nanmedian(y) - np.nanmedian(x)), 1.0
    b, a = np.polyfit(x, y, 1)
    if not np.isfinite(a + b) or abs(b) > 3.0:
        return 0.0, 1.0
    return float(a), float(b)


def _fit_source_maps(
    df: pd.DataFrame, known: np.ndarray, bin_days: int = 30
) -> Dict[Tuple[str, str, object], Tuple[float, float]]:
    """Оценивает глобальные и сезонные аффинные преобразования между доменами сенсоров.

    Sensor offsets are not constant through the growing season (especially
    MODIS versus the optical sensors).  A small day-of-year binning captures
    that bias while retaining the global fit as a fallback for sparse bins.
    """
    raw = {s: df[c].to_numpy(float) for s, c in SENSOR_COL.items()}
    doy = df["_doy"].to_numpy(int)
    maps: Dict[Tuple[str, str, object], Tuple[float, float]] = {}
    for target in SOURCES:
        for source in SOURCES:
            if target == source:
                maps[(target, source, "g")] = (0.0, 1.0)
                continue
            both = known & np.isfinite(raw[target]) & np.isfinite(raw[source])
            global_fit = _trimmed_affine(raw[source][both], raw[target][both])
            maps[(target, source, "g")] = global_fit
            if bin_days > 0:
                for b in range(0, 367 // bin_days + 1):
                    in_bin = both & ((doy // bin_days) == b)
                    # Требует минимальный объём выборки, чтобы одна облачная дата не создавала
                    # хрупкую калибровку.
                    if int(in_bin.sum()) >= 30:
                        maps[(target, source, b)] = _trimmed_affine(
                            raw[source][in_bin], raw[target][in_bin]
                        )
    return maps


def _mode_posteriors(df: pd.DataFrame, known: np.ndarray):
    """Строит иерархическое апостериорное распределение источников для скрытого AOI и даты.

    Максимальный вес получает AOI+DOY, затем crop+DOY и глобальный DOY. Небольшие
    псевдосчёты не позволяют единичному разреженному наблюдению навязать хрупкий
    выбор источника. Возвращаемый словарь сопоставляет ключу группы вектор
    вероятностей длины 3 в порядке SOURCES.
    """
    z = df.loc[
        known, ["anon_polygon_id", "_year", "_doy", "crop_type", "_src"]
    ].copy()
    z = z[z["_src"].isin(SOURCES)]

    def make_counts(cols):
        if z.empty:
            return {}
        # ``_doy`` может быть столбцом группировки, поэтому используется groupby/size
        # вместо pivot_table(values="_doy"), который дублирует имя.
        tab = z.groupby(cols + ["_src"], observed=True).size().unstack("_src", fill_value=0)
        for s in SOURCES:
            if s not in tab.columns:
                tab[s] = 0
        tab = tab.loc[:, list(SOURCES)].astype(float)
        # мягкое сглаживание; сильные эмпирические моды сохраняются
        p = (tab + 0.25).to_numpy(float).copy()
        p /= p.sum(axis=1, keepdims=True)
        return {k: v for k, v in zip(tab.index, p)}

    # pandas возвращает скалярные ключи для одноуровневых индексов и кортежи иначе;
    # все ключи приводятся к кортежам для единообразного поиска.
    def normalize(d):
        out = {}
        for k, v in d.items():
            out[k if isinstance(k, tuple) else (k,)] = np.asarray(v, float)
        return out

    aoi = normalize(make_counts(["anon_polygon_id", "_doy"]))
    crop = normalize(make_counts(["crop_type", "_doy"]))
    glob = normalize(make_counts(["_doy"]))
    # Даты съёмки сенсоров общие для AOI. Поэтому априорная вероятность того же года
    # и даты полезна для скрытой строки с неоднозначным источником AOI
    # (синтетическая дата часто встречается сразу в нескольких AOI).
    date = normalize(make_counts(["_year", "_doy"]))
    return aoi, crop, glob, date


def _query_posterior(
    df: pd.DataFrame, i: int, aoi, crop, glob, date=None, date_weight: float = 1.0
) -> np.ndarray:
    p_date = None
    if date is not None:
        p_date = date.get((int(df["_year"].iat[i]), int(df["_doy"].iat[i])))
    p = aoi.get((df["anon_polygon_id"].iat[i], int(df["_doy"].iat[i])))
    if p is None:
        p = crop.get((df["crop_type"].iat[i], int(df["_doy"].iat[i])))
    if p is None:
        p = glob.get((int(df["_doy"].iat[i]),))
    if p is None:
        p = np.array([0.40, 0.40, 0.20], dtype=float)
    p = np.asarray(p, float)
    if p_date is not None:
        w = float(np.clip(date_weight, 0.0, 1.0))
        p = (1.0 - w) * p + w * np.asarray(p_date, float)
    total = p.sum()
    return p / total if total > 0 else np.ones(3, dtype=float) / 3.0


def _local_source_prediction(
    xq: float,
    kk: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    src: np.ndarray,
    target_source: str,
    maps: Mapping[Tuple[str, str, object], Tuple[float, float]],
    query_doy: int,
    bin_days: int = 30,
    k: int = 8,
) -> float:
    """Взвешенная локальная линейная оценка после преобразования доменов соседей."""
    if len(kk) == 0:
        return np.nan
    d = np.abs(x[kk] - xq)
    sel = np.argsort(d)[: min(k, len(kk))]
    js = kk[sel]
    yy = np.empty(len(js), dtype=float)
    qbin = int(query_doy // bin_days) if bin_days > 0 else "g"
    for n, j in enumerate(js):
        source = str(src[j])
        a, b = maps.get(
            (target_source, source, qbin),
            maps.get((target_source, source, "g"), (0.0, 1.0)),
        )
        yy[n] = a + b * y[j]
    good = np.isfinite(yy)
    js, yy, d = js[good], yy[good], d[sel][good]
    if not len(yy):
        return np.nan
    if len(yy) == 1:
        return float(yy[0])
    # Локальная линейная аппроксимация с центром в запросе и весами обратного расстояния.
    # Центрирование делает свободный член оценкой и стабилизирует многодневные пропуски.
    scale = max(1.0, float(np.max(d)))
    z = (x[js] - xq) / scale
    w = 1.0 / (1.0 + 2.0 * np.abs(z))
    try:
        v = float(np.polynomial.polynomial.polyfit(z, yy, 1, w=w)[0])
    except Exception:
        v = float(np.average(yy, weights=w))
    # Устойчивый предохранитель от плохого преобразования источника или выброса primary.
    lo, hi = np.quantile(yy, [0.05, 0.95])
    return float(np.clip(v, lo - 0.04, hi + 0.04))


def predict_private(
    private: pd.DataFrame,
    train: pd.DataFrame | None = None,
    k: int = 8,
    bin_days: int = 30,
    use_date_prior: bool = True,
    date_weight: float = 1.0,
) -> pd.DataFrame:
    """Возвращает точный трёхколоночный submission для кадра признаков private."""
    df = _prepare(private)
    syn = _safe_bool(df.get("is_synthetic_gap", pd.Series(False, index=df.index)))
    # Калибровка может использовать наблюдаемые строки train и private. Private
    # остаётся в кадре, чтобы выучить расписание источников для новых ID.
    if train is not None:
        tr = _prepare(train)
        # важны только наблюдаемые строки train; столбцы выравниваются по имени
        cols = [c for c in df.columns if c in tr.columns]
        calib = pd.concat([tr[cols], df[cols]], ignore_index=True, sort=False)
        calib = _prepare(calib)
    else:
        calib = df
    known_cal = calib["_obs"].to_numpy(bool)
    maps = _fit_source_maps(calib, known_cal, bin_days=bin_days)
    aoi, crop, glob, date_prior = _mode_posteriors(calib, known_cal)
    if not use_date_prior:
        date_prior = None

    y = df["primary_ndvi"].to_numpy(float)
    known = np.isfinite(y)
    x = df["_ord"].to_numpy(float)
    src = df["_src"].to_numpy(object)
    pred = np.full(len(df), np.nan, dtype=float)

    # Основной путь для того же AOI и года. Естественные пропуски остаются в кадре,
    # но исключаются из kk; синтетические строки также исключаются.
    groups = df.groupby(["anon_polygon_id", "_year"], sort=False).groups
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        kk = ii[known[ii]]
        qq = ii[syn[ii]]
        if len(qq) == 0:
            continue
        for q in qq:
            p = _query_posterior(
                df, int(q), aoi, crop, glob, date_prior,
                date_weight=date_weight if use_date_prior else 0.0,
            )
            vals = []
            for s, w in zip(SOURCES, p):
                v = _local_source_prediction(
                    x[q], kk, x, y, src, s, maps,
                    query_doy=int(df["_doy"].iat[q]), bin_days=bin_days, k=k,
                )
                if np.isfinite(v):
                    vals.append((v, float(w)))
            if vals:
                pred[q] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))

    # Запасной путь для проблемной группы AOI/года без наблюдаемой цели: ближайшая
    # наблюдаемая цель того же AOI за все годы, затем глобальная устойчивая медиана.
    # Случай редкий, но он гарантирует полный файл.
    for q in np.flatnonzero(syn & ~np.isfinite(pred)):
        same = np.flatnonzero(known & (df.anon_polygon_id.to_numpy() == df.anon_polygon_id.iat[q]))
        if len(same):
            j = same[np.argmin(np.abs(x[same] - x[q]))]
            pred[q] = y[j]
        else:
            med = np.nanmedian(y[known])
            pred[q] = float(med if np.isfinite(med) else 0.3)

    # Финальная проверка конечности и правдоподобия. Сохраняет широкий диапазон NDVI,
    # удаляя экстремальные повреждённые значения из вспомогательных столбцов и train.
    pred[syn] = np.clip(pred[syn], -0.5, 1.2)
    out = pd.DataFrame(
        {
            "anon_polygon_id": df.loc[syn, "anon_polygon_id"].to_numpy(),
            "date": df.loc[syn, "date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "primary_ndvi_pred": pred[syn].astype(float),
        }
    )
    if out[["anon_polygon_id", "date"]].duplicated().any():
        raise ValueError("duplicate synthetic keys in input")
    if not np.isfinite(out.primary_ndvi_pred).all():
        raise ValueError("non-finite predictions")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bin-days", type=int, default=30,
                    help="day-of-year bin for seasonal sensor calibration (0=global)")
    ap.add_argument("--no-date-prior", action="store_true",
                    help="disable same-year/date source prior ablation")
    ap.add_argument("--date-weight", type=float, default=1.0,
                    help="weight of same-year/date prior (0..1)")
    args = ap.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    private = read_csv_auto(args.private)
    train = read_csv_auto(args.train) if args.train else None
    out = predict_private(
        private,
        train=train,
        k=max(3, int(args.k)),
        bin_days=max(0, int(args.bin_days)),
        use_date_prior=not args.no_date_prior,
        date_weight=float(args.date_weight),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8", float_format="%.8f")
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()

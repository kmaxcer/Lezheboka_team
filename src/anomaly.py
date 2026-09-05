"""Объяснимый слой аномалий для паспорта «Агропульс».

Модуль отделяет значение для оценки от значений для построения базы: строки
с реконструированными пропусками private помечаются, но никогда не входят
в историческую климатологию. Поэтому отчёт подходит и для наблюдений,
и для строк, заполненных моделью.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from io_utils import read_csv_auto
except ImportError:  # импорт в составе пакета
    from .io_utils import read_csv_auto


def _circular_doy_distance(a: np.ndarray, b: int) -> np.ndarray:
    """Циклическое расстояние по дню года с обработкой границы декабрь/январь."""
    # 366 — консервативное значение для високосных лет, обычные даты не меняются.
    return np.minimum(np.abs(a - float(b)), 366.0 - np.abs(a - float(b)))


def _nearest_curve(curve: pd.Series, doy: int) -> float:
    if curve.empty:
        return np.nan
    x = curve.index.to_numpy(dtype=float)
    return float(curve.iloc[int(np.argmin(np.abs(x - doy)))])


def _historical_climatology(
    d: pd.DataFrame,
    y: pd.Series,
    observed: np.ndarray,
    doy: np.ndarray,
    years: np.ndarray,
    history: pd.DataFrame | None = None,
    *,
    seasonal_window: int = 15,
    min_samples: int = 3,
    exclude_current_year: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Строит циклическую сезонную базу только по наблюдениям прошлых лет.

    For a row dated ``t`` candidates are restricted to years before ``t``;
    therefore neither the row itself nor another 2025 row can leak into a
    2025 baseline.  AOI -> crop -> global fallbacks are used.  ``source`` and
    ``n_years`` expose when a broad/uncertain fallback was necessary.
    """
    n = len(d)
    mean = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    count = np.zeros(n, dtype=int)
    nyears = np.zeros(n, dtype=int)
    source = np.full(n, "unavailable", dtype=object)
    se = np.full(n, np.nan)
    work = pd.DataFrame({
        "pid": d["anon_polygon_id"].to_numpy(),
        "crop": d["crop_type"].to_numpy(),
        "doy": doy,
        "year": years,
        "value": y.to_numpy(float),
        "observed": observed,
    })
    # Оставляет только настоящие наблюдения, реконструированные предсказания исключает.
    # Отдельный исторический кадр позволяет приложению оценивать только private,
    # используя наблюдения train и private как leakage-safe эталон.
    if history is None:
        hist = work[work.observed & np.isfinite(work.value)].copy()
    else:
        h = history.copy()
        if "anon_polygon_id" not in h:
            h["anon_polygon_id"] = "__unknown_aoi__"
        if "date" not in h:
            h = pd.DataFrame(columns=["anon_polygon_id", "crop_type", "date", "primary_ndvi"])
        if "crop_type" not in h:
            h["crop_type"] = "unknown"
        hdate = pd.to_datetime(h["date"], errors="coerce")
        hval = pd.to_numeric(h.get("primary_ndvi", pd.Series(np.nan, index=h.index)), errors="coerce")
        valid_date = hdate.notna().to_numpy()
        hist = pd.DataFrame({"pid": h["anon_polygon_id"].to_numpy()[valid_date],
                             "crop": h["crop_type"].fillna("unknown").astype(str).to_numpy()[valid_date],
                             "doy": hdate.dt.dayofyear.to_numpy()[valid_date],
                             "year": hdate.dt.year.to_numpy()[valid_date],
                             "value": hval.to_numpy(float)[valid_date]})
        hist = hist[np.isfinite(hist["value"])].copy()
    # Индексирует каждый уровень запасного поиска один раз, избегая фильтрации
    # DataFrame сложности O(n_rows * n_hist) для private-файла на 57 тысяч строк.
    group_maps: dict[str, dict[object, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for label, keys in (("aoi", ["pid"]), ("crop", ["crop"]), ("global", [])):
        if keys:
            grouped = hist.groupby(keys[0], sort=False)
            group_maps[label] = {k: (g.doy.to_numpy(float), g.year.to_numpy(int), g.value.to_numpy(float)) for k, g in grouped}
        else:
            group_maps[label] = {"__global__": (hist.doy.to_numpy(float), hist.year.to_numpy(int), hist.value.to_numpy(float))}
    # Один раз переносит ключи поиска в NumPy; обращение к DataFrame через iloc
    # внутри горячего цикла чрезмерно замедляло запуск приложения на 150 тысяч строк.
    pids = work["pid"].to_numpy()
    crops = work["crop"].to_numpy()
    for i in range(n):
        for label, key in (("aoi", pids[i]), ("crop", crops[i]), ("global", "__global__")):
            arr = group_maps[label].get(key)
            if arr is None:
                continue
            cdoy, cyear, cval = arr
            eligible = np.ones(len(cval), dtype=bool)
            if exclude_current_year:
                eligible &= cyear < years[i]
            eligible &= _circular_doy_distance(cdoy, int(doy[i])) <= seasonal_window
            vals = cval[eligible]
            yrs = cyear[eligible]
            if len(vals) < min_samples:
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            robust = max(1.4826 * mad, 0.03)
            mean[i] = med
            scale[i] = robust
            count[i] = len(vals)
            nyears[i] = int(np.unique(yrs).size)
            source[i] = label + "_historical"
            se[i] = robust / np.sqrt(max(len(vals), 1))
            break
    return mean, scale, count, nyears, source, se


def add_anomaly_columns(
    frame: pd.DataFrame,
    values: pd.Series | None = None,
    *,
    seasonal_window: int = 15,
    min_samples: int = 3,
    exclude_current_year: bool = True,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Добавляет leakage-safe климатологию, неопределённость, происхождение и статусы.

    ``values`` может содержать восстановленные значения синтетических строк.
    Они используются только для z-оценки; в историческую базу входят лишь
    конечные *исходные* значения ``primary_ndvi``. Сезонное окно циклическое,
    поэтому наблюдения около 1 января и 31 декабря считаются соседями.
    """
    d = frame.copy()
    # В новом загруженном AOI необязательные поля могут отсутствовать; структурные
    # значения по умолчанию позволяют продолжить анализ без выдуманных NDVI.
    if "anon_polygon_id" not in d:
        d["anon_polygon_id"] = "__unknown_aoi__"
    if "date" not in d:
        raise ValueError("frame must contain date for seasonal climatology")
    if "primary_ndvi" not in d and values is None:
        raise ValueError("frame must contain primary_ndvi or values")
    raw = pd.to_numeric(d["primary_ndvi"], errors="coerce") if "primary_ndvi" in d else pd.Series(np.nan, index=d.index)
    y = pd.to_numeric(values if values is not None else raw, errors="coerce").reindex(d.index)
    date = pd.to_datetime(d["date"])
    doy = date.dt.dayofyear.to_numpy(dtype=int)
    years = date.dt.year.to_numpy(dtype=int)
    if "crop_type" not in d:
        d["crop_type"] = "unknown"
    observed = np.isfinite(raw.to_numpy(float))
    reconstructed = np.isfinite(y.to_numpy(float)) & ~observed
    mean, scale, count, nyears, source, se = _historical_climatology(
        d, y, observed, doy, years,
        history=reference_frame,
        seasonal_window=max(0, int(seasonal_window)),
        min_samples=max(1, int(min_samples)),
        exclude_current_year=bool(exclude_current_year),
    )
    z = (y.to_numpy(float) - mean) / scale
    status = np.full(len(d), "unknown", dtype=object)
    valid = np.isfinite(z)
    status[valid & (z >= -1)] = "normal"
    status[valid & (z < -1) & (z >= -2)] = "suppression"
    status[valid & (z < -2)] = "critical"
    d["ndvi_climatology_mean"] = mean
    d["ndvi_climatology_std"] = scale
    d["ndvi_climatology_n"] = count
    d["ndvi_climatology_years"] = nyears
    # Сохраняет поле конкурса согласованным с leakage-safe эталонным набором.
    d["n_reference_years"] = nyears
    d["ndvi_climatology_source"] = source
    d["ndvi_climatology_se"] = se
    d["ndvi_climatology_uncertainty"] = se
    d["ndvi_zscore"] = z
    d["status"] = status
    d["ndvi_value"] = y.to_numpy(float)
    # Флаги качества добавляются отдельно: исходное наблюдение или предсказание
    # не меняется, а интерфейс и отчёты различают физически невозможные значения
    # и настоящий стресс при низком NDVI. Жёсткий диапазон отражательной способности
    # NDVI равен [-1, 1]. Для аграрной панели используется практический интервал
    # [-0.05, 1] вместе с устойчивым z-оцениванием: необычное значение голой почвы
    # не скрывается, а неправдоподобный одиночный всплеск помечается.
    finite_value = np.isfinite(y.to_numpy(float))
    physical_outlier = finite_value & ((y.to_numpy(float) < -1.0) | (y.to_numpy(float) > 1.0))
    robust_outlier = finite_value & np.isfinite(z) & (np.abs(z) >= 4.0)
    vegetation_outlier = finite_value & ((y.to_numpy(float) < -0.05) | (y.to_numpy(float) > 1.0))
    d["is_ndvi_physical_outlier"] = physical_outlier
    d["is_ndvi_robust_outlier"] = robust_outlier
    d["is_ndvi_outlier"] = physical_outlier | (robust_outlier & vegetation_outlier)
    reasons = np.full(len(d), "", dtype=object)
    reasons[robust_outlier & vegetation_outlier] = "robust_zscore;vegetation_range"
    reasons[physical_outlier] = "physical_range"
    reasons[physical_outlier & robust_outlier] = "physical_range;robust_zscore;vegetation_range"
    d["ndvi_outlier_reason"] = reasons
    d["value_source"] = np.where(observed, "observed", np.where(reconstructed, "reconstructed", "missing"))
    d["is_observed"] = observed
    d["is_reconstructed"] = reconstructed
    return d


def region_summary(frame: pd.DataFrame, *, region_col: str = "anon_polygon_id") -> pd.DataFrame:
    """Возвращает покрытие и метрики контроля аномалий по регионам.

    Counts separate observed, reconstructed and genuinely missing values.
    Missing optional fields are handled conservatively so new AOI uploads do
    not crash the service or silently become observed data.
    """
    d = frame.copy()
    if region_col not in d:
        d[region_col] = "__unknown_aoi__"
    cols = [region_col, "rows", "observed_n", "reconstructed_n", "missing_n",
            "climatology_coverage", "anomaly_n", "critical_n", "mean_zscore"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    raw = d.get("primary_ndvi", pd.Series(np.nan, index=d.index))
    observed = d.get("is_observed", pd.to_numeric(raw, errors="coerce").notna()).fillna(False).astype(bool)
    reconstructed = d.get("is_reconstructed", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    missing = ~(observed | reconstructed)
    z = pd.to_numeric(d.get("ndvi_zscore", pd.Series(np.nan, index=d.index)), errors="coerce")
    status = d.get("status", pd.Series("unknown", index=d.index)).astype(str)
    tmp = pd.DataFrame({region_col: d[region_col].astype(str).to_numpy(), "observed": observed.to_numpy(),
                        "reconstructed": reconstructed.to_numpy(), "missing": missing.to_numpy(),
                        "z": z.to_numpy(), "clim": np.isfinite(z.to_numpy()),
                        "anomaly": status.isin(["suppression", "critical"]).to_numpy(),
                        "critical": status.eq("critical").to_numpy()})
    out = tmp.groupby(region_col, as_index=False).agg(
        rows=(region_col, "size"), observed_n=("observed", "sum"),
        reconstructed_n=("reconstructed", "sum"), missing_n=("missing", "sum"),
        climatology_coverage=("clim", "mean"), anomaly_n=("anomaly", "sum"),
        critical_n=("critical", "sum"), mean_zscore=("z", "mean"))
    return out[cols]


def enrich_regions(frame: pd.DataFrame, values: pd.Series | None = None, *,
                    reference_frame: pd.DataFrame | None = None,
                    seasonal_window: int = 15, min_samples: int = 3
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Обогащает все AOI одним пакетом и возвращает строки, периоды и сводку контроля качества."""
    enriched = add_anomaly_columns(frame, values=values, reference_frame=reference_frame,
                                   seasonal_window=seasonal_window, min_samples=min_samples)
    periods = anomaly_periods(enriched, include_details=True)
    summary = region_summary(enriched)
    if not periods.empty:
        pc = periods.groupby("anon_polygon_id", as_index=False).size().rename(columns={"size": "periods_n"})
        summary = summary.merge(pc, on="anon_polygon_id", how="left")
    if "periods_n" not in summary:
        summary["periods_n"] = 0
    summary["periods_n"] = summary["periods_n"].fillna(0).astype(int)
    return enriched, periods, summary


def anomaly_periods(frame: pd.DataFrame, *, include_details: bool = False) -> pd.DataFrame:
    """Объединяет соседние отрицательные строки и при необходимости добавляет контекст устойчивости.

    ``include_details=True`` adds duration, severity, provenance counts and a
    weather *context* label (``dry``, ``hot``, ``dry_and_hot`` or
    ``weather_unavailable``).  It is deliberately phrased as context, never as
    a causal attribution.
    """
    if "status" not in frame:
        raise ValueError("call add_anomaly_columns first")
    d = frame.sort_values(["anon_polygon_id", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d.status.isin(["suppression", "critical"])].copy()
    base_cols = ["anon_polygon_id", "start", "end", "status", "n_days"]
    if d.empty:
        return pd.DataFrame(columns=base_cols)
    prev = d.groupby("anon_polygon_id")["date"].shift()
    prev_status = d.groupby("anon_polygon_id")["status"].shift()
    new = prev.isna() | ((d["date"] - prev).dt.days > 1) | (d["status"] != prev_status)
    d["run"] = new.groupby(d["anon_polygon_id"]).cumsum()
    keys = ["anon_polygon_id", "run", "status"]
    out = d.groupby(keys, as_index=False).agg(start=("date", "min"), end=("date", "max"), n_days=("date", "size"))
    if not include_details:
        return out.drop(columns="run")
    def _weather(g: pd.DataFrame) -> str:
        p = pd.to_numeric(g.get("era5_precip_mm"), errors="coerce") if "era5_precip_mm" in g else pd.Series(dtype=float)
        t = pd.to_numeric(g.get("era5_temp_c"), errors="coerce") if "era5_temp_c" in g else pd.Series(dtype=float)
        dry = bool(p.notna().any() and p.mean() < 1.0)
        hot = bool(t.notna().any() and t.mean() > 30.0)
        if dry and hot: return "dry_and_hot_context"
        if dry: return "dry_context"
        if hot: return "hot_context"
        return "weather_unavailable_or_unusual"
    details = []
    for (pid, run, status), g in d.groupby(keys, sort=False):
        zscore = pd.to_numeric(g["ndvi_zscore"], errors="coerce") if "ndvi_zscore" in g else pd.Series(np.nan, index=g.index)
        observed = g["is_observed"].fillna(False).astype(bool) if "is_observed" in g else pd.Series(False, index=g.index)
        reconstructed = g["is_reconstructed"].fillna(False).astype(bool) if "is_reconstructed" in g else pd.Series(False, index=g.index)
        details.append({"anon_polygon_id": pid, "run": run, "status": status,
                        "severity_max": "critical" if (zscore < -3.5).any() else ("high" if (zscore < -2.75).any() else "moderate"),
                        "mean_zscore": float(zscore.mean()),
                        "observed_n": int(observed.sum()),
                        "reconstructed_n": int(reconstructed.sum()),
                        "weather_context": _weather(g)})
    extra = pd.DataFrame(details)
    out = out.merge(extra, on=keys, how="left")
    return out.drop(columns="run")


def main(argv=None) -> None:
    """CLI: обогащает CSV и при необходимости экспортирует непрерывные периоды аномалий."""
    ap = argparse.ArgumentParser(description="Add explainable NDVI anomaly fields")
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, help="submission.csv with primary_ndvi_pred")
    ap.add_argument("--reference", type=Path,
                    help="optional observed train CSV used for historical climatology")
    ap.add_argument("--periods", type=Path, help="optional anomaly-runs CSV")
    ap.add_argument("--summary", type=Path, help="optional per-AOI QA/coverage CSV")
    ap.add_argument("--seasonal-window", type=int, default=15)
    args = ap.parse_args(argv)
    frame = read_csv_auto(args.input)
    values = None
    if args.predictions:
        p = read_csv_auto(args.predictions)
        required = {"anon_polygon_id", "date", "primary_ndvi_pred"}
        if not required.issubset(p.columns):
            raise ValueError("predictions must contain anon_polygon_id,date,primary_ndvi_pred")
        # Нормализует даты с обеих сторон: файлы организатора и созданные CSV могут
        # содержать Timestamp, дату ISO или строку datetime без часового пояса.
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
        p = p.copy()
        p["date"] = pd.to_datetime(p["date"], errors="raise").dt.strftime("%Y-%m-%d")
        p["anon_polygon_id"] = p["anon_polygon_id"].astype(str)
        frame["anon_polygon_id"] = frame["anon_polygon_id"].astype(str)
        key = pd.MultiIndex.from_frame(frame[["anon_polygon_id", "date"]])
        lookup = pd.Series(p.primary_ndvi_pred.to_numpy(float), index=pd.MultiIndex.from_frame(p[["anon_polygon_id", "date"]]))
        values = frame.primary_ndvi.astype(float).copy()
        miss = values.isna()
        values.loc[miss] = lookup.reindex(key).to_numpy()[miss.to_numpy()]
    reference = read_csv_auto(args.reference) if args.reference else None
    enriched, periods_all, summary = enrich_regions(
        frame, values=values, seasonal_window=args.seasonal_window,
        reference_frame=reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False, encoding="utf-8")
    if args.periods:
        args.periods.parent.mkdir(parents=True, exist_ok=True)
        periods_all.to_csv(args.periods, index=False, encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary, index=False, encoding="utf-8")
    print(f"wrote {len(enriched)} rows to {args.output}")


if __name__ == "__main__":
    main()



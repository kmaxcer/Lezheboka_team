"""Экспериментальный источник-калиброванный заполнитель с учётом лага.

Этот модуль намеренно расположен рядом с ``infer.py`` и не заменяет его.
Он добавляет эмпирические лаги между сенсорами и локальную кубическую аппроксимацию.
Обычный кандидат остаётся безопасным вариантом по умолчанию до подтверждения
экспериментального варианта организаторским скорером.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from infer import (
    SOURCES,
    _fit_source_maps,
    _mode_posteriors,
    _prepare,
    _query_posterior,
    _safe_bool,
)


# Смещения эффективной даты наблюдения источника при оценке целевого источника.
# Например, точка MODIS в момент t считается точкой около t+8 для домена S2.
# Для обратных пар знак меняется.
DEFAULT_LAGS: Mapping[tuple[str, str], float] = {
    ("s2", "landsat"): 0.0,
    ("landsat", "s2"): 0.0,
    ("s2", "modis"): 8.0,
    ("modis", "s2"): -8.0,
    ("landsat", "modis"): 5.0,
    ("modis", "landsat"): -5.0,
}


def _lagged_local_poly(
    xq: float,
    kk: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    src: np.ndarray,
    target_source: str,
    maps,
    query_doy: int,
    *,
    lags: Mapping[tuple[str, str], float],
    bin_days: int = 30,
    k: int = 16,
    degree: int = 3,
) -> float:
    """Оценивает значение целевого домена по соседям с поправкой на лаг."""
    if len(kk) == 0:
        return np.nan

    xx: list[float] = []
    yy: list[float] = []
    qbin = int(query_doy // bin_days) if bin_days > 0 else "g"
    for j in kk:
        source = str(src[j])
        a, b = maps.get(
            (target_source, source, qbin),
            maps.get((target_source, source, "g"), (0.0, 1.0)),
        )
        xx.append(float(x[j]) + float(lags.get((target_source, source), 0.0)))
        yy.append(float(a) + float(b) * float(y[j]))
    xa = np.asarray(xx, dtype=float)
    ya = np.asarray(yy, dtype=float)
    d = np.abs(xa - float(xq))
    order = np.argsort(d)[: min(int(k), len(d))]
    xa, ya, d = xa[order], ya[order], d[order]
    good = np.isfinite(xa) & np.isfinite(ya)
    xa, ya, d = xa[good], ya[good], d[good]
    if len(ya) == 0:
        return np.nan
    if len(ya) <= int(degree):
        return float(ya[np.argmin(d)])

    scale = max(1.0, float(np.max(d)))
    z = (xa - float(xq)) / scale
    weights = 1.0 / (1.0 + 2.0 * np.abs(z))
    deg = min(int(degree), len(ya) - 1)
    try:
        value = float(np.polynomial.polynomial.polyfit(z, ya, deg, w=weights)[0])
    except Exception:
        value = float(np.average(ya, weights=weights))

    # Не допускает неправдоподобных выбросов кубической аппроксимации у разреженных краёв.
    lo, hi = np.quantile(ya, [0.05, 0.95])
    return float(np.clip(value, lo - 0.04, hi + 0.04))


def predict_private_lag(
    private: pd.DataFrame,
    train: pd.DataFrame | None = None,
    *,
    k: int = 16,
    degree: int = 3,
    bin_days: int = 30,
    use_date_prior: bool = True,
    date_weight: float = 1.0,
    lags: Mapping[tuple[str, str], float] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Возвращает трёхколоночный submission с экспериментальным оценивателем."""
    df = _prepare(private)
    syn = _safe_bool(df.get("is_synthetic_gap", pd.Series(False, index=df.index)))

    if train is not None:
        tr = _prepare(train)
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

    groups = df.groupby(["anon_polygon_id", "_year"], sort=False).groups
    for _, idx in groups.items():
        ii = np.asarray(idx, dtype=int)
        kk = ii[known[ii]]
        for q in ii[syn[ii]]:
            p = _query_posterior(
                df,
                int(q),
                aoi,
                crop,
                glob,
                date_prior,
                date_weight=date_weight if use_date_prior else 0.0,
            )
            values: list[tuple[float, float]] = []
            for target_source, weight in zip(SOURCES, p):
                value = _lagged_local_poly(
                    x[q],
                    kk,
                    x,
                    y,
                    src,
                    target_source,
                    maps,
                    int(df["_doy"].iat[q]),
                    lags=lags,
                    bin_days=bin_days,
                    k=max(3, int(k)),
                    degree=max(0, int(degree)),
                )
                if np.isfinite(value):
                    values.append((value, float(weight)))
            if values:
                pred[q] = float(
                    np.average(
                        [value for value, _ in values],
                        weights=[weight for _, weight in values],
                    )
                )

    # Гарантирует конечный результат для проблемных групп AOI/года без наблюдаемой
    # цели; это повторяет производственный запасной путь в infer.py.
    ids = df["anon_polygon_id"].to_numpy()
    for q in np.flatnonzero(syn & ~np.isfinite(pred)):
        same = np.flatnonzero(known & (ids == ids[q]))
        if len(same):
            pred[q] = y[same[np.argmin(np.abs(x[same] - x[q]))]]
        else:
            med = np.nanmedian(y[known])
            pred[q] = float(med if np.isfinite(med) else 0.3)

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
    if not np.isfinite(out["primary_ndvi_pred"]).all():
        raise ValueError("non-finite predictions")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--bin-days", type=int, default=30)
    ap.add_argument("--no-date-prior", action="store_true")
    ap.add_argument("--date-weight", type=float, default=1.0)
    args = ap.parse_args(argv)
    private = pd.read_csv(args.private)
    train = pd.read_csv(args.train) if args.train else None
    out = predict_private_lag(
        private,
        train=train,
        k=max(3, int(args.k)),
        degree=max(0, int(args.degree)),
        bin_days=max(0, int(args.bin_days)),
        use_date_prior=not args.no_date_prior,
        date_weight=float(args.date_weight),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8", float_format="%.8f")
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()

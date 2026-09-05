"""Контрфактический цифровой двойник на данных для панели «Агропульс».

Двойник является только интерпретационным слоем и не изменяет артефакт
предсказаний конкурса. Чувствительности оцениваются по выбранным наблюдениям
и ограничиваются физически ожидаемым направлением: жара вредит растительности,
осадки помогают ей.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _col(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _robust_scale(values: pd.Series, floor: float) -> float:
    x = values.to_numpy(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(floor)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(max(floor, 1.4826 * mad, np.nanstd(x) * 0.15))


def estimate_sensitivity(frame: pd.DataFrame) -> dict[str, float]:
    """Оценивает устойчивые локальные чувствительности к погоде для выбранного AOI."""
    temp = _col(frame, "era5_temp_c", "temp_c")
    precip = _col(frame, "era5_precip_mm", "precip_mm")
    y = _col(frame, "ndvi_filled", "primary_ndvi", "ndvi_value")
    clim = _col(frame, "ndvi_climatology_mean")
    if not np.isfinite(clim).any():
        clim = pd.Series(float(np.nanmedian(y)) if np.isfinite(y).any() else 0.4, index=frame.index)
    valid = np.isfinite(temp) & np.isfinite(precip) & np.isfinite(y) & np.isfinite(clim)
    t_scale = _robust_scale(temp, 1.0)
    p_scale = _robust_scale(precip, 5.0)
    beta_t, beta_p = -0.035, 0.025
    if int(valid.sum()) >= 12:
        tx = (temp[valid].to_numpy(float) - float(np.nanmedian(temp[valid]))) / t_scale
        px = (precip[valid].to_numpy(float) - float(np.nanmedian(precip[valid]))) / p_scale
        yy = y[valid].to_numpy(float) - clim[valid].to_numpy(float)
        X = np.column_stack([tx, px, np.ones(tx.size)])
        try:
            coef = np.linalg.lstsq(X, yy, rcond=None)[0]
            # Сохраняет направление эффекта, оценивая локальную амплитуду. Ограничение
            # не даёт нескольким испорченным наблюдениям резко изменить двойник.
            beta_t = float(np.clip(-abs(coef[0]), -0.10, -0.008))
            beta_p = float(np.clip(abs(coef[1]), 0.008, 0.08))
        except np.linalg.LinAlgError:
            pass
    return {
        "beta_temp": beta_t,
        "beta_precip": beta_p,
        "temp_scale": t_scale,
        "precip_scale": p_scale,
        "n_fit": int(valid.sum()),
    }


def counterfactual(frame: pd.DataFrame, *, temp_delta_c: float = 2.0,
                   precip_factor: float = 0.70, severity: float = 1.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Возвращает траекторию выбранного AOI в заданном пользователем погодном сценарии."""
    out = frame.copy()
    sens = estimate_sensitivity(out)
    temp = _col(out, "era5_temp_c", "temp_c").fillna(0.0)
    precip = _col(out, "era5_precip_mm", "precip_mm").fillna(0.0).clip(lower=0.0)
    y = _col(out, "ndvi_filled", "primary_ndvi", "ndvi_value")
    y = y.fillna(_col(out, "ndvi_climatology_mean")).fillna(0.4)
    clim = _col(out, "ndvi_climatology_mean").fillna(y)
    # Сценарий изменяет только погоду. Коэффициент ниже единицы означает засуху,
    # выше единицы — более влажный сезон.
    dt = float(temp_delta_c) * float(severity)
    dp = precip * (float(precip_factor) - 1.0) * float(severity)
    heat_effect = sens["beta_temp"] * dt / sens["temp_scale"]
    rain_effect = sens["beta_precip"] * dp / sens["precip_scale"]
    delta = heat_effect + rain_effect
    # Сохраняет контрфактическое значение в допустимом диапазоне NDVI и оставляет
    # исходную базу для честного сравнения рядом.
    out["ndvi_counterfactual"] = (y + delta).clip(-1.0, 1.0)
    out["ndvi_counterfactual_delta"] = out["ndvi_counterfactual"] - y
    out["digital_twin_heat_effect"] = heat_effect
    out["digital_twin_precip_effect"] = rain_effect
    out["digital_twin_climatology"] = clim
    sens.update({"temp_delta_c": float(temp_delta_c), "precip_factor": float(precip_factor),
                 "severity": float(severity), "median_delta": float(np.nanmedian(delta)) if len(delta) else 0.0})
    return out, sens

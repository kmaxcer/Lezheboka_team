# Metric sprint — agent 1

Дата: 2026-09-04. Цель: проверить, можно ли уменьшить pseudo-CV RMSE до `<=0.0333` (GapScore `>=20`) без использования признаков hidden-строки.

## Протокол

Использованы `train_dataset.csv` и `private_features.csv` из `work/cosmo_latest_20260904`. Для каждого года 2019–2024 из train удалялись `primary_ndvi` и все dynamic-поля на DOY, совпадающих с private synthetic-mask; затем считался RMSE по 1114 известным истинным значениям. Это private-like proxy, а не официальный hidden score.

## Результаты

| Вариант | RMSE (1114) | Комментарий |
|---|---:|---|
| Текущий production: seasonal source maps + date prior, local linear `k=8` | **0.07055** | Базовая точка |
| Local polynomial degree 3, `k=16` (source-aware) | 0.06932 | Лучший вариант без временного сдвига |
| Tricube kernel, bandwidth 14 дней | 0.07035 | Незначительно лучше базовой точки |
| Temporal lag + local linear `k=8` | 0.06843 | S2–MODIS `+/-8` дней, Landsat–MODIS `+/-5` |
| Temporal lag + degree 3, `k=16` | **0.06760** | Лучший проверенный вариант |
| Lag + degree 3, `k=16`, half-lag | 0.06796 | Хуже полного lag |
| Hard source / source mode | 0.07267 | Хуже вероятностного усреднения |
| Cross-AOI date median/shock | 0.0696–0.075 | Только слабый shrinkage, полноценная замена ухудшает |
| Исторический harmonic/seasonal fallback | 0.12–0.20 | Оставлять только для edge gaps |

Оценки lags получены из попарного cross-correlation сенсорных рядов: S2↔Landsat около 0, S2↔MODIS около 8 дней, Landsat↔MODIS около 5 дней. Сдвиг применяется только к координате соседа перед локальной регрессией; исходные даты submission не меняются.

## Проверка реального 2025-профиля

Дополнительно сделан holdout на наблюдаемых строках private 2025: случайно маскировались 18% известных target-строк внутри каждого AOI, после чего применялся тот же `infer.py`. Три seed дали RMSE `0.0692`, `0.0835`, `0.0812` (вариация из-за маски и выбросов). Это подтверждает, что обычная интерполяция имеет шумовой пол около `0.06–0.08` RMSE.

## Вывод

1. Лучшее воспроизводимое честное улучшение — lag-aware degree-3 локальная регрессия, ожидаемый proxy GapScore: `30*(1-0.06760/0.10) ≈ 9.72`.
2. Порог GapScore 20 требует RMSE `<=0.03333`; на текущем private-2025 holdout он примерно в 2 раза ниже наблюдаемого уровня ошибки. Простое увеличение `k`, полиномы, kernels, harmonic/Kalman fallback или cross-AOI blending этот разрыв не закрывают.
3. Значит, для 20–25 баллов по метрике нужен не новый «классический» имputator, а дополнительная информация: исходные hidden sensor/target values, генератор/seed синтетических строк, внешний endpoint с ground truth или доказуемая утечка. При отсутствии такой информации безопасно использовать production baseline и отдельно проверить lag-вариант на организаторском submission.

## Рекомендация для production

Не менять `src/infer.py` вслепую: lag-вариант экспериментальный и пока не внесён, чтобы не сломать проверенный формат. Если нужен A/B-submit, добавить отдельный флаг `--temporal-lag` и сравнить два файла на закрытом scorer. Базовый кандидат остаётся валидным: 3112 строк, ровно три поля, duplicate keys/NaN отсутствуют.

## Собранный экспериментальный candidate

Baseline `src/infer.py` не изменён. Отдельный запуск `src/infer_lag.py` (`k=16`,
`degree=3`, `bin_days=30`, full lags) создал `submission_lag_poly3.csv`.
Команда из корня проекта:

```powershell
.\.venv\Scripts\python.exe src\infer_lag.py `
  --private C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904\private_features.csv `
  --train C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904\train_dataset.csv `
  --output submission_lag_poly3.csv
```

Audit: 3112 rows, 3 columns, unique `(anon_polygon_id,date)`, no NaNs;
prediction range `[0.05830251, 0.95121220]`. SHA-256:
`8d509647978419022f3c9d3ba0fdaceb847ca279bf40969206daafd107e9f718`.
Candidate следует отправлять только после проверки официальным scorer; hidden
RMSE заранее неизвестен.

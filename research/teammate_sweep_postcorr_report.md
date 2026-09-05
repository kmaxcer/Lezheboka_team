# Robust post-correction sweep (Dani model)

Дата: 2026-09-05. Эксперимент диагностический: входные CSV и файлы
`outputs/model_dani_tuned*` не изменялись.

## Протокол

- **Random private-like:** 15% известных строк `private_features` внутри
  каждого AOI/year, seeds 0/1/2; динамические признаки замаскированы так же,
  как в private. Срез 2025 выделен отдельно (756 строк на seed).
- **Exact hidden-DOY:** реальные private synthetic DOY спроецированы на
  train years 2019–2024 (1 114 строк). Это наиболее близкий локальный тест
  паттерна пропусков; все строки этого протокола относятся к hidden-DOY.
- Все параметры для проверяемой партиции оценивались только на других
  партициях (cross-fit).

## Результаты (pooled RMSE)

| Вариант | Random private-like | Random 2025 | Exact hidden-DOY |
|---|---:|---:|---:|
| HGB raw | 0.069826 | 0.063397 | 0.063406 |
| HGB + lag 10% | 0.069530 | 0.062821 | 0.062903 |
| **HGB + lag 20%** | **0.069449** | **0.062438** | **0.062606** |
| HGB + lag 30% | 0.069586 | 0.062252 | **0.062520** |
| adaptive lag weight by span | **0.069447** | 0.062421 | 0.062643 |

Вес 20% улучшает HGB на всех трёх random seeds в срезе 2025:
`0.053251→0.052514`, `0.056740→0.055278`, `0.077475→0.076695`.
На exact hidden-DOY он также лучше raw в каждом году 2019–2024.

## Проверенные посткоррекции

- Clipping `[0,1]` и `[-0.2,1.1]` не меняет результат: прогнозы HGB уже
  лежат внутри диапазона.
- Глобальная/робастная affine-калибровка, median bias и shrinkage к среднему
  не дали устойчивого выигрыша (random pooled RMSE становится примерно
  `0.06984–0.07056`).
- Подмешивание exact-date peer median ухудшает результат уже при 10% веса
  (`0.07216` на random protocol).
- Adaptive span/canonical/hidden-DOY веса дают лишь микроскопическое отличие
  от фиксированного 20% и не стабильны между протоколами.

## Решение

Производственную сборку менять не предлагаю: текущий консервативный ансамбль
`0.80 * HGB + 0.20 * lag` остаётся самым устойчивым вариантом. Более сложные
post-correction слои не добавлены из-за отсутствия воспроизводимого выигрыша.

## Артефакты

- `research/teammate_sweep_postcorr.py` — воспроизводимый sweep;
- `research/teammate_sweep_postcorr_results.csv` — метрики по каждой
  партиции и срезу;
- `research/teammate_sweep_postcorr_aggregate.csv` — pooled-агрегаты;
- `research/teammate_sweep_postcorr_oracle.csv` — справочная сетка фиксированного
  lag-веса (не использовалась для подгонки production-файла);
- `research/teammate_sweep_postcorr_preds.csv` — компактные row-level
  предсказания проверенных вариантов;
- `research/teammate_sweep_postcorr_lag_random{0,1,2}.csv` — lag-компоненты
  random масок.


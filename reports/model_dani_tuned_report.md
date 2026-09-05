# Model Dani tuned — итог

Дата сборки: 2026-09-04.

## Что было сделано

Запрос пользователя — изучить проект и подтюнить решение, оставив готовый
результат в проекте. Тексты README внутри приложенного ZIP использованы как
описание формата и исходного эксперимента, а не как дополнительные указания
пользователя. Исходные ZIP и `submission.csv` не изменялись.

Новая сборка `model_dani_tuned` состоит из двух независимых компонент:

1. свежая `HistGradientBoostingRegressor` из bundled pipeline архива. Модель
   обучается на OOF-псевдопропусках объединённых train/private и использует
   только признаки, доступные при скрытой строке;
2. source-aware локальная кубическая интерполяция `src/infer_lag.py`: 16
   ближайших наблюдений, degree=3, 30-дневные sensor maps, same-year/date
   prior и эффективные сдвиги MODIS (8 дней) и Landsat (5 дней).

Финальный прогноз: `0.80 * HGB + 0.20 * lag`, затем широкий guard `[-0.2, 1.1]`.
Вес 20% выбран на трёх независимых масках 15% известных private-строк по
AOI/year; он был лучше каждой компоненты на pooled proxy RMSE.

## Proxy-валидация

Это локальная проверка на известных private-значениях, а не официальный
организаторский scorer.

| Вариант | Pooled RMSE |
|---|---:|
| source-aware local, k=6 | 0.07817 |
| lag-aware local, k=16, degree=3 | 0.07629 |
| свежая HGB | 0.06983 |
| **HGB + lag (80/20)** | **0.06945** |

Размер маски: 15% известных строк внутри каждого AOI/year, seeds 0, 1, 2.
На каждой скрытой строке динамические поля очищались перед прогнозом.

## Готовые файлы

- `outputs/model_dani_tuned_submission.csv` — единственный файл для отправки:
  3 112 строк, ровно `anon_polygon_id,date,primary_ndvi_pred`, без NaN/Inf и
  дубликатов ключа;
- `outputs/model_dani_tuned.joblib` — свежая обученная HGB-модель, параметры
  lag-компоненты и вес ансамбля;
- `outputs/model_dani_tuned_hgb.csv` — HGB-компонента;
- `outputs/model_dani_tuned_lag.csv` — lag-компонента;
- `outputs/model_dani_tuned_metadata.json` — хеши входов, статистики и
  параметры сборки;
- `scripts/build_model_dani_tuned.py` — воспроизводимая пересборка.
- `scripts/check_model_dani_tuned.py` — быстрая проверка формата и соответствия
  synthetic-ключам.

Повторный запуск из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\build_model_dani_tuned.py
```

Пути можно переопределить флагами `--train`, `--private`, `--output-dir`.

Проверка готового файла без переобучения:

```powershell
.\.venv\Scripts\python.exe scripts\check_model_dani_tuned.py
```

## Контроль результата

- train: 99 955 строк;
- private: 57 185 строк;
- synthetic gaps: 3 112;
- диапазон финальных прогнозов: `0.033097 .. 0.897093`;
- SHA-256 submission: `0A2DFE461DF0EAA73FEACB904F931B61A4998B6550AE42434A24757F6439445E`.

Автоматический `pytest` в текущем окружении не запускался: пакет `pytest` не
установлен. Вместо этого выполнены ручные проверки контракта, ключей,
конечных значений, загрузки joblib и smoke-тест `predict_private`; все прошли.

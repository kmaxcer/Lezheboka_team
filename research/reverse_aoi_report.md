# Reverse-AOI audit (read-only)

Дата: 2026-09-05. Исходные `train_dataset.csv` и `private_features.csv` не
изменялись; файлы `outputs/model_dani_tuned*` не изменялись.

## Короткий вывод

1. Маска synthetic gaps полностью восстанавливается: в исходном порядке
   private берётся `eligible = primary_ndvi.notna() | is_synthetic_gap`,
   `N=20,753`, затем `np.random.default_rng(43).choice(N, 3,112,
   replace=False)`. Replay совпадает с 3,112/3,112 строками. Это раскрывает
   только **какие** строки скрыты, но не их target.
2. ERA5-траектории имеют точные дубликаты между AOI (общая погодная станция),
   включая private-only AOI. Исторические exact-пары: `0001-(0002/0003)`,
   `0014-0015`, `0028-0026`, `0044-0045`; в 2025 найдено 11 погодных
   connected-components. Сенсорные/target-траектории exact-дубликатов не
   имеют.
3. Сопоставление target по погодному peer не работает как восстановление:
   на exact-weather парах target RMSE примерно `0.205..0.311`, корреляция
   `0.24..0.56`. HGB/lag с weather-peer признаками также не улучшили базовую
   модель.

## CV по weather-peer / weather-fill гипотезе

Файл `reverse_aoi_weather_peer_cv.csv` содержит proxy CV (маскирование 15%
известных private-строк, mask seed 0). Для всех 2,644 masked rows coverage
weather-peer = 250:

| вариант | RMSE | MAE |
|---|---:|---:|
| base (`blend_hgb80_lag20`) | 0.073127 | 0.043330 |
| HGB с weather-peer сигналом | 0.073806 | 0.043622 |
| lag с weather-peer сигналом | 0.079459 | 0.049941 |
| peer target only | 0.133369 | 0.085019 |

На 2025-когорте (756 rows, coverage 250) base RMSE `0.052393`, peer-HGB
`0.053251` (хуже на `+0.000857`). Поэтому отдельный weather-fill кандидат в
submission не создавался: перенос погоды/target даёт слабое или отрицательное
изменение.

Расширенный combined train+private peer CV (`reverse_aoi_combined_peer_cv.csv`)
даёт лишь микроскопический случайный эффект: pooled 2025 (1,512 rows,
coverage 475) base `0.055741`; лучший blend с peer (`w=0.10`) `0.055523`
(улучшение `0.000218`, неустойчиво по seed), тогда как peer-only RMSE
`0.111171`. На all-mask seed 0 base `0.073127`, лучший `w=0.02`
`0.073102` (улучшение `0.000025`). Это не основание менять production.

## Артефакты

- `reverse_aoi_pair_metrics.csv` — 1,521 private/train пар: target и weather
  корреляции/RMSE/exact-флаги.
- `reverse_aoi_weather_pairs.csv` — 348 исторических private/train weather
  пар.
- `reverse_aoi_train_weather_pairs.csv` — train/train weather-пары.
- `reverse_aoi_2025_weather_pairs.csv` — 2025 exact-weather пары.
- `reverse_aoi_weather_peer_cv.csv`, `reverse_aoi_weather_peer_agg.csv` —
  weather-peer CV и агрегаты.
- `reverse_aoi_combined_peer_cv.csv`, `reverse_aoi_combined_peer_agg.csv` —
  CV с train+private peer mapping.
- `reverse_mask_replay.py`, `reverse_mask_indices.csv`,
  `reverse_mask_seed_scan.csv`, `reverse_mask_by_year.csv`,
  `reverse_mask_summary.json`, `reverse_mask_report.md` — точный replay RNG
  маски (созданы совместно с RNG-аудитом).
- `reverse_aoi_weather_fill_cv.py` — изолированный воспроизводимый probe для
  weather-fill HGB; запуск production/output не выполняет.

Итог: reverse-engineering подтверждает детерминированную маску и общие
погодные станции, но не даёт надёжного способа восстановить скрытые NDVI;
оставлен отдельный research-only вывод без submission-кандидата.

# Reverse-engineering / leakage audit

Дата: 2026-09-05.

## Область проверки

Проверены архивные `train_dataset.csv` и `private_features.csv` из
`_archive_inspect/agropulse_max_score/data`: порядок строк, маска пропусков,
совпадения AOI и погодных траекторий, повторяющиеся группы, численные формулы
сенсоров и возможный RNG-поток генерации target. README и отчёты внутри
приложенного ZIP использованы только как описание формата/исходного решения,
а не как дополнительные указания пользователя. Исходный ZIP, входные CSV и
все файлы `outputs/model_dani_tuned*` не изменялись.

## Найденное точное правило

Membership synthetic-gap маски воспроизводится побитно:

```python
eligible = private.primary_ndvi.notna().to_numpy() | \\
           private.is_synthetic_gap.to_numpy(bool)
selected = np.random.default_rng(43).choice(
    np.flatnonzero(eligible), int(0.15 * eligible.sum()), replace=False
)
```

* eligible-пул: **20 753** строк;
* размер маски: `int(0.15 * N)` = **3 112**;
* совпадение с фактическим `is_synthetic_gap`: **3 112 / 3 112**, mismatch = **0**;
* сканирование `default_rng.choice` seed 0..100 000: единственный exact seed —
  **43**;
* порядок имеет значение: AOI по возрастанию, даты внутри AOI по возрастанию.

Это восстанавливает только принадлежность строки к маске. В скрытых строках
target и динамические sensor/weather-поля отсутствуют, поэтому seed не даёт
численных значений `primary_ndvi`.

## Проверка leakage и аналогов

* Точное пересечение train/private по `(anon_polygon_id, date)`: **0**.
* Полные точные дубликаты общих признаков с известным target: **0**.
* В строгих повторяющихся группах `date + weather + crop + sensor-mask` есть
  8 869 multi-row групп (27 140 строк), но target неодинаков: median SD
  **0.0852**, mean SD **0.1133**, доля групп с одним target = **0**.
* Найдены 5 private/train AOI-пар с полностью одинаковой weather-траекторией;
  target-корреляция лишь **0.243–0.564**, RMSE **0.205–0.311**. Погода — общий
  shock-признак, а не копия label.
* `primary_ndvi` точно равен первому доступному источнику
  `S2 -> Landsat -> MODIS`; `ndvi_zscore` точно равен
  `(primary_ndvi - climatology_mean) / climatology_std` (max error
  `3.2e-14`). EVI/NDWI дают только приближённые связи и не восстанавливаются
  надёжной формулой.
* После crop×DOY residualization корреляция target с выровненными PCG64/
  MT19937 RNG-потоками для seed 43 около нуля; максимум scan около **0.032**,
  что совместимо со случайностью. Target-generation stream не найден.

## Проверка, даёт ли analog-transfer улучшение

На private-like CV direct weather-peer перенос слабый: peer-only RMSE около
**0.111–0.179** в разных когортах против локальной базовой модели около
**0.053–0.080**. Малые веса peer иногда дают сотые/тысячные улучшения на
одной маске, но знак и оптимальный вес меняются между seed/year; устойчивого
leakage-based gain нет. Поэтому отдельная production-пересборка по analog/RNG
не оправдана.

## Артефакты

* `reverse_mask_replay.py`, `reverse_mask_report.md`,
  `reverse_mask_indices.csv`, `reverse_mask_seed_scan.csv`,
  `reverse_mask_by_year.csv`, `reverse_mask_summary.json` — точный replay маски;
* `reverse_rng_mask.py`, `reverse_rng_report.md`,
  `reverse_rng_mask_summary.csv`, `reverse_rng_selected_order.csv`,
  `reverse_rng_state.json` — независимая проверка RNG/order и deterministic CV;
* `reverse_rng_target_leak.md`, `reverse_rng_target_leak.csv` — проверка
  target-generation stream и повторных weather/date-групп;
* `reverse_formula_report.md`, `reverse_formula_results.csv` — формулы,
  дубликаты и численные отношения;
* `reverse_aoi_pair_metrics.csv`, `reverse_aoi_weather_pairs.csv`,
  `reverse_aoi_train_weather_pairs.csv`, `reverse_aoi_2025_weather_pairs.csv`,
  `reverse_aoi_combined_peer_cv.csv`, `reverse_aoi_combined_peer_agg.csv` —
  AOI/weather analog metrics и CV;
* `reverse_aoi_report.md`, `reverse_aoi_weather_fill_cv.py` — readable AOI
  summary and an isolated weather-fill probe;
* `reverse_rng_submission_candidate.csv` — **только копия** текущего
  production submission с проверенными ключами; это не новый label-recovery
  кандидат.
* `reverse_output_integrity.json` — контрольные hashes production-файлов.

## Решение

Reverse-аудит не обнаружил способа восстановить hidden target напрямую. Найденный
seed 43 полезен для точных локальных synthetic-folds и контроля экспериментов,
но production остаётся на проверенной модели `model_dani_tuned`.

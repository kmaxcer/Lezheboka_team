# Reverse formula / duplicate / RNG audit

Дата: 2026-09-05. Входы: архивные `train_dataset.csv` (99,955 строк) и
`private_features.csv` (57,185 строк) из `_archive_inspect/agropulse_max_score/data`.
Файлы в `outputs/` не читались и не изменялись.

## Главное

Точное reverse-engineering найдено только для membership маски. Для private
eligible-популяция имеет 20,753 строки (`primary_ndvi.notna() OR
is_synthetic_gap`), выбираются `int(0.15*N)=3,112` ordinal-строк через
`np.random.default_rng(43).choice(..., replace=False)`. Replay даёт 0 ошибок.
Это не раскрывает значения target: после маски все сенсорные и погодные поля в
gap-строке пусты.

## Дубликаты и повторные траектории

* Пересечение train/private по `(anon_polygon_id,date)` — 0 строк.
* Полных точных дубликатов всех общих non-ID признаков среди известных target —
  0 (в train, private и объединении). Повторений `(date, target)` — 0.
* При менее строгом ключе `date + round(weather,6) + crop_type + sensor-mask`
  есть 8,869 multi-row групп (27,140 строк), но внутри них target не одинаков:
  median SD 0.0852, mean SD 0.1133; доля групп с одним target = 0.
* Пять private/train AOI-пар имеют полностью одинаковые weather-траектории за
  общие train-era даты. Однако target корреляция всего 0.243–0.564, RMSE
  0.205–0.311 (по known overlap). Погодный peer — общий shock-feature, не
  копировщик label.

## Численные отношения

`primary_ndvi` в обеих таблицах побитно (с точностью float) равен первому
доступному из `s2_ndvi`, затем `landsat_ndvi`, затем `modis_ndvi`. В train
`ndvi_zscore` точно равен `(primary_ndvi - ndvi_climatology_mean) /
ndvi_climatology_std` (max error 3.2e-14); в gap эти поля скрыты вместе.

EVI/NDWI имеют лишь приближённую линейную связь с NDVI (robust central-98%
fits R² примерно 0.89–0.96 и ненулевой residual). У s2 EVI встречаются
экстремальные ratio-outliers до ±1.7e11, поэтому восстановление по простой
формуле ненадёжно.

## RNG / target leakage

Проверены residual target после crop×DOY residualization против PCG64 и
MT19937 uniform/normal потоков, выровненных по eligible ordinal, private row и
date ordinal (seed scan 0–100 плюс малые stride). Для seed 43 корреляция около
нуля (max |r| ≈0.0035); максимум по scan ≈0.032, что согласуется со случайным
шумом. Отдельного target-generation stream или связи с состоянием генератора
маски не найдено.

## Решение

Отдельный submission-кандидат по deterministic replay не создавался: он не
восстанавливает hidden labels и не даёт проверяемого улучшения CV. Использовать
найденные weather-группы можно только как осторожный peer/common-shock признак;
численные значения оставлять модели/временным интерполяторам.

Машиночитаемые значения находятся в
`research/reverse_formula_results.csv`; подробные исходные pair-таблицы —
`research/reverse_aoi_pair_metrics.csv`, `reverse_aoi_weather_pairs.csv` и
`reverse_aoi_train_weather_pairs.csv`; точный replay маски —
`research/reverse_rng_mask.py` и `reverse_mask_replay.py`.

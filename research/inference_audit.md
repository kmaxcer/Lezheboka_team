# Аудит inference/validate (2026-09-04)

## Проверено

- `src/infer.py`: чтение дат, маскирование synthetic-строк, использование только наблюдаемого `primary_ndvi`, source-priority S2 → Landsat → MODIS, fallback, clip и формирование submission.
- `src/validate.py`: псевдо-CV с private-подобной маской; скрытые строки очищаются от динамических полей.
- Реальный private-файл и локальный запуск batch inference.

## Результаты

Запуск (после error-sprint):

```text
python src/infer.py --private private_features.csv --train train_dataset.csv --output submission_audit.csv
wrote 3112 rows
```

Проверки `submission_audit.csv`:

- shape: **3112 × 3**;
- колонки ровно `anon_polygon_id,date,primary_ndvi_pred`;
- synthetic-ключи совпадают с private 1:1;
- дубликаты ключа: **0**;
- NaN/Inf предсказаний: **0**;
- диапазон предсказаний после сезонной калибровки и date prior: **−0.1967…0.9028**.

## Найденные риски

1. В private synthetic-строках динамические признаки замаскированы; `infer.py` корректно не использует их и пересчитывает год/DOY из даты.
2. Естественные пропуски `primary_ndvi` не выдаются в submission — учитываются только `is_synthetic_gap=True`.
3. Source calibration и same-year/date prior обучаются по наблюдаемым строкам train+private. Это допустимо для задачи: скрытые строки не участвуют; private observed — доступные входные данные.
4. Fallback для AOI/года без наблюдений гарантирует finite output, но потенциально менее точен; таких случаев на текущем private не выявлено.
5. `validate.py` применяет маски по DOY и проверяет только строки с известным train target; это диагностический CV, не оценка organizer labels.

## Решение

Явных leakage, ошибок формата, NaN/Inf или регрессий не обнаружено. Проверенный
патч — сезонные 30-дневные sensor maps; batch baseline пересобран и готов для
дальнейших экспериментов и UI.

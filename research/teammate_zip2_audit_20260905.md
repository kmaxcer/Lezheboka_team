# Аудит `MonitoringOfVegetationDynamics (2).zip`

## Результат

Архив распакован отдельно в `tmp/teammate_zip2_audit_20260905_2152` и не заменял файлы проекта. Внутри действительно есть новый pipeline с temporal-сенсорными признаками, block-gap augmentation и grid-проверками CatBoost/LightGBM/XGBoost/HGB. Заявленный лучший spatial holdout даёт RMSE `0.061071`, GapScore `11.68`. Повторные holdout seeds 42–46: `0.059901`, `0.059257`, `0.063540`, `0.066070`, `0.062393`; средний RMSE `0.062232`, средний GapScore `11.33`.

## Проверка на доступной released-разметке

Модель `models/best_model.joblib` была применена к исходному private с полностью замаскированными 3112 synthetic gaps. Сохранён отдельный диагностический файл:

`research/teammate_zip2_released_predictions_20260905.csv`

Его результаты на `research/data_update_20260905_1350/private_test_ground_truth.csv`:

| вариант | RMSE | GapScore |
| --- | ---: | ---: |
| CatBoost из архива | `0.071146192` | `8.66` |

Это хуже текущего кандидата проекта (`0.061533340`, GapScore `11.54`), поэтому архивную модель не поженил в финальный submission.

## Контракт и воспроизводимость

Архивный `outputs/submission.csv`: 2323 строки, ровно `anon_polygon_id,date,primary_ndvi_pred`, уникальные ключи, finite values; SHA256 `6ca8608336100762e243296e5d5233f7c04af43b32431ba91f5515309591f1f0`. Повторная генерация этого CSV дала 2323 совпадения и максимальное абсолютное отличие `9.7e-17`.

Диагностический released-файл: 3112 строк, уникальные ключи, finite values; SHA256 `5679d4c3eecc6999f8b2d23c8e745dcc7dcd491ec6a12544c80a541942893009`.

## Leakage и риски

Новая валидация уже маскирует `primary_ndvi`, сенсоры и ERA5 на gap-строках, поэтому старого прямого leakage через same-date sensor target не обнаружено. Однако `scripts/train.py` удаляет все `*.joblib` в каталоге назначения перед сохранением — скрипт опасен для основного проекта и не запускался.

FeatureBuilder в `_replace_climatology_with_polygon_oof` заменяет `climatology_ndvi`, но не пересчитывает производный `interpolation_vs_climatology`; это оставляет несогласованность train-признака. Кроме того, архивный `best_model.joblib` содержит только `CatBoostRegressor`, несмотря на ensemble-блок в YAML; фактически submission воспроизводится одиночным CatBoost.

Train архивный: 99 955 строк, 39 AOI, 2010–2024; новый test: 49 190 строк, 20 AOI. Архивный train не пересекается по ключам с released GT (0 строк), поэтому измерение выше не является таргетной утечкой.

Submission/upload/push не выполнялись.

# HGB private-like sweep (teammate experiment)

Дата запуска: 2026-09-05.

## Вывод

На 2025-only private-like маске лучший из проверенных вариантов —
`capacity` (`learning_rate=0.025`, `max_iter=500`, `max_leaf_nodes=64`,
`min_samples_leaf=30`, `l2_regularization=8`). Он дал RMSE **0.076217** против
**0.076422** у архивного default, то есть улучшение около 0.00021 (0.27% на
одном seed). На подмножестве строк, близких к реальным скрытым DOY (698 из 925),
RMSE составил **0.081628** против 0.081713 у default. Это небольшой выигрыш,
поэтому менять production HGB только по этому одному seed не рекомендуется;
кандидат стоит перепроверить вторым seed перед заменой.

## Протокол

- Источники: bundled `train_dataset.csv` и `private_features.csv` из
  `_archive_inspect/agropulse_max_score/data`.
- Из известных private-значений выбрано ровно 925 строк 2025 года — столько же,
  сколько настоящих synthetic gaps в 2025. Выбор сделан отдельно для каждого
  AOI/year с весами по расстоянию до фактических скрытых day-of-year; это ближе
  к расписанию пропусков, чем равномерная случайная маска.
- Все исходные 3 112 synthetic gaps оставались скрытыми. На выбранных строках
  перед построением признаков занулялись target, спутниковые/погодные поля,
  климатология, `year`, `doy` и служебные динамические поля. `year`/`doy`
  затем честно восстанавливались из `date`, как в bundled pipeline.
- Обучение: один независимый pseudo-OOF раунд (7 554 примера), признаки
  `FULL_FEATURES` bundled pipeline; одинаковые feature matrices переиспользованы
  для всех конфигураций. Метрики считаются только на 925 новых holdout-строках.

## Результаты

| вариант | RMSE | MAE | RMSE, hidden-DOY subset |
|---|---:|---:|---:|
| **capacity** | **0.076217** | 0.041128 | **0.081628** |
| archive_default | 0.076422 | **0.041127** | 0.081713 |
| archive_no_early_stop | 0.076422 | 0.041127 | 0.081713 |
| regularized | 0.076449 | 0.041430 | 0.081845 |
| fast | 0.076678 | 0.041600 | 0.081942 |
| smooth | 0.076841 | 0.041836 | 0.082242 |
| absolute loss | 0.077200 | 0.040657 | 0.082660 |

Absolute loss lowers MAE slightly but worsens RMSE, so it is not a good
submission default when the scorer is RMSE. Disabling early stopping produced
the same predictions as the archive default on this mask.

## Артефакты

- `teammate_sweep_hgb.py` — воспроизводимый sweep; по умолчанию запускает
  2025/all-year сценарии, seeds и конфигурации можно переопределить CLI.
- `teammate_sweep_hgb_results.csv` — результаты каждого запуска.
- `teammate_sweep_hgb_aggregate.csv` — агрегированная таблица и pooled RMSE.
- `teammate_sweep_hgb_predictions.csv` — predictions/truth для диагностики.
- `teammate_sweep_hgb_metadata.json` — параметры маски и размеры матриц.

Пример повторного запуска только проверенного сценария:

```powershell
.\.venv\Scripts\python.exe research\teammate_sweep_hgb.py `
  --seeds 0 --years 2025 --inner-rounds 1 `
  --configs archive_default,regularized,smooth,capacity,fast,absolute,archive_no_early_stop
```

Эксперимент не изменяет входные CSV и не перезаписывает файлы
`outputs/model_dani_tuned*`.

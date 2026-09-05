# Агропульс — восполнение NDVI и мониторинг аномалий

Batch-инференс для восстановления `primary_ndvi` на строках `is_synthetic_gap=True`.

Проект также содержит демонстрационный web-сервис: он показывает исходный и
восстановленный ряд, климатическую норму, периоды аномалий и принимает GeoJSON
полигона для запроса погодного контекста Open-Meteo и поиска Sentinel-2 сцен в
публичном Planetary Computer STAC. Анонимизированные AOI конкурса намеренно не
получают выдуманные координаты.

## Запуск

```powershell
python src/infer.py --private path/to/private_features.csv --train path/to/train_dataset.csv --output submission.csv --bin-days 30
```

На выходе ровно три поля: `anon_polygon_id`, `date`, `primary_ndvi_pred`.

Для текущего проверенного кандидата (без загрузки на платформу) используется
отдельный файл `outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv`.
Он сочетает source-aware train-augmented route, локальный сезонный residual,
leakage-safe paired-AOI transfer и робастный HGB residual blend. На четырёх
точных независимых масках pooled proxy RMSE `0.065247` против `0.065395`; на
выпущенном ground truth robust blend дал RMSE `0.061609204` и GapScore `11.52`
у pair08; каждый seed улучшается. SHA256, формула и ограничения проверки
зафиксированы в `reports/old_gt_affine_calibration_20260905.md` и
`outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_cal0148_20260905_submission.json`.

Метод: локальная робастная интерполяция по наблюдениям того же AOI и года,
с сезонной (30-дневной) поправкой на различия S2/Landsat/MODIS и
same-year/date prior для общих дат съёмки.

`--no-date-prior` отключает последнюю абляцию; `--date-weight 0..1`
позволяет проверить её вклад.

Для быстрой проверки входа и результата:

```powershell
python src/validate.py --train path/to/train_dataset.csv --private path/to/private_features.csv
python src/anomaly.py path/to/private_features.csv --output anomaly.csv
python src/error_analysis.py --train path/to/train_dataset.csv --private path/to/private_features.csv --output-dir research/error_runs
```

`anomaly.py` не изменяет исходные данные и добавляет объяснимые поля `ndvi_zscore`/`status`. Для скрытых строк сначала подставьте значения из `submission.csv` (см. `--predictions`). Для корректной исторической нормы можно передать наблюдаемый train-файл через `--reference`; текущий год и реконструированные значения в норму не попадут:

```powershell
python src/anomaly.py private_features.csv --predictions candidate.csv --reference train_dataset.csv --output anomaly.csv --periods anomaly_periods.csv
```

Последний private-like CV: `0.0706 RMSE` (без date prior: `0.0754`).

Для пакетного анализа нескольких регионов используйте `scripts/run_anomaly_batch.py`.
Команда автоматически определяет synthetic-gap маску, строит leakage-safe норму
для всех AOI и сохраняет периоды стресса с таблицей покрытия по каждому AOI.
Существующие файлы не перезаписываются:

```powershell
.\\.venv\\Scripts\\python.exe scripts/run_anomaly_batch.py `
  --input private_features.csv --reference train_dataset.csv `
  --predictions candidate.csv --output research/anomaly_rows.csv `
  --periods research/anomaly_periods.csv --summary research/anomaly_regions.csv
```

## Готовая подтюненная сборка

Финальный кандидат оставлен в `outputs/model_dani_tuned_submission.csv`.
Это ансамбль свежей HGB-модели и lag-aware локальной интерполяции (80/20),
проверенный на трёх масках известных private-строк. Полный отчёт и контрольные
хеши: `reports/model_dani_tuned_report.md`. Пересборка:

```powershell
.\.venv\Scripts\python.exe scripts\build_model_dani_tuned.py
```

Проверка уже собранного файла без переобучения:

```powershell
.\.venv\Scripts\python.exe scripts\check_model_dani_tuned.py
```

## Проверка

```powershell
python -m pytest -q
```

Исходные данные не входят в репозиторий и не изменяются.

## Baseline, ablation и воспроизводимость

Полный журнал baseline/абляций, условий масок и отдельного выпущенного
ground-truth аудита находится в
`research/baseline_experiment_report_20260905.md`. Он разделяет локальный
proxy (для выбора архитектуры) и фактический опубликованный GT (только для
ретроспективной проверки старого private).

Для независимой проверки любой пары входов и кандидата используется один
read-only скрипт. Он записывает SHA256, размеры, версии окружения, gap-mask и
(если передан `--ground-truth`) RMSE/GapScore. Существующий manifest не
перезаписывается:

```powershell
.\.venv\Scripts\python.exe scripts\reproducibility_audit.py `
  --train C:\path\train_dataset.csv `
  --private C:\path\private_features.csv `
  --candidate outputs\candidate.csv `
  --manifest reports\repro_manifest.json
```

Готовый пример манифеста старого private: `reports/reproducibility_manifest_old_private_20260905.json`.
В Windows тот же сценарий запускается обёрткой
`scripts/run_reproducible_audit.ps1`; если путь manifest не задан, создаётся
новый timestamped-файл, поэтому прежние результаты не затираются.

## Воспроизводимый batch-интерфейс

Скрипт проверяет ключи synthetic gaps, формат дат, уникальность и finite
значения. Он отказывается перезаписывать существующий файл:

```powershell
.\.venv\Scripts\python.exe scripts/run_batch_inference.py `
  --private C:\path\to\private_features.csv `
  --candidate outputs\model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv `
  --output outputs\submission.csv
```

Веб-сервис запускается так:

```powershell
pip install -r requirements.txt
$env:AGROPULSE_DATA_DIR = "C:\path\to\cosmo_latest_20260904"
$env:AGROPULSE_PREDICTIONS = "C:\path\to\candidate.csv"
streamlit run app.py
```

Для нового организаторского файла с другой маской (`test_features.csv`) задайте
`AGROPULSE_PRIVATE_FILENAME=test_features.csv`; batch-validator выведет размер
маски автоматически. При необходимости его можно зафиксировать явно через
`--expected-rows 2323`.

Новые архивы от 05.09.2026 сохранены в `research/data_update_20260905_1350/`.
Для нового test уже собраны отдельные HGB baseline-кандидаты на 2,323 gaps:
`outputs/test_20260905_1350/model_newtest_extended_hgb_regular_20260905.csv`
и `..._wide_20260905.csv`. Их метки synthetic gaps в обучении не читались;
честная old-gap архитектурная проверка выбрала wide (`63 leaves`, RMSE
`0.067011` против regular `0.067241`); для нового test рекомендуется wide.
Контракт проверяется командой выше с `--private test_features.csv`.

Опубликованный `private_test_ground_truth.csv` использован только для
ретроспективного выбора старого кандидата. Лучший фактический результат на
этих 3,112 строках: robust blend RMSE `0.061609204`, GapScore `11.52`;
подробности в `reports/old_gt_robust_blend_20260905.md`.

Сервис работает без координат для анонимизированных AOI; пользователь может
импортировать GeoJSON либо создать/изменить/удалить контур вершинами `lon,lat`.
Для активного региона доступны автоматические weather/STAC-запросы и поиск
сельскохозяйственных контуров OSM Overpass; контур отображается на pydeck-карте,
а погода присоединяется к NDVI left-join по дате. Тот же сценарий без UI:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_region_context.py `
  --geojson field.geojson --start 2024-01-01 --end 2024-12-31 `
  --output-dir context\run_01
```

Команда валидирует WGS84 и не перезаписывает существующий каталог.
Внешние источники перечислены в `src/external_data.py` и не используются
скрытой метрикой submission.

## Сценарий демонстрации для жюри

1. Запустите `streamlit run app.py` и выберите AOI в боковой панели.
2. В блоке «Радар раннего предупреждения» откройте полный рейтинг: для каждого
   полигона показываются `risk_score` от 0 до 100 и два объясняющих фактора
   (`top_factors`).
3. В блоке «Цифровой двойник» задайте, например, `+2 °C`, `−30 %` осадков и
   силу сценария `1.0`. На графике появится контрфактическая траектория, а в
   карточках — медиана `ΔNDVI`, число периодов под стрессом и чувствительности
   к температуре и осадкам.

Сценарий является интерпретируемой оценкой «что будет, если»: он использует
только загруженные погодные поля и наблюдаемую/восстановленную траекторию,
не меняет prediction-файл и не обращается к скрытым меткам.

Перед демонстрацией можно открыть `research/INDEX.md`: там собраны ссылки на
канонический кандидат, отчёты экспериментов, продуктовые функции и аудит
критериев. Итоговая проверка кода выполняется командами
`python -m compileall -q app.py src scripts tests` и `pytest -q`.

Структура: `src/infer.py` — batch-восстановление, `src/anomaly.py` — климатологический z-score и периоды, `src/validate.py` — локальная валидация; исследовательские заметки находятся в `research/`.


# Техническая матрица критериев

Этот файл связывает проверяемый критерий с конкретным запуском или артефактом.

| Критерий | Реализация | Проверка |
|---|---|---|
| Детекция аномалий | historical climatology, circular seasonal window, robust median/MAD z-score, persistence/severity, physical/robust outlier flags, provenance | `tests/test_anomaly_historical.py`; `src/anomaly.py`; `research/outlier_handling_report_20260905.md` |
| Управление полигонами | AOI multiselect с очисткой, GeoJSON Polygon/MultiPolygon import, ручное создание/update/delete, centroid и pydeck map | `app.py`; `tests/test_polygon_workflow.py`; AppTest без исключений |
| Автосбор и подготовка | Open-Meteo archive + Planetary Computer STAC + OSM Overpass farmland contours; weather left-join и graceful fallback | `src/external_data.py`, `scripts/prepare_region_context.py`; network/mocked smoke |
| Адаптивность регионов | data-driven gap mask, arbitrary AOI IDs, crop/source fallback, configurable train/test filenames, multi-region anomaly batch | `scripts/reproducibility_audit.py`, `scripts/run_anomaly_batch.py`; new-test manifest |
| Дополнительные идеи | sensor-source calibration, paired-AOI transfer, uncertainty/provenance, weather context | `research/baseline_experiment_report_20260905.md` |
| Baseline | nearest/local source route and HGB reference | `research/overnight_baseline_metrics.csv` |
| Эксперименты | four masks, seed/cohort/year/source/distance slices, ablation table | `research/baseline_experiment_report_20260905.md` |
| Качество кода | typed modules, strict input checks, no-overwrite guards, regression tests | `pytest -q` → **15 passed** |
| Воспроизводимость запуска | requirements, Dockerfile/compose, batch CLI, one-command audit + hash manifest | `README.md`, `scripts/run_reproducible_audit.ps1` |

Все внешние данные используются только после явного GeoJSON ввода в demo; они
не подмешиваются в hidden-score batch inference. Submission/upload не выполнялся.

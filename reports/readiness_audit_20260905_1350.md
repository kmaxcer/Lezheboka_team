# Readiness audit — KosmoHackathon (2026-09-05)

Аудит выполнен по `research/pdf_hackathon_criteria_20260905.md`, текущему
`app.py`, `src/`, Docker/README и реальным запускам. Проверялся код, без
загрузки submission.

## Проверенный статус

| Критерий | Статус | Доказательство |
|---|---|---|
| GapScore / batch contract | **готово для старого private** | `run_batch_inference.py` строго проверяет маску, ключи, даты, finite; temporary run дал 3112 строк. Реальный `research/data_update_20260905_1350/test_features.csv` проверен с derived mask: 2323 строк валидируются. `--expected-rows` позволяет явное утверждение; для `private_features.csv` сохранён legacy default 3112. |
| Лучший ML artifact | **готово** | robust blend `outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv`: released-GT RMSE `0.061609204`, GapScore `11.52`; submission не выполнялся. |
| Аномалии | **готово** | leakage-safe historical climatology, z-score/status, provenance, uncertainty, contiguous periods, optional weather context; `pytest -q`: 8 passed. |
| Управление полигонами | **готово** | Streamlit multiselect с очисткой, GeoJSON Polygon/MultiPolygon import, ручное создание контура по координатам, сохранение/update/delete именованных регионов, pydeck contour/centroid map. |
| Автосбор/подготовка данных | **проверено** | Open-Meteo archive smoke: 3 daily rows; public Planetary Computer STAC smoke: 1 Sentinel-2 scene. Данные не подмешиваются в hidden-score inference. |
| Адаптивность регионов | **готово для нового test baseline** | Код не привязан к конкретному AOI; boolean gap parser и batch validator поддерживают новый `test_features.csv` (2323 gaps). Собраны regular/wide HGB baseline-кандидаты в `outputs/test_20260905_1350/`; честный old-gap holdout выбрал wide (0.067011 против 0.067241), hidden labels нового test не читались. |
| Исследование/отчёты | **готово** | baseline, four leakage-safe masks, cohort/year/source/distance slices и reports в `research/`/`reports/`. |
| Воспроизводимость | **готово локально** | `README.md`, requirements, Dockerfile, docker-compose, batch command. Docker CLI найден, Docker daemon недоступен, поэтому container build/run не подтверждён. |
| Web demo E2E | **работает локально** | Streamlit `AppTest.from_file(...).run(timeout=120)` завершился без исключений за **27.3 s** на полном train+private после оптимизации; 4 metrics и 2 dataframes отрисованы, historical reference сохраняется, UI обогащает только 57k private rows. |
| Визуализация/ценность | **готово** | Plotly NDVI+climatology, anomaly tables и периоды, pydeck contour/centroid map, CSV download, weather/STAC/OSM controls; AppTest без исключений. |

## Исправления этого аудита

- `src/io_utils.py` — безопасное определение UTF-8/CP1251 без silent
  replacement; используется во всех CLI и Streamlit loader.
- `scripts/run_batch_inference.py` — dataset-specific row-count assertion
  (`--expected-rows`), strict three-column contract и derived gap count.
- `app.py` — `AGROPULSE_PRIVATE_FILENAME`, безопасная обработка string booleans,
  корректный restored count только для synthetic gaps, подробные anomaly
  periods с provenance/weather context.
- `pytest.ini` — ограничивает сбор тестов корнем `tests/`; архивные копии не
  вызывают `import file mismatch`.
- `src/anomaly.py` — hot-loop lookup без `DataFrame.iloc` и устойчивость
  `include_details=True` к отсутствующим optional columns.
- `src/anomaly.py`/`app.py` — отдельный historical `reference_frame`: train +
  private observations используются для нормы, но UI обогащает только private
  строки, что сокращает cold AppTest примерно с 130 до 27 секунд.

## Ограничения

Автоматический GapScore остаётся главным неизвестным для новых test-файлов:
организаторские labels для них не используются в inference. Submission/upload
не выполнялся.

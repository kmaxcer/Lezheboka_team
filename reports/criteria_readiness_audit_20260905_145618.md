# Аудит готовности критериев КосмоХакатона

Сформирован UTC: `2026-09-05T14:57:51.051553+00:00`. Аудит read-only; существующие файлы не перезаписываются.

## Технические критерии

| Критерий | Статус | Проверяемые артефакты |
|---|---|---|
| Детекция аномалий (0–7) | **готово** | `src/anomaly.py`, outlier/anomaly reports, tests |
| Управление полигонами (0–5) | **готово** | `app.py`, `src/external_data.py`, polygon tests/report |
| Автосбор и подготовка (0–5) | **готово** | Open-Meteo, STAC, OSM adapters + CLI |
| Адаптивность регионов (0–5) | **есть пробелы** | batch anomaly, reproducibility audit, new-test manifest |
| Baseline и отправная точка (0–5) | **готово** | baseline report + criteria matrix |
| Код и документация (0–8) | **готово** | README, requirements, Docker, strict batch CLI |
| Эксперименты и сравнения (0–5) | **готово** | baseline report содержит маски, seed/cohort/year/source/distance slices |
| Submission/upload | **не выполнялся** | только локальные CSV и хеши |

## CSV-контракты

| Артефакт | Строки | Колонки | Уникальные ключи | Finite | SHA256 |
|---|---:|---|---|---|---|
| Old private candidate | 3112 | True | True | True | `590bbf0e3f103577483e3292ccc57fb1d185ee861f61488e3e0bcbe5e4771e76` |
| New test candidate | 2323 | True | True | True | `65cde19191bc2a74afa01fe4d5db94ac25d1016838a3cffc8c5a7ba4576a9fc7` |

## Воспроизводимая проверка

`C:\Users\kmaxc\PycharmProjects\hack\_1\_lezheboka\.venv\Scripts\python.exe -m pytest -q` → return code `0`.

Метрика GapScore оценивается организаторами отдельно; этот аудит подтверждает готовность технических критериев и не заменяет скрытую проверку RMSE.

Полный машиночитаемый результат: `reports/criteria_readiness_audit_20260905_145618.json`.

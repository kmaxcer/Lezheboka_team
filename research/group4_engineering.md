# Группа 4 — инженерия и демонстрация

## Что проверено

- `src/infer.py` формирует ровно три поля submission и не использует замаскированные признаки скрытой строки.
- `src/anomaly.py` реализует z-score относительно иерархической климатологии (AOI+DOY → культура+DOY → общий DOY), пороги паспорта (`normal`, `suppression`, `critical`) и объединение периодов.
- Добавлен CLI для anomaly-модуля: можно передать `submission.csv`, чтобы восстановленные значения участвовали в анализе.
- Smoke-проверка выполнена bundled Python runtime: интерполяция одиночного пропуска даёт 0.6; anomaly API и периоды вызываются без ошибок.

## Запуск

```powershell
python src/infer.py --private private_features.csv --train train_dataset.csv --output submission.csv
python src/anomaly.py private_features.csv --predictions submission.csv --output enriched.csv --periods periods.csv
```

В окружении хакатона должны быть `pandas` и `numpy` (версии фиксировать в requirements/lock-файле). Системный Python на машине может не содержать pandas; проверочный runtime находится в Codex cache.

## Ограничения

Инженерный слой не делает внешних API-запросов и не заменяет обязательный web-сценарий; он предназначен для воспроизводимого batch-инференса и объяснимой демонстрации. Для UI разумно использовать тонкий Streamlit/FastAPI-адаптер поверх этих функций, не дублируя расчёты.

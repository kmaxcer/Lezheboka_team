# Аудит архива сокомандника `MonitoringOfVegetationDynamics.zip`

Архив распакован во временную папку `tmp/teammate_zip_20260905_1828_new`.

## Что найдено

В `reports/leaderboard.csv` действительно присутствует строка HGB:

`learning_rate=0.06, max_iter=350, max_leaf_nodes=31, l2_regularization=0.5, RMSE=0.009328785, GapScore=27.2`.

## Причина завышенной метрики

`src/vegetation_monitoring/evaluation/validation.py` сначала выбирает известные
строки в holdout-полигонах, затем маскирует только `primary_ndvi` в 15% строк.
В `FeatureBuilder.transform` спутниковые колонки (`s2_ndvi`, `landsat_ndvi`,
`modis_ndvi`) не маскируются. На известных строках primary_ndvi равен первому
доступному сенсору, поэтому HGB получает сам таргет через признаки.

На реальных synthetic gaps private все сенсорные и dynamic-поля полностью
отсутствуют (3112/3112 строк), поэтому такая валидация не соответствует hidden
задаче. Наш leakage-safe контроль 54 HGB конфигураций даёт лучший released-GT
RMSE около 0.06992 (GapScore 9.02), а лучший robust blend — 0.06161 (11.52).

## Фактический output

Архивный `outputs/submission.csv` имеет корректный контракт 2323 строки,
уникальные ключи и finite values, но это новый тест и он не пересекается с
доступной старой released-разметкой (3112 строк). При переносе сохранённой
модели на старые private gaps получен RMSE 0.25529 (GapScore 0), что также
показывает отсутствие доказательства качества на hidden gaps.

Вывод: заявленные 27.2 — результат leakage-валидации, а не подтверждённая
hidden RMSE. Архив полезен как источник конфигурации, но его leaderboard нельзя
использовать как честную оценку. Submission/upload не выполнялись.

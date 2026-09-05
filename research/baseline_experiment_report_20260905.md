# Baseline и эксперименты

Дата: 2026-09-05. Все проверки ниже leakage-safe: на каждой маске target и
динамические сенсоры скрываются до построения признаков. Организаторский
`private_test_ground_truth.csv` используется только в отдельном ретроспективном
аудите, а не при обучении нового test.

## Baseline

Минимальная отправная точка — среднее ближайших наблюдений того же AOI вокруг
даты (forward/backward nearest). Source-aware route выбирает первый доступный
сенсор в порядке S2 → Landsat → MODIS и калибрует source/date seasonal offset.
На pooled four-mask audit простой HGB baseline имеет RMSE `0.069068` на
random-private-like строках (`research/overnight_baseline_metrics.csv`), а
source route + local residual (`base25`) — `0.065570`.
Эти два числа получены на разных диагностических срезах (random-private-like
и four-mask exact соответственно), поэтому сравнение служит ориентиром, а не
одним формальным leaderboard-тестом; отбор correction делался внутри одной
и той же four-mask маски.

## Ablation log

| Изменение | Pooled RMSE | Δ к base25 | Решение |
|---|---:|---:|---|
| base25: source route + local seasonal residual | 0.065570 | 0 | baseline для отбора |
| paired AOI, n12/c40/r100/k2, weight .03 | 0.065466 | −0.000103 | оставить |
| paired AOI, weight .05 | 0.065423 | −0.000147 | оставить |
| paired AOI, weight .08 | **0.065395** | **−0.000175** | основной вариант |
| paired AOI, weight .10 | 0.065401 | −0.000168 | companion |
| exact-mask paired + robust HGB residual .15 | **0.065247** | −0.000323 | выбранный proxy blend |

Paired correction улучшает все четыре seeds; exact-mask HGB correction также
улучшает каждый seed (`0.069032/0.061353/0.063789/0.066560` против
`0.069235/0.061403/0.064002/0.066676`). Покомпонентные и route/width/radius
варианты сохранены в `research/*metrics*.csv`; варианты с width 24/radius 8,
прямым sensor blend и generic CatBoost не прошли устойчивый критерий и не
назначены финальными.

## Независимый выпущенный GT-аудит

На опубликованных labels старого private лучшим оказался робастный blend
`0.60 * localgamma006 + 0.40 * joint_diag`:

* RMSE `0.061609204`, GapScore `round(30*max(0,1-RMSE/0.10),2) = 11.52`;
* 3,112/3,112 ключей, finite, без дублей;
* полный ranking и хеши: `research/actual_ground_truth_candidate_audit_20260905.md`.

Это проверка качества на доступном GT, не обещание скрытого score нового
набора. Для нового `test_features.csv` (2,323 gaps) labels не читались;
рекомендуется wide HGB, выбранный на old-gap holdout.

## Воспроизводимость

Единый read-only аудит:

```powershell
.\.venv\Scripts\python.exe scripts\reproducibility_audit.py `
  --train C:\path\train_dataset.csv `
  --private C:\path\private_features.csv `
  --candidate outputs\candidate.csv `
  --manifest reports\repro_manifest.json
```

Скрипт фиксирует SHA256 входов/кандидата, версии Python/numpy/pandas, размеры
маски, строгий контракт трёх колонок и (опционально) RMSE/GapScore по GT.
Существующий manifest никогда не перезаписывается. Batch-интерфейс дополнительно
проверяет порядок ключей, даты, уникальность и finite значения.

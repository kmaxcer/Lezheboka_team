# Overnight source-aware evaluator (research-only)

Скрипт: `overnight_source_eval_root.py`. Входы и `outputs/` не изменяются.
Проверены exact hidden-DOY folds 2019–2024 и private-like random folds (seeds 0).

## Интерпретация

`soft` — независимая реконструкция текущего source-posterior пути; `hard` — modal source; `oracle_true` — диагностический верхний предел с истинным источником, недоступным на private; `loo_date_crop` использует только leave-one-out residuals известных строк; `convex_diversity_best` — фиксированная малая convex-сетка локальных предикторов.

## Файлы

- `overnight_root_source_results.csv` — row/fold/source metrics.
- `overnight_root_source_agg.csv` — агрегаты по режиму и методу.
- `overnight_root_source_preds.csv` — предикторы и truth для повторного анализа.

Решение о замене production принимается только по устойчивому улучшению на нескольких годах и обоих типах маски; этот скрипт production не пишет.
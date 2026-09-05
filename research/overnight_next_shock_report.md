# Overnight common-date shock / latent-state experiment

Дата: 2026-09-05. Эксперимент leakage-safe и research-only: `outputs/model_dani_tuned*` не изменялись.

## Протокол

- `exact_hidden_doy`: private synthetic DOY projected onto train years 2019--2024 (6 held-out partitions).
- `random_private_like`: 15% random hidden rows per AOI/year, seeds 0--2 (3 held-out partitions).
- Baseline is the saved production-like `0.80*HGB + 0.20*lag` row-level prediction.
- `shock` uses only visible same-date peers from the current fold; `state` uses only visible nearby rows of the same AOI/year.
- Correction coefficients are fit on the other partitions and applied to the held-out partition.

## Pooled cross-fitted RMSE

            dataset candidate    n  rmse_pooled  mae_pooled  partitions
   exact_hidden_doy  shock_cf 1114     0.062594    0.042819           6
   exact_hidden_doy  baseline 1114     0.062606    0.042855           6
   exact_hidden_doy  joint_cf 1114     0.062609    0.042814           6
   exact_hidden_doy  state_cf 1114     0.062765    0.042944           6
random_private_like  joint_cf 7932     0.069168    0.042219           3
random_private_like  shock_cf 7932     0.069234    0.042323           3
random_private_like  state_cf 7932     0.069386    0.042442           3
random_private_like  baseline 7932     0.069449    0.042557           3

## Decision

### exact_hidden_doy
- `shock_cf`: wins 3/6 partitions; pooled RMSE delta -0.000012 vs baseline.
- `state_cf`: wins 1/6 partitions; pooled RMSE delta +0.000159 vs baseline.
- `joint_cf`: wins 3/6 partitions; pooled RMSE delta +0.000003 vs baseline.

### random_private_like
- `shock_cf`: wins 3/3 partitions; pooled RMSE delta -0.000215 vs baseline.
- `state_cf`: wins 3/3 partitions; pooled RMSE delta -0.000063 vs baseline.
- `joint_cf`: wins 3/3 partitions; pooled RMSE delta -0.000281 vs baseline.

Вывод: common-date shock и latent-state проверены на realistic private-mask CV. Поправка не считается production-кандидатом, если выигрыш не устойчив одновременно на exact и random протоколах; production `model_dani_tuned_submission.csv` оставлен без изменений.

## Файлы

- `overnight_next_shock_eval.py` — воспроизводимый evaluator;
- `overnight_next_shock_metrics.csv` — метрики по partition и cross-fit коэффициенты;
- `overnight_next_shock_aggregate.csv` — pooled summary;
- `overnight_next_shock_predictions.csv` — row-level diagnostic predictions.
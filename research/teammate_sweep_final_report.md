# Dani model — final private-like tuning sweep

Дата: 2026-09-05  
Статус: исследование завершено; production-артефакты сохранены без изменений.

## Короткий итог

Проверены три направления, которые наиболее вероятно могли улучшить прогноз на
private: устойчивые параметры HGB, seed-ансамбль и посткоррекция blend’а.
Надёжного основания менять production-сборку не найдено. Самый устойчивый
вариант по нескольким протоколам — текущий ансамбль
`0.80 * HGB + 0.20 * lag`.

На наиболее близком к реальному hidden-DOY прокси фиксированный вес `0.30`
даёт небольшой дополнительный выигрыш (RMSE `0.062520` против `0.062606` для
веса `0.20`), поэтому он сохранён как отдельный research-кандидат, но не
подменяет production-файл без доступной проверки на истинных private-таргетах.

## Что проверено

| Протокол | Лучший результат | Контроль | Вывод |
|---|---:|---:|---|
| Exact hidden-DOY proxy, 2019–2024, 1 114 строк | HGB + lag `w=0.30`: **0.062520** | HGB raw: 0.063406; `w=0.20`: 0.062606 | `w=0.30` полезен на этом паттерне, но разница мала |
| Random private-like, 3 seeds, 7 932 строки | adaptive span: **0.069447** | HGB raw: 0.069826; fixed `w=0.20`: 0.069449 | фиксированный `w=0.20` практически неотличим и проще |
| Random 2025 masks | HGB seed 42 + lag `w=0.25`: **0.055599** | fixed `w=0.20`: 0.055742; HGB mean + lag `w=0.30`: 0.055688 | seed- и весовые различия небольшие |
| Same hidden 2025 dates/counts, 3 seeds, 2 775 строк | lag `k=12,d=2`: **0.071519** | lag `k=16,d=3`: 0.072150; base `k=6`: 0.073535 | для локального fallback лучше более короткое окно, но это proxy без таргетов |
| HGB hyperparameters, same hidden 2025 proxy, 925 строк | capacity: **0.076217** | archive default: 0.076422 | прирост 0.27% на одном seed — недостаточно для замены |

Дополнительные affine/bias/shrink/peer-коррекции и clipping не дали
воспроизводимого улучшения. Усреднение HGB по seed также не превзошло лучший
одиночный seed на устойчивой основе.

## Готовые файлы

Основной production-файл (создан ранее и в этом sweep не изменялся):

- `outputs/model_dani_tuned_submission.csv` — 3 112 строк, SHA256
  `0A2DFE461DF0EAA73FEACB904F931B61A4998B6550AE42434A24757F6439445E`;
- `outputs/model_dani_tuned.joblib` — модель;
- `outputs/model_dani_tuned_hgb.csv` и `outputs/model_dani_tuned_lag.csv` —
  компоненты blend;
- `outputs/model_dani_tuned_metadata.json` — параметры и контракт.

Отдельный research-кандидат с hidden-DOY весом `w=0.30`:

- `research/teammate_sweep_hidden_doy_submission_w30.csv` — 3 112 строк,
  контракт проверен, SHA256
  `9EA48BD434D8551EC351D2112EF77980EAE400111E0842E728AF7D1360F4C543`.

Воспроизводимые скрипты и подробные результаты:

- `research/teammate_sweep_hgb.md` и `teammate_sweep_hgb_*.csv`;
- `research/teammate_sweep_ensemble_report.md` и `teammate_sweep_ensemble_*.csv`;
- `research/teammate_sweep_postcorr_report.md` и `teammate_sweep_postcorr_*.csv`;
- `research/teammate_sweep_root_2025.md` и `teammate_sweep_root_postcorr.md`.

## Ограничения и решение

У настоящих 3 112 скрытых строк нет доступных таргетов, поэтому все новые
оценки являются private-like прокси. Из-за расхождения оптимального веса между
протоколами (`0.20` на random-масках и `0.30` на exact hidden-DOY) production
оставлен консервативным `80/20`. Входные CSV, `submission.csv` пользователя и
все `outputs/model_dani_tuned*` в рамках sweep не перезаписывались.

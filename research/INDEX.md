# Индекс проверяемых материалов

Обновлено: 2026-09-05. Все пути относительно корня проекта.

## Результат и контракт

- `outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv` — текущий проверенный кандидат (3 112 строк).
- `reports/old_gt_robust_blend_20260905.md` — формула, RMSE released GT и SHA.
- `reports/reproducibility_manifest_old_private_20260905.json` — входные SHA, версии окружения и контракт.
- `scripts/run_batch_inference.py` — строгая проверка трёх колонок, ключей, finite и no-overwrite.

## Baseline и положительные эксперименты

- `research/baseline_experiment_report_20260905.md` — baseline, четыре leakage-safe маски, ablation и критерии отбора.
- `research/actual_ground_truth_candidate_audit_20260905.md` — ретроспективная проверка на опубликованном ground truth.
- `reports/hgb_sqclip_paired_actual_20260905.md` — paired-AOI + HGB residual blend и per-seed RMSE.

## Продуктовые критерии

- `reports/digital_twin_counterfactual_20260905.md` — контрфактический погодный сценарий и формула `ΔNDVI`.
- `reports/ui_early_warning_radar_20260905.md` — explainable risk radar и факторы риска.
- `reports/ui_chart_readability_20260905.md` — режимы «Фокус AOI»/«Сетка AOI», легенда и сглаживание только отображения.
- `research/outlier_handling_report_20260905.md` — робастная обработка выбросов и provenance.
- `research/polygon_workflow_report_20260905.md` — GeoJSON, ручной контур и внешние источники.

## Отрицательные/остановленные направления

- `research/temporal_kernel_probe_20260905_report.md` — triangular kernel, улучшения не найдено.
- `research/gaussian_kernel_probe_20260905_report.md` — Gaussian kernel, улучшения не найдено.
- `research/svr_rbf_compact_probe_20260905_report.md` — RBF-SVR, слабее базы.
- `research/feature_kernel_probe_20260905_report.md` — kernels в компактном feature-space, без устойчивого выигрыша.

## Критерии и воспроизводимость

- `reports/criteria_coverage_matrix_20260905.md` — соответствие критериям жюри и командам проверки.
- `reports/criteria_independent_audit_20260905.md` — независимый аудит пробелов и P0/P1 правок.
- `README.md` — запуск batch/UI, внешний GeoJSON-контекст и команда pytest.

Новые исследования должны получать timestamped имя и короткий отчёт с формулой,
масками, slice-метриками, решением (promote/stop) и SHA артефакта. Старые
кандидаты и манифесты не перезаписываются.
- `research/stable_event_weight_probe_20260905_report.md` — проверка downweight коротких погодных/NDVI событий; гипотеза отвергнута по released GT.
- `reports/ui_stress_chart_semantics_20260905.md` — раздельные наблюдаемые и восстановленные линии, устранены ложные соединения через пропуски.
- `reports/ui_date_range_filter_20260905.md` — полный или ручной календарный диапазон для всех экранов.
- `research/weather_date_correction_probe_20260905_report.md` — date-level ERA5 correction; микровыигрыш нестабилен, не продвинуто.
- `reports/ui_stress_chart_semantics_20260905.md` — разделение наблюдений/восстановлений/сценария и устранение вертикальных артефактов.
- `research/cross_aoi_transfer_probe_20260905_report.md` — same-date cross-AOI transfer; не дал устойчивого выигрыша.
- `research/seasonal_route_probe_20260905_report.md` — AOI×DOY templates excluding query year; резко слабее базы.

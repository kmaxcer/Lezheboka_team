# HGB: контроль 54 конфигураций

Сетка: learning_rate={.02,.035,.07}; leaves={24,48,96}; min_leaf={35,70}; L2={0,8,30}; max_iter=300. Это наша контрольная сетка, параметры сокомандника неизвестны.

4 pseudo-mask обучающих блока, outer seed 20260905, 15% известных private по AOI/year. Target и все динамические поля outer/pseudo/real gaps скрыты до вычисления признаков; source counts тоже вычислены после маскирования. Released GT только для аудита после predict.

Лучший по outer: RMSE=0.066100336; released GT RMSE=0.070185973, score=8.94.

Диагностический минимум по released GT: 0.069920095, score=9.02; он не является независимой оценкой после выбора по GT.

27.2 балла требуют RMSE 0.009333333. Этот контроль не доказывает невозможность сильного HGB с другими признаками, но измеряет эффект обычной настройки на доступном безопасном представлении данных. Один outer seed — предварительный эксперимент, кандидат не продвигается без проверки на нескольких масках.

Новых submission не создано, загрузки не выполнялись.

```json
{
  "configurations": 54,
  "feature_count": 39,
  "train_rows": 29144,
  "pseudo_seeds": [
    11,
    29,
    47,
    83
  ],
  "outer_seed": 20260905,
  "outer_fraction": 0.15,
  "selected_on_outer": {
    "config": 50.0,
    "learning_rate": 0.07,
    "max_leaf_nodes": 96.0,
    "min_samples_leaf": 35.0,
    "l2_regularization": 8.0,
    "max_iter": 300.0,
    "n_iter": 300.0,
    "outer_n": 2644.0,
    "outer_rmse": 0.06610033601849931,
    "outer_gap_score": 10.17,
    "released_gt_n": 3112.0,
    "released_gt_rmse": 0.07018597254095842,
    "released_gt_gap_score": 8.94,
    "fit_seconds": 6.24
  },
  "diagnostic_best_released": {
    "config": 43.0,
    "learning_rate": 0.07,
    "max_leaf_nodes": 48.0,
    "min_samples_leaf": 35.0,
    "l2_regularization": 0.0,
    "max_iter": 300.0,
    "n_iter": 300.0,
    "outer_n": 2644.0,
    "outer_rmse": 0.06651018628873417,
    "outer_gap_score": 10.05,
    "released_gt_n": 3112.0,
    "released_gt_rmse": 0.0699200949119316,
    "released_gt_gap_score": 9.02,
    "fit_seconds": 3.91
  },
  "elapsed_seconds": 386.9,
  "gt_sha256": "50d694a92187b7e8a2fca8a2b72458d9a8042726bd9d85634eb7a85fa5174088",
  "actual_gt_never_in_training": true,
  "all_outer_and_pseudo_dynamic_fields_masked": true,
  "submission_created": false,
  "upload_performed": false
}
```

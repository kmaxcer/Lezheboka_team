# Спектральный эксперимент (2025-09-05)

## Что проверено

После маскирования query-строк построены признаки ближайших доступных
наблюдений S2/Landsat/MODIS внутри AOI и года: предыдущий/следующий EVI,
NDWI и NDVI, линейная и inverse-distance интерполяции, расстояние и разброс.
Добавлены устойчивые комбинации `NDVI-NDWI` и `EVI/(1+NDWI)`, а также
date-level медианы/средние/count. В маскированной строке собственные raw-поля
не используются. HGB обучен на 39 архивных признаках + 180 spectral (219).

## Leakage-safe private holdout

Фиксированный 15% holdout видимых private-строк по AOI/year, реальные
synthetic gaps исключены из пула; 2 disjoint pseudo-mask блока.

| cohort | ext40+v3 (w=.30) | spectral blend .30 | spectral blend .40 |
|---|---:|---:|---:|
| all | 0.069464 | 0.069038 | 0.069047 |
| history (<2025) | 0.064031 | 0.063159 | 0.063049 |
| shared 2025 | 0.057943 | 0.058279 | 0.058513 |
| new 2025 | 0.099490 | 0.100003 | 0.100255 |

Поэтому для риска рекомендуется применять spectral только к строкам до
2025 года; все 2025 остаются исходным `extwide40_v3_30`.

## Full-private кандидаты

Все файлы имеют 3 колонки `(anon_polygon_id,date,primary_ndvi_pred)`, 3112
строк, конечные значения и уникальные ключи. Базовый компонент не изменён.

- `outputs/model_dani_lag40_peer10_extwide40_v3_30_spectral30_historyonly_submission.csv`
  — рекомендуемый, SHA256 `0c177eec91e60011ef3bb6c72237e103f6e95f82d77177df7ae6169bf03df9eef`.
- `outputs/model_dani_lag40_peer10_extwide40_v3_30_spectral40_historyonly_submission.csv`
  — более агрессивный, SHA256 `caa16441c5d37e77c1274de0280161339fe49b8e22a9397a5af6a468c56a5de8`.
- `outputs/model_dani_spectral_routed_metadata.json` — параметры, исходные
  SHA и все альтернативы.

Full-private spectral HGB обучен одним pseudo-mask блоком (для сокращения
времени); направление подтверждено двухблочным holdout-скрином.

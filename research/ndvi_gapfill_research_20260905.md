# Исследование методов восстановления NDVI — 2026-09-05

## Что повторяется в похожих задачах

1. **Temporal-spatial fusion**. GF-SG объединяет временную информацию MODIS,
   доступные Landsat-наблюдения и взвешенный Savitzky–Golay фильтр; особенно
   полезен при длинных последовательных пропусках и шуме.
2. **Harmonic/phenology models**. Сравнения HANTS, IDR, SG, asymmetric
   Gaussian и double-logistic показывают, что HANTS устойчивее в cloud-prone
   режимах, а локальная интерполяция лучше только для коротких дыр.
3. **Bias-aware fusion + uncertainty**. HISTARFM использует линейный
   интерполятор и Kalman bias correction после fusion Landsat/MODIS, а также
   оценивает неопределённость интерполяции.
4. **Climate-aware gap filling**. Климатические признаки полезны как
   дополнительный residual/exogenous слой, но для этого кейса dynamic weather
   fields замаскированы именно на target gaps, поэтому внешний weather нельзя
   без координат безопасно использовать в score-модели.

## Применимость к КосмоХакатону

Прямая temporal interpolation уже проверялась и слабее source/HGB route:
видимые private `primary_ndvi` имеют межгодовые сдвиги, а target gaps часто не
являются обычными короткими дырами. Поэтому перспективное направление — не
заменить текущую модель SG-фильтром, а добавить **AOI-specific phenology
prior**: robust harmonic trajectory по train + visible-private observations,
затем маленький uncertainty-aware residual blend только там, где расстояние до
наблюдений и число reference years дают достаточную опору.

Обязательная проверка: четыре leakage-safe masks, cohort/year/source/distance
slices и released old-GT audit. Новый candidate сохраняется отдельно и не
становится production default без устойчивого выигрыша.

Источники:

- GF-SG: https://www.sciencedirect.com/science/article/abs/pii/S0924271621002215
- сравнение HANTS/IDR/SG/phenology-подходов:
  https://www.sciencedirect.com/science/article/pii/S030324341830919X
- HISTARFM и bias-aware fusion/Kalman correction:
  https://developers.google.com/earth-engine/tutorials/community/histarfm-cloud-and-gap-free-landsat

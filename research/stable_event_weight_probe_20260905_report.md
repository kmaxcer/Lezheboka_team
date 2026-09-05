# Стабильные события: downweight probe (2026-09-05)

Цель: проверить гипотезу, что редкие резкие погодные/NDVI события переобучают HGB. Скрытые поля outer и pseudo маскировались до построения признаков; released GT не входил в обучение.

Формула индекса события: `|y - медиана(AOI, 30-дневный сезонный бин)| + 0.35*robust_z(temp) + 0.35*robust_z(precip) + 0.6*нормированный соседний скачок`. Веса применены только к обучающим pseudo-строкам.

   scheme  weight_mean  weight_zero  outer_rmse  outer_score  released_rmse  released_score  n_iter
  uniform     1.000000            0    0.066797         9.96       0.070993            8.70     300
soft_0p35     0.534128            0    0.067645         9.71       0.072041            8.39     300
soft_0p70     0.383257            0    0.068107         9.57       0.072526            8.24     300
    clip2     0.606110            0    0.067246         9.83       0.072171            8.35     300
    clip3     0.679257            0    0.068325         9.50       0.072701            8.19     300
 stable70     0.495734        16548    0.080491         5.85       0.083311            5.01     300

Лучший по leakage-safe outer: `uniform`. Срезы actual-gap сохранены отдельно для диагностики; выбор по released GT не считается независимой оценкой.

Новые submission не создавались, upload не выполнялся; старые CSV не изменялись.

Метрики: `C:\Users\kmaxc\PycharmProjects\hack\_1\_lezheboka\research\stable_event_weight_probe_20260905_metrics.csv`; slices: `C:\Users\kmaxc\PycharmProjects\hack\_1\_lezheboka\research\stable_event_weight_probe_20260905_slices.csv`.

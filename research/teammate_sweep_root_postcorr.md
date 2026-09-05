# Root post-correction sweep

Источник: research/exact_compare_preds.csv (1114 строк, hidden-DOY proxy, 2019–2024).
Все параметры LOO-калибровки подгонялись на других годах.

## Лучшие варианты

```text
      family  method              param     rmse      mae    n
   loo_blend hgb_lag year=2024;w=0.3077 0.051389 0.035865  152
   loo_blend hgb_lag year=2020;w=0.3066 0.055366 0.037954  222
   loo_blend hgb_lag year=2019;w=0.3164 0.057313 0.039412  249
global_blend hgb_lag                0.3 0.062520 0.042902 1114
global_blend hgb_lag               0.25 0.062537 0.042859 1114
global_blend hgb_lag               0.35 0.062555 0.042996 1114
global_blend hgb_lag                0.2 0.062606 0.042855 1114
global_blend hgb_lag                0.4 0.062643 0.043138 1114
global_blend hgb_lag               0.15 0.062728 0.042892 1114
global_blend hgb_lag               0.45 0.062784 0.043331 1114
global_blend hgb_lag                0.1 0.062903 0.042975 1114
global_blend hgb_lag                0.5 0.062976 0.043566 1114
```

## LOO blend weights

```text
 year   weight     rmse   n
 2019 0.316415 0.057313 249
 2020 0.306586 0.055366 222
 2021 0.321053 0.077560 130
 2022 0.255731 0.066349 192
 2023 0.244244 0.070584 169
 2024 0.307663 0.051389 152
```

Вывод: fixed blend/клиппинг сравниваются с HGB без изменения production outputs;
параметр не переносится автоматически без отдельной проверки на 2025.
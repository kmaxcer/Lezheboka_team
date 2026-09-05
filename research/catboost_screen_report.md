# CatBoost screen

 protocol  year kind    n     rmse      mae  seconds
    exact  2019 base  249 0.061358 0.043573      5.0
    exact  2019 deep  249 0.061424 0.043937     48.5
    exact  2020 base  222 0.060914 0.041115      2.0
    exact  2020 deep  222 0.059688 0.040550     26.8
    exact  2021 base  130 0.082257 0.060497      3.6
    exact  2021 deep  130 0.081837 0.060181     52.4
    exact  2022 base  192 0.072093 0.054303      2.3
    exact  2022 deep  192 0.071216 0.052649     40.3
    exact  2023 base  169 0.076209 0.050271      4.7
    exact  2023 deep  169 0.074799 0.049309     41.7
    exact  2024 base  152 0.054565 0.037838      4.1
    exact  2024 deep  152 0.056049 0.039527     36.9
   random    -1 base 4571 0.067224 0.045628      NaN
   random    -1 deep 4571 0.066532 0.044965      NaN
proxy2025  2025 base  756 0.062543 0.041890      NaN
proxy2025  2025 deep  756 0.062574 0.041977      NaN

## pooled
 protocol kind      n  rmse_pooled  mae_pooled
    exact deep 1114.0     0.066980    0.046872
    exact base 1114.0     0.067481    0.047141
proxy2025 base  756.0     0.062543    0.041890
proxy2025 deep  756.0     0.062574    0.041977
   random deep 4571.0     0.066532    0.044965
   random base 4571.0     0.067224    0.045628

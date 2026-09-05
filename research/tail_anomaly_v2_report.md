# Tail/anomaly v2 residual correction

Research-only; production unchanged.

Features use visible primary/climatology residuals around each masked row and same-date peers; status/hidden labels are diagnostics only.

## Pooled cross-fitted results

           protocol       tag     candidate    n  rmse_pooled  mae_pooled  partitions
   exact_hidden_doy  loo_year      baseline 1114     0.062606    0.042855           6
   exact_hidden_doy  loo_year ridge30_s0.25 1114     0.062624    0.042900           6
   exact_hidden_doy  loo_year    hgb8_s0.25 1114     0.062639    0.042857           6
   exact_hidden_doy  loo_year ridge10_s0.25 1114     0.062696    0.042914           6
   exact_hidden_doy  loo_year   hgb16_s0.25 1114     0.062716    0.042933           6
   exact_hidden_doy  loo_year ridge30_s0.50 1114     0.062828    0.043169           6
   exact_hidden_doy  loo_year    hgb8_s0.50 1114     0.062926    0.043166           6
   exact_hidden_doy  loo_year ridge10_s0.50 1114     0.063024    0.043240           6
   exact_hidden_doy  loo_year   hgb16_s0.50 1114     0.063138    0.043381           6
   exact_hidden_doy  loo_year ridge30_s0.75 1114     0.063216    0.043647           6
   exact_hidden_doy  loo_year      ridge100 1114     0.063412    0.044085           6
   exact_hidden_doy  loo_year    hgb8_s0.75 1114     0.063463    0.043757           6
   exact_hidden_doy  loo_year ridge10_s0.75 1114     0.063587    0.043816           6
   exact_hidden_doy  loo_year       ridge30 1114     0.063786    0.044323           6
   exact_hidden_doy  loo_year   hgb16_s0.75 1114     0.063868    0.044142           6
   exact_hidden_doy  loo_year         huber 1114     0.064065    0.044240           6
   exact_hidden_doy  loo_year          hgb8 1114     0.064244    0.044654           6
   exact_hidden_doy  loo_year       ridge10 1114     0.064378    0.044597           6
   exact_hidden_doy  loo_year         hgb16 1114     0.064895    0.045288           6
random_private_like fit_exact ridge30_s0.25 7932     0.069442    0.042608           3
random_private_like fit_exact      baseline 7932     0.069449    0.042557           3
random_private_like fit_exact ridge10_s0.25 7932     0.069488    0.042675           3
random_private_like fit_exact    hgb8_s0.25 7932     0.069488    0.042724           3
random_private_like fit_exact   hgb16_s0.25 7932     0.069499    0.042762           3
random_private_like fit_exact ridge30_s0.50 7932     0.069721    0.043026           3
random_private_like fit_exact    hgb8_s0.50 7932     0.069811    0.043262           3
random_private_like fit_exact   hgb16_s0.50 7932     0.069895    0.043410           3
random_private_like fit_exact ridge10_s0.50 7932     0.069966    0.043303           3
random_private_like fit_exact ridge30_s0.75 7932     0.070283    0.043732           3
random_private_like fit_exact    hgb8_s0.75 7932     0.070414    0.044094           3
random_private_like fit_exact      ridge100 7932     0.070567    0.044079           3
random_private_like fit_exact   hgb16_s0.75 7932     0.070632    0.044420           3
random_private_like fit_exact ridge10_s0.75 7932     0.070875    0.044293           3
random_private_like fit_exact       ridge30 7932     0.071122    0.044685           3
random_private_like fit_exact         huber 7932     0.071236    0.044653           3
random_private_like fit_exact          hgb8 7932     0.071291    0.045203           3
random_private_like fit_exact         hgb16 7932     0.071699    0.045777           3
random_private_like fit_exact       ridge10 7932     0.072198    0.045574           3
random_private_like  loo_seed         hgb16 7932     0.068706    0.042157           3
random_private_like  loo_seed   hgb16_s0.75 7932     0.068719    0.042079           3
random_private_like  loo_seed   hgb16_s0.50 7932     0.068848    0.042117           3
random_private_like  loo_seed          hgb8 7932     0.068926    0.042283           3
random_private_like  loo_seed    hgb8_s0.75 7932     0.068945    0.042226           3
random_private_like  loo_seed    hgb8_s0.50 7932     0.069038    0.042257           3
random_private_like  loo_seed   hgb16_s0.25 7932     0.069091    0.042272           3
random_private_like  loo_seed    hgb8_s0.25 7932     0.069207    0.042365           3
random_private_like  loo_seed ridge10_s0.50 7932     0.069254    0.042425           3
random_private_like  loo_seed ridge30_s0.50 7932     0.069272    0.042427           3
random_private_like  loo_seed ridge10_s0.75 7932     0.069277    0.042512           3
random_private_like  loo_seed ridge30_s0.75 7932     0.069286    0.042498           3
random_private_like  loo_seed ridge10_s0.25 7932     0.069311    0.042434           3
random_private_like  loo_seed ridge30_s0.25 7932     0.069326    0.042443           3
random_private_like  loo_seed      ridge100 7932     0.069368    0.042612           3
random_private_like  loo_seed       ridge30 7932     0.069370    0.042654           3
random_private_like  loo_seed       ridge10 7932     0.069381    0.042696           3
random_private_like  loo_seed      baseline 7932     0.069449    0.042557           3
random_private_like  loo_seed         huber 7932     0.069481    0.042231           3

## exact_hidden_doy

        protocol      tag     candidate    n  rmse_pooled  mae_pooled  partitions
exact_hidden_doy loo_year      baseline 1114     0.062606    0.042855           6
exact_hidden_doy loo_year ridge30_s0.25 1114     0.062624    0.042900           6
exact_hidden_doy loo_year    hgb8_s0.25 1114     0.062639    0.042857           6
exact_hidden_doy loo_year ridge10_s0.25 1114     0.062696    0.042914           6
exact_hidden_doy loo_year   hgb16_s0.25 1114     0.062716    0.042933           6
exact_hidden_doy loo_year ridge30_s0.50 1114     0.062828    0.043169           6
exact_hidden_doy loo_year    hgb8_s0.50 1114     0.062926    0.043166           6
exact_hidden_doy loo_year ridge10_s0.50 1114     0.063024    0.043240           6
exact_hidden_doy loo_year   hgb16_s0.50 1114     0.063138    0.043381           6
exact_hidden_doy loo_year ridge30_s0.75 1114     0.063216    0.043647           6
exact_hidden_doy loo_year      ridge100 1114     0.063412    0.044085           6
exact_hidden_doy loo_year    hgb8_s0.75 1114     0.063463    0.043757           6
exact_hidden_doy loo_year ridge10_s0.75 1114     0.063587    0.043816           6
exact_hidden_doy loo_year       ridge30 1114     0.063786    0.044323           6
exact_hidden_doy loo_year   hgb16_s0.75 1114     0.063868    0.044142           6
exact_hidden_doy loo_year         huber 1114     0.064065    0.044240           6
exact_hidden_doy loo_year          hgb8 1114     0.064244    0.044654           6
exact_hidden_doy loo_year       ridge10 1114     0.064378    0.044597           6
exact_hidden_doy loo_year         hgb16 1114     0.064895    0.045288           6

## random_private_like

           protocol       tag     candidate    n  rmse_pooled  mae_pooled  partitions
random_private_like fit_exact ridge30_s0.25 7932     0.069442    0.042608           3
random_private_like fit_exact      baseline 7932     0.069449    0.042557           3
random_private_like fit_exact ridge10_s0.25 7932     0.069488    0.042675           3
random_private_like fit_exact    hgb8_s0.25 7932     0.069488    0.042724           3
random_private_like fit_exact   hgb16_s0.25 7932     0.069499    0.042762           3
random_private_like fit_exact ridge30_s0.50 7932     0.069721    0.043026           3
random_private_like fit_exact    hgb8_s0.50 7932     0.069811    0.043262           3
random_private_like fit_exact   hgb16_s0.50 7932     0.069895    0.043410           3
random_private_like fit_exact ridge10_s0.50 7932     0.069966    0.043303           3
random_private_like fit_exact ridge30_s0.75 7932     0.070283    0.043732           3
random_private_like fit_exact    hgb8_s0.75 7932     0.070414    0.044094           3
random_private_like fit_exact      ridge100 7932     0.070567    0.044079           3
random_private_like fit_exact   hgb16_s0.75 7932     0.070632    0.044420           3
random_private_like fit_exact ridge10_s0.75 7932     0.070875    0.044293           3
random_private_like fit_exact       ridge30 7932     0.071122    0.044685           3
random_private_like fit_exact         huber 7932     0.071236    0.044653           3
random_private_like fit_exact          hgb8 7932     0.071291    0.045203           3
random_private_like fit_exact         hgb16 7932     0.071699    0.045777           3
random_private_like fit_exact       ridge10 7932     0.072198    0.045574           3
random_private_like  loo_seed         hgb16 7932     0.068706    0.042157           3

Decision: retain only a correction that improves the pooled exact and strict fit_exact random protocols; no outputs/model_dani_tuned* files are modified.
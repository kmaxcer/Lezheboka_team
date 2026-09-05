# PCA mix against production

PCA predictions are observed-only-calibrated diagnostics. Production reference is HGB+lag 80/20.

('safe_pca_rank1_b0.5', 'exact_hidden_doy')
             method  weight_pca    n     rmse      mae          dataset
safe_pca_rank1_b0.5         0.0 1114 0.062606 0.042855 exact_hidden_doy
safe_pca_rank1_b0.5         0.1 1114 0.062773 0.043068 exact_hidden_doy
safe_pca_rank1_b0.5         0.2 1114 0.063060 0.043376 exact_hidden_doy
safe_pca_rank1_b0.5         0.3 1114 0.063466 0.043765 exact_hidden_doy
safe_pca_rank1_b0.5         0.5 1114 0.064623 0.044850 exact_hidden_doy
safe_pca_rank1_b0.5         1.0 1114 0.069374 0.048977 exact_hidden_doy

('safe_pca_rank1_b0.5', 'random_private_like')
             method  weight_pca    n     rmse      mae             dataset
safe_pca_rank1_b0.5         0.0 7932 0.069449 0.042557 random_private_like
safe_pca_rank1_b0.5         0.1 7932 0.069623 0.042759 random_private_like
safe_pca_rank1_b0.5         0.2 7932 0.069938 0.043100 random_private_like
safe_pca_rank1_b0.5         0.3 7932 0.070391 0.043570 random_private_like
safe_pca_rank1_b0.5         0.5 7932 0.071701 0.044837 random_private_like
safe_pca_rank1_b0.5         1.0 7932 0.077139 0.049778 random_private_like

('safe_pca_rank1_b0.5', nan)
             method  weight_pca    n     rmse      mae dataset
safe_pca_rank1_b0.5        0.00 9046 0.068643 0.042593     NaN
safe_pca_rank1_b0.5        0.05 9046 0.068713 0.042679     NaN
safe_pca_rank1_b0.5        0.10 9046 0.068817 0.042797     NaN
safe_pca_rank1_b0.5        0.15 9046 0.068955 0.042948     NaN
safe_pca_rank1_b0.5        0.20 9046 0.069128 0.043134     NaN
safe_pca_rank1_b0.5        0.30 9046 0.069575 0.043594     NaN
safe_pca_rank1_b0.5        0.50 9046 0.070867 0.044838     NaN
safe_pca_rank1_b0.5        1.00 9046 0.076226 0.049679     NaN

('safe_pca_rank2_b0.5', 'exact_hidden_doy')
             method  weight_pca    n     rmse      mae          dataset
safe_pca_rank2_b0.5         0.0 1114 0.062606 0.042855 exact_hidden_doy
safe_pca_rank2_b0.5         0.1 1114 0.062778 0.043065 exact_hidden_doy
safe_pca_rank2_b0.5         0.2 1114 0.063069 0.043373 exact_hidden_doy
safe_pca_rank2_b0.5         0.3 1114 0.063478 0.043759 exact_hidden_doy
safe_pca_rank2_b0.5         0.5 1114 0.064639 0.044837 exact_hidden_doy
safe_pca_rank2_b0.5         1.0 1114 0.069389 0.048864 exact_hidden_doy

('safe_pca_rank2_b0.5', 'random_private_like')
             method  weight_pca    n     rmse      mae             dataset
safe_pca_rank2_b0.5         0.0 7932 0.069449 0.042557 random_private_like
safe_pca_rank2_b0.5         0.1 7932 0.069625 0.042744 random_private_like
safe_pca_rank2_b0.5         0.2 7932 0.069935 0.043062 random_private_like
safe_pca_rank2_b0.5         0.3 7932 0.070380 0.043501 random_private_like
safe_pca_rank2_b0.5         0.5 7932 0.071658 0.044696 random_private_like
safe_pca_rank2_b0.5         1.0 7932 0.076945 0.049372 random_private_like

('safe_pca_rank2_b0.5', nan)
             method  weight_pca    n     rmse      mae dataset
safe_pca_rank2_b0.5        0.00 9046 0.068643 0.042593     NaN
safe_pca_rank2_b0.5        0.05 9046 0.068714 0.042673     NaN
safe_pca_rank2_b0.5        0.10 9046 0.068818 0.042784     NaN
safe_pca_rank2_b0.5        0.15 9046 0.068956 0.042925     NaN
safe_pca_rank2_b0.5        0.20 9046 0.069127 0.043100     NaN
safe_pca_rank2_b0.5        0.30 9046 0.069567 0.043532     NaN
safe_pca_rank2_b0.5        0.50 9046 0.070831 0.044713     NaN
safe_pca_rank2_b0.5        1.00 9046 0.076055 0.049309     NaN

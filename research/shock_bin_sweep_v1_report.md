# Seasonal shock-bin sweep v1

Only visible targets build seasonal profiles; coefficients are leave-partition-out.

 width             dataset method      n  rmse_pooled  rmse_mean  mae_mean
    24    exact_hidden_doy   crop 1114.0     0.061894   0.062316  0.043166
    12    exact_hidden_doy   crop 1114.0     0.061899   0.062340  0.042912
     8    exact_hidden_doy   crop 1114.0     0.062058   0.062488  0.043038
    24    exact_hidden_doy   date 1114.0     0.062232   0.062652  0.043404
    24    exact_hidden_doy  joint 1114.0     0.062304   0.062719  0.043450
    16    exact_hidden_doy   crop 1114.0     0.062305   0.062848  0.043467
    32    exact_hidden_doy   crop 1114.0     0.062329   0.062760  0.043357
    12    exact_hidden_doy   date 1114.0     0.062385   0.062826  0.043441
    45    exact_hidden_doy   crop 1114.0     0.062394   0.062830  0.043438
     8    exact_hidden_doy   date 1114.0     0.062416   0.062875  0.043501
    12    exact_hidden_doy  joint 1114.0     0.062421   0.062867  0.043499
     8    exact_hidden_doy  joint 1114.0     0.062451   0.062902  0.043499
    16    exact_hidden_doy   date 1114.0     0.062452   0.062938  0.043661
    32    exact_hidden_doy  joint 1114.0     0.062493   0.062912  0.043536
    45    exact_hidden_doy   date 1114.0     0.062524   0.062963  0.043528
    32    exact_hidden_doy   date 1114.0     0.062530   0.062948  0.043487
    45    exact_hidden_doy  joint 1114.0     0.062541   0.062982  0.043581
    16    exact_hidden_doy  joint 1114.0     0.062552   0.063030  0.043740
    45    exact_hidden_doy  state 1114.0     0.062583   0.063034  0.043730
    32    exact_hidden_doy  state 1114.0     0.062602   0.063048  0.043728
    12    exact_hidden_doy  state 1114.0     0.062681   0.063134  0.043796
     8    exact_hidden_doy  state 1114.0     0.062681   0.063118  0.043672
    24    exact_hidden_doy  state 1114.0     0.062697   0.063135  0.043780
    16    exact_hidden_doy  state 1114.0     0.062769   0.063206  0.043838
    24 random_private_like   crop 7932.0     0.069074   0.069008  0.042182
     8 random_private_like   crop 7932.0     0.069114   0.069049  0.042302
    16 random_private_like  joint 7932.0     0.069121   0.069056  0.042128
    16 random_private_like   crop 7932.0     0.069130   0.069066  0.042244
     8 random_private_like  joint 7932.0     0.069155   0.069086  0.042289
    32 random_private_like   crop 7932.0     0.069179   0.069114  0.042295
     8 random_private_like   date 7932.0     0.069202   0.069135  0.042381
    24 random_private_like  joint 7932.0     0.069207   0.069138  0.042231
    16 random_private_like   date 7932.0     0.069221   0.069155  0.042304
    12 random_private_like   crop 7932.0     0.069226   0.069159  0.042342
    24 random_private_like   date 7932.0     0.069226   0.069157  0.042294
    12 random_private_like  joint 7932.0     0.069272   0.069206  0.042271
    32 random_private_like  joint 7932.0     0.069281   0.069213  0.042413
    12 random_private_like   date 7932.0     0.069302   0.069234  0.042391
    32 random_private_like   date 7932.0     0.069322   0.069255  0.042400
    45 random_private_like   crop 7932.0     0.069322   0.069253  0.042394
    16 random_private_like  state 7932.0     0.069356   0.069296  0.042391
     8 random_private_like  state 7932.0     0.069406   0.069342  0.042459
    32 random_private_like  state 7932.0     0.069420   0.069358  0.042576
    45 random_private_like  joint 7932.0     0.069429   0.069362  0.042523
    45 random_private_like   date 7932.0     0.069433   0.069367  0.042502
    12 random_private_like  state 7932.0     0.069434   0.069373  0.042429
    24 random_private_like  state 7932.0     0.069437   0.069375  0.042508
    45 random_private_like  state 7932.0     0.069449   0.069387  0.042571

Files: `research/shock_bin_sweep_v1_results.csv`, `research/shock_bin_sweep_v1_aggregate.csv`, `research/shock_bin_sweep_v1_preds.csv`

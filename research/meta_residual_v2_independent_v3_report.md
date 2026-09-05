# meta residual v2 independent v3 audit

Fresh masks 0/1; v3 fit sees target-masked reference only.

  cap  runs      n  baseline_rmse     rmse  delta_rmse  wins
0.005   6.0 2928.0       0.064711 0.064778    0.000068   1.0
0.010   6.0 2928.0       0.064711 0.064953    0.000243   1.0
0.015   6.0 2928.0       0.064711 0.065092    0.000381   1.0
0.020   6.0 2928.0       0.064711 0.065176    0.000465   1.0
0.030   6.0 2928.0       0.064711 0.065304    0.000594   1.0

{
  "mask_seeds": [
    0,
    1
  ],
  "rows": 5288,
  "hidden_rows": 3112,
  "private_sha256": "3c5c0e27eef8266bcf6dce09c9b556c073cee3902c065a94e4ea7a59edb00993",
  "train_sha256": "a75e530d0fb51581ad6800f84b3875233778801491f02236917862faf9b424ec",
  "v3_component": ".7*ext40+.3*v3",
  "seconds": 303.2,
  "production_baseline_overwritten": false
}

No production artifact changed.

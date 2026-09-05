# Meta residual v2 route audit

Only visible-mask context and saved OOF candidates are used. GroupShuffleSplit seeds 0/1/2; groups are AOI or AOI×year.

Top pooled rows:
  route  features         group    model  cap  runs      n     rmse  baseline_rmse  delta_rmse  improved_runs
history   compact     group_aoi  ridge30 0.02   9.0 3578.0 0.063922       0.064510   -0.000589            9.0
history   compact     group_aoi  ridge30 0.01   9.0 3578.0 0.063965       0.064510   -0.000545            9.0
history candidate     group_aoi  ridge30 0.03   9.0 3578.0 0.064050       0.064510   -0.000460            9.0
history candidate     group_aoi  ridge30 0.04   9.0 3578.0 0.064050       0.064510   -0.000460            9.0
history candidate     group_aoi  ridge30 0.06   9.0 3578.0 0.064050       0.064510   -0.000460            9.0
history   compact     group_aoi ridge100 0.02   9.0 3578.0 0.064053       0.064510   -0.000457            9.0
history candidate     group_aoi  ridge30 0.02   9.0 3578.0 0.064054       0.064510   -0.000456            9.0
history candidate     group_aoi  ridge30 0.01   9.0 3578.0 0.064060       0.064510   -0.000450            9.0
history   compact     group_aoi ridge100 0.01   9.0 3578.0 0.064061       0.064510   -0.000450            9.0
history candidate group_aoiyear  ridge30 0.04   9.0 3345.0 0.061331       0.061700   -0.000369            9.0
history candidate group_aoiyear  ridge30 0.06   9.0 3345.0 0.061331       0.061700   -0.000369            9.0
history candidate group_aoiyear  ridge30 0.03   9.0 3345.0 0.061332       0.061700   -0.000369            9.0
history candidate group_aoiyear  ridge30 0.01   9.0 3345.0 0.061350       0.061700   -0.000350            9.0
history candidate group_aoiyear  ridge30 0.02   9.0 3345.0 0.061351       0.061700   -0.000349            9.0
history candidate     group_aoi     hgb8 0.01   9.0 3578.0 0.064199       0.064510   -0.000311            8.0
    all   compact group_aoiyear ridge100 0.01   9.0 4814.0 0.078031       0.078284   -0.000254            8.0
history   compact     group_aoi     hgb8 0.01   9.0 3578.0 0.064260       0.064510   -0.000251            8.0
history   compact     group_aoi     hgb8 0.02   9.0 3578.0 0.064268       0.064510   -0.000242            8.0
    all candidate     group_aoi  ridge30 0.01   9.0 4873.0 0.063688       0.063900   -0.000213            8.0
    all candidate group_aoiyear  ridge30 0.01   9.0 4814.0 0.078133       0.078284   -0.000151            8.0

No production artifact changed.

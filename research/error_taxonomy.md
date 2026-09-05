# Error taxonomy (pseudo-CV)

Private synthetic DOYs projected to train years 2022–2024. n=513, overall
RMSE=0.07666 в исходном аудите. После seasonal source-map и same-year/date
prior патчей полный six-year CV (n=1114) стал 0.07055.

## Gap length

```
           n      rmse
gap_len               
1        487  0.076890
2         26  0.072215
```

## Edge/interior

```
         n      rmse
edge                
False  506  0.076653
True     7  0.077111
```

## Target sensor

```
           n      rmse
source                
landsat  193  0.066494
modis    160  0.080764
s2       160  0.083582
```

## Crop

```
                     n      rmse
crop_type                       
зерновые           139  0.071917
озимая пшеница     283  0.080487
пастбища/зерновые   17  0.066157
подсолнечник        74  0.072424
```

## Recommendations

- Prioritize interior singleton gaps; they dominate.
- Edge and multi-day gaps need seasonal/AOI-history fallback.
- Retain source calibration and robust clipping; report RMSE by sensor/crop.

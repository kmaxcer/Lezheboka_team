from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
D=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date'],low_memory=False); tr['yr']=tr.date.dt.year; tr['doy2']=tr.date.dt.dayofyear
for c in ['primary_ndvi','s2_ndvi','landsat_ndvi','modis_ndvi','s2_evi','s2_ndwi','landsat_evi','landsat_ndwi','modis_evi','era5_temp_c','era5_precip_mm','ndvi_climatology_mean','ndvi_climatology_std']:
 x=pd.to_numeric(tr[c],errors='coerce'); z=x[np.isfinite(x)]
 print(c,'n',len(z),'uniq',z.nunique(),'decimals sample',z.head(5).tolist(),'quant',z.quantile([.001,.01,.1,.5,.9,.99,.999]).round(6).tolist())
# Correlations and linear residual diagnostics among available sources
print('\nCORR')
print(tr[['primary_ndvi','s2_ndvi','landsat_ndvi','modis_ndvi','s2_evi','s2_ndwi','landsat_evi','landsat_ndwi','modis_evi','era5_temp_c','era5_precip_mm','ndvi_climatology_mean','ndvi_climatology_std']].corr(min_periods=100).round(3).to_string())
# target by crop/year and residual after smooth date/crop/year means
tr['crop']=tr.crop_type.astype(str)
known=tr[tr.primary_ndvi.notna()].copy(); known['res']=known.primary_ndvi-known.groupby(['crop','doy2']).primary_ndvi.transform('median');
for key in ['crop','yr','doy2','anon_polygon_id','src']:
 if key=='src': known['src']=np.select([known.s2_ndvi.notna(),known.landsat_ndvi.notna(),known.modis_ndvi.notna()],[0,1,2],-1)
 print('res by',key,known.groupby(key).res.agg(['count','mean','std']).sort_values('std',ascending=False).head(20).round(4).to_string())
# Check if residuals look normal / autocorrelation and quantized
r=known.res.dropna().to_numpy(); print('res stats',stats.describe(r), 'normal p',stats.normaltest(r[:min(len(r),50000)]).pvalue)
for lag in [1,2,3,4,5,7,8,16,32]:
 z=known.sort_values(['anon_polygon_id','date']); z['rr']=z.res; z['lag']=z.groupby(['anon_polygon_id','yr']).rr.shift(lag); q=z[['rr','lag']].dropna(); print('lag',lag,'n',len(q),'corr',q.rr.corr(q.lag))
# Exact duplicate target values / mantissa patterns
print('rounded collision',[(k,int((known.primary_ndvi.round(k).value_counts()>1).sum())) for k in [3,4,5,6,7,8]])

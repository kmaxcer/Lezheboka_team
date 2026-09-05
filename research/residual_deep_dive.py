from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
f=pd.read_csv(R/'exact_compare_preds.csv',parse_dates=['date'])
f['res_hgb']=f.hgb-f._truth; f['res_lag']=f.lag_k16_d3-f._truth; f['src']=f._true_src.astype(str); f['doy']=f.date.dt.dayofyear; f['bin']=(f.doy//8).astype(int)
for col in ['hgb','lag_k16_d3','base_k8']:
 f['e']=f[col]-f._truth
 print('\n',col,'overall',np.sqrt(np.mean(f.e**2)))
 print('src',f.groupby('src').e.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),bias='mean',sd='std').to_string())
 print('year',f.groupby('year').e.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),bias='mean').to_string())
 print('bin top abs',f.groupby('bin').e.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),bias='mean').sort_values('rmse',ascending=False).head(12).to_string())
 print('aoi top',f.groupby('anon_polygon_id').e.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),bias='mean').sort_values('rmse',ascending=False).head(10).to_string())
print('\ncorrelations',f[['res_hgb','res_lag','_truth']].corr())
print('residual hgb by src/year')
print(f.pivot_table(index='year',columns='src',values='res_hgb',aggfunc=['mean','count']).round(4).to_string())
f.to_csv(R/'residual_deep_dive_rows.csv',index=False)

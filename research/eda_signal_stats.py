from pathlib import Path
import numpy as np, pandas as pd
D=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date'],low_memory=False)
pr=pd.read_csv(D/'private_features.csv',parse_dates=['date'],low_memory=False)
def report(name,d):
 d=d.copy(); d['yr']=d.date.dt.year; d['dd']=d.date.dt.dayofyear
 d['src']=np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],[0,1,2],-1)
 print('\n###',name,'###')
 print('rows',len(d),'ids',d.anon_polygon_id.nunique(),'target',d.primary_ndvi.notna().sum())
 print('crop',d.groupby('crop_type').size().to_dict())
 print('source',d.groupby('src').size().to_dict())
 print('target quant',d.primary_ndvi.quantile([0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).to_dict())
 # adjacent differences among observed target in each aoi-year sorted dates
 d=d.sort_values(['anon_polygon_id','date']); d['dy']=d.groupby(['anon_polygon_id','yr']).primary_ndvi.diff(); d['dt']=d.groupby(['anon_polygon_id','yr']).date.diff().dt.days
 z=d[d.primary_ndvi.notna() & d.dy.notna()]
 print('adj n',len(z),'dt',z.dt.describe().to_dict(),'absdy quant',z.dy.abs().quantile([.5,.75,.9,.95,.99]).to_dict(),'rmse dy',np.sqrt(np.mean(z.dy**2)))
 # same doy across years
 g=d[d.primary_ndvi.notna()].groupby(['anon_polygon_id','dd']).primary_ndvi.agg(['count','std','mean'])
 print('same doy groups',len(g),'multi',int((g['count']>1).sum()),'std median',g.loc[g['count']>1,'std'].median(),'mean',g.loc[g['count']>1,'std'].mean())
 # year-level means/std
 print('aoi-year target count quant',d[d.primary_ndvi.notna()].groupby(['anon_polygon_id','yr']).size().quantile([0,.1,.5,.9,1]).to_dict())
 print('year mean',d[d.primary_ndvi.notna()].groupby('yr').primary_ndvi.mean().round(4).to_dict())
 # source target stats by year
 print('src/year',d[d.primary_ndvi.notna()].pivot_table(index='yr',columns='src',values='primary_ndvi',aggfunc=['count','mean']).round(4).to_string())
 return d
a=report('train',tr); b=report('private',pr)
# Date schedules: count rows and source composition by doy/year
for name,d in [('tr',a),('pr',b)]:
 z=d[d.primary_ndvi.notna()].groupby(['yr','dd','src']).size().unstack(fill_value=0)
 print(name,'date groups',len(z),'top counts',z.sum(axis=1).describe().to_dict())
 print(name,'doy top',z.sum(axis=1).sort_values(ascending=False).head(30).to_dict())
 # source entropy / dominant source
 print(name,'source dominant share', (z.max(axis=1)/z.sum(axis=1)).describe().to_dict())

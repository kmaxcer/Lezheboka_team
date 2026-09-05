from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
ROOT=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
def main():
 tr=pd.read_csv(ROOT/'train_dataset.csv',parse_dates=['date']); tr['_year']=tr.date.dt.year;tr['_doy']=tr.date.dt.dayofyear
 tr['_src']=np.select([tr.s2_ndvi.notna(),tr.landsat_ndvi.notna(),tr.modis_ndvi.notna()],['s2','l8','mod'],'none')
 # residual from aoi+year+doy-bin means and examine cross-AOI same-date correlation
 z=tr[tr.primary_ndvi.notna()].copy(); z['_bin']=z._doy//4
 z['_mean']=z.groupby(['anon_polygon_id','_year','_bin']).primary_ndvi.transform('median'); z['_res']=z.primary_ndvi-z['_mean']
 p=z.pivot_table(index='date',columns='anon_polygon_id',values='_res',aggfunc='mean'); cc=p.corr(min_periods=30); vals=cc.where(~np.eye(len(cc),dtype=bool)).stack(); print('cross aoi residual corr',vals.describe().to_dict(),'top',vals.sort_values(ascending=False).head(20).to_dict())
 # same aoi residual correlation by year at same DOY (using medians)
 q=z.pivot_table(index=['anon_polygon_id','_doy'],columns='_year',values='_res',aggfunc='median'); c=q.corr(min_periods=20); vv=c.where(~np.eye(len(c),dtype=bool)).stack(); print('cross year residual corr',vv.describe().to_dict(),'top',vv.sort_values(ascending=False).head(20).to_dict())
 # source-specific residual correlation across same-date rows
 for s in ['s2','l8','mod']:
  q=z[z._src==s].pivot_table(index='date',columns='anon_polygon_id',values='_res',aggfunc='mean'); c=q.corr(min_periods=20);v=c.where(~np.eye(len(c),dtype=bool)).stack(); print(s,'n',len(q),'corr mean/median/top',v.mean(),v.median(),v.max())
 # residual autocorrelation by exact day lags within aoi/year for each source
 for s in ['s2','l8','mod']:
  vals=[]
  for (pid,y),g in z[z._src==s].groupby(['anon_polygon_id','_year']):
   g=g.sort_values('date'); a=g['_res'].to_numpy(); d=g.date.diff().dt.days.to_numpy();
   for lag in [1,2,3,4,5,7,8,16,32]:
    # pairs whose date difference lag
    for i in range(1,len(g)):
     if d[i]==lag: vals.append((lag,a[i-1],a[i]))
  vv=pd.DataFrame(vals,columns=['lag','a','b']); print(s,'lag corr',vv.groupby('lag').apply(lambda g:g.a.corr(g.b)).to_dict(),'n',vv.groupby('lag').size().to_dict())
 # duplicate exact values of residual/date across IDs and source
 print('target rounded value frequencies',z.primary_ndvi.round(6).value_counts().head(20).to_dict())
if __name__=='__main__':main()

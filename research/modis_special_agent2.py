"""Evaluate MODIS-specific interpolation on the periodic hidden dates."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import SOURCES,SENSOR_COL,_prepare,_fit_source_maps,_local_source_prediction,predict_private
from validate import make_fold

PERIOD=set([97,113,129,145,161,177,193,209,225,241,257,273,289])
def pred(fold):
 d=_prepare(fold);syn=fold.is_synthetic_gap.to_numpy(bool);known=np.isfinite(d.primary_ndvi.to_numpy(float));y=d.primary_ndvi.to_numpy(float);x=d._ord.to_numpy(float);src=d._src.to_numpy(object);maps=_fit_source_maps(d,known,30);out={'prod':predict_private(fold).primary_ndvi_pred.to_numpy(float)}
 # raw MODIS values at any rows, and primary rows converted to MODIS
 raw=d.modis_ndvi.to_numpy(float); modok=np.isfinite(raw)
 for typ in ['raw','converted']:
  for k in [2,3,4,6,8,12,16]:
   arr=[]
   for _,idx in d.groupby(['anon_polygon_id','_year']).groups.items():
    ii=np.asarray(idx,dtype=int)
    for q in ii[syn[ii]]:
     if typ=='raw':
      kk=ii[modok[ii]]; yy=raw; ss=np.full(len(d),'modis',object)
     else:kk=ii[known[ii]];yy=y;ss=src
     arr.append(_local_source_prediction(x[q],kk,x,yy,ss,'modis',maps,int(d._doy.iat[q]),30,k))
   out[f'{typ}{k}']=np.array(arr)
 return out, d
def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);src0=np.select([tr.s2_ndvi.notna(),tr.landsat_ndvi.notna(),tr.modis_ndvi.notna()],['s2','landsat','modis'],'none'); rec=[]
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr);o,d=pred(f); q=f.is_synthetic_gap.to_numpy(bool); ds=d._doy.to_numpy(); truth=t.to_numpy(); true_src=src0[f.index.to_numpy()][q];
  for subset,name in [(np.ones(len(truth),bool),'all'),(np.isin(ds[q],list(PERIOD)),'period'),(true_src=='modis','true_modis'),(np.isin(ds[q],list(PERIOD))&(true_src=='modis'),'period_modis')]:
   for meth,p in o.items():
    e=p[subset]-truth[subset];rec.append((yr,name,meth,len(e),np.sqrt(np.nanmean(e*e))))
 out=pd.DataFrame(rec,columns=['year','subset','method','n','rmse']);print(out[out.year=='all'] if False else out.groupby(['subset','method']).apply(lambda z:pd.Series({'n':z.n.sum(),'rmse':np.sqrt(np.average(z.rmse**2,weights=z.n))})).reset_index().sort_values(['subset','rmse']).to_string(index=False));out.to_csv(ROOT/'research'/'modis_special_agent2.csv',index=False)
if __name__=='__main__':main()

from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
def main():
 tr=pd.read_csv(ROOT/'train_dataset.csv',parse_dates=['date']); pr=pd.read_csv(ROOT/'private_features.csv',parse_dates=['date'])
 tr['_year']=tr.date.dt.year; tr['_doy']=tr.date.dt.dayofyear
 ids=sorted(tr.anon_polygon_id.unique()); pv=tr.pivot(index='date',columns='anon_polygon_id',values='era5_temp_c').reindex(columns=ids)
 # exact signatures based on all available values; group by pairwise max abs diff
 rem=set(ids); groups=[]
 while rem:
  i=sorted(rem)[0]; g=[j for j in sorted(rem) if np.array_equal(pv[i].to_numpy(),pv[j].to_numpy(),equal_nan=True)]
  groups.append(g); rem-=set(g)
 print('exact temp groups',groups)
 # identify near groups by max error and corr
 rows=[]
 for i in ids:
  for j in ids:
   if i>=j:continue
   a=pv[i];b=pv[j]; ok=a.notna()&b.notna(); diff=np.abs(a[ok]-b[ok]); rows.append((i,j,float(diff.max()) if len(diff) else np.nan,float(diff.mean()) if len(diff) else np.nan,float(a[ok].corr(b[ok])) if len(diff)>2 else np.nan))
 print('near temp pairs'); print(pd.DataFrame(rows,columns=['i','j','max','mae','corr']).sort_values(['mae']).head(80).to_string(index=False))
 # aggregate target co-movement by weather groups / pair differences after crop adjustment
 tr['_src']=np.select([tr.s2_ndvi.notna(),tr.landsat_ndvi.notna(),tr.modis_ndvi.notna()],['s2','l8','mod'],'none')
 for g in groups:
  if len(g)<2:continue
  z=tr[tr.anon_polygon_id.isin(g)&tr.primary_ndvi.notna()].pivot(index='date',columns='anon_polygon_id',values='primary_ndvi')
  print('group',g,'target corr',z.corr().round(3).to_dict(),'n',z.notna().sum().to_dict())
  # same crop only pairs
 # private rows: map exact temp signature of new IDs against train where overlapping history
 print('private ids',len(pr.anon_polygon_id.unique()))
 for c in ['era5_temp_c','era5_precip_mm']:
  pp=pr[pr.date.dt.year<2025].pivot(index='date',columns='anon_polygon_id',values=c)
  print(c,'private duplicate signatures among full ids')
  pids=list(pp.columns); out=[]
  for ix,i in enumerate(pids):
   for j in pids[ix+1:]:
    a=pp[i];b=pp[j];ok=a.notna()&b.notna();
    if ok.sum()>100:
     dd=np.abs(a[ok]-b[ok]); out.append((i,j,int(ok.sum()),float(dd.max()),float(dd.mean())))
  print(pd.DataFrame(out,columns=['i','j','n','max','mae']).sort_values('mae').head(50).to_string(index=False))
if __name__=='__main__':main()

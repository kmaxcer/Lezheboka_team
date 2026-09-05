from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'research'; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(R)); from teammate_sweep_postcorr import _mask_private
ID,DATE,TARGET='anon_polygon_id','date','primary_ndvi'; SEEDS=(0,1,2,70404)
def rm(y,p):
 o=np.isfinite(y)&np.isfinite(p); return float(np.sqrt(np.mean((np.asarray(p)[o]-np.asarray(y)[o])**2)))
def circ(a,b):
 d=np.abs(np.asarray(a)-b); return np.minimum(d,366-d)
def seasonal(frame,qkeys,window=21,kind='mean',exclude_year=True,min_n=2):
 d=frame.copy(); d[DATE]=pd.to_datetime(d[DATE]); d['_year']=d[DATE].dt.year.to_numpy(); d['_doy']=d[DATE].dt.dayofyear.to_numpy(float); d['_y']=pd.to_numeric(d[TARGET],errors='coerce')
 groups={}
 for a,g in d[np.isfinite(d['_y'])].groupby(ID,sort=False): groups[a]=(g['_doy'].to_numpy(float),g['_year'].to_numpy(int),g['_y'].to_numpy(float))
 cropgroups={}
 if 'crop_type' in d:
  for c,g in d[np.isfinite(d['_y'])].groupby('crop_type',sort=False): cropgroups[c]=(g['_doy'].to_numpy(float),g['_year'].to_numpy(int),g['_y'].to_numpy(float))
 glob=(d.loc[np.isfinite(d['_y']),'_doy'].to_numpy(float),d.loc[np.isfinite(d['_y']),'_year'].to_numpy(int),d.loc[np.isfinite(d['_y']),'_y'].to_numpy(float))
 out=[]
 for _,r in qkeys.iterrows():
  a=r[ID]; dt=pd.Timestamp(r[DATE]); y=int(dt.year); doy=float(dt.dayofyear); arr=groups.get(a)
  def calc(ar):
   if ar is None:return np.nan
   xyr,yrs,vals=ar; ds=circ(xyr,doy); ok=ds<=window
   if exclude_year: ok &= yrs!=y
   if ok.sum()<min_n:return np.nan
   z=vals[ok]; dist=ds[ok]; w=np.exp(-0.5*(dist/max(window/2,1))**2)
   if kind=='median': return float(np.median(z))
   if kind=='trimmed':
    lo,hi=np.quantile(z,[.15,.85]); z=z[(z>=lo)&(z<=hi)]; return float(np.mean(z)) if len(z) else np.nan
   return float(np.sum(w*z)/np.sum(w))
  v=calc(arr)
  if not np.isfinite(v) and 'crop_type' in r: v=calc(cropgroups.get(r['crop_type']))
  if not np.isfinite(v): v=calc(glob)
  out.append(v)
 return np.asarray(out,float)
def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
 rr=pd.read_csv(R/'source_expert_route_v2_fixed_radius_trainaug_rows.csv',parse_dates=[DATE],low_memory=False); rr['seed']=rr.seed.astype(int)
 sp=pd.read_csv(R/'source_schedule_route_probe_rows.csv',parse_dates=[DATE],usecols=[ID,DATE,'seed','sp_crop_2_n','sp_crop_8_n'],low_memory=False); sp['seed']=sp.seed.astype(int)
 rr=rr.merge(sp,on=[ID,DATE,'seed'],how='left',validate='one_to_one'); n2=rr.sp_crop_2_n.fillna(0).to_numpy(float); n8=rr.sp_crop_8_n.fillna(0).to_numpy(float); near=n2>0; mid=(~near)&(n8>0); aa=np.where(near,.5,np.where(mid,.4,.3)); yy=rr.year.to_numpy(int); co=rr.cohort.astype(str).to_numpy(); aa=np.where((co=='new')&(yy==2025),.6,aa); aa=np.where((co=='shared')&(yy==2025),.35,aa); rr['route']=rr.baseline.to_numpy(float)+aa*(rr.expert_trainaug_r2.to_numpy(float)-rr.baseline.to_numpy(float))
 gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=[DATE]); base=pd.read_csv(ROOT/'outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_cal0148_20260905_submission.csv',parse_dates=[DATE]); rows=[]
 for s in SEEDS:
  f,m=_mask_private(pr,s); q=rr[rr.seed.eq(s)].copy(); qkeys=q[[ID,DATE]].reset_index(drop=True); combo=pd.concat([tr,f],ignore_index=True,sort=False)
  for w in [7,14,21,30,45]:
   for kind in ['mean','median','trimmed']:
    p=seasonal(combo,qkeys,w,kind); y=q.truth.to_numpy(float); b=q.route.to_numpy(float); ok=np.isfinite(p)
    rows.append(dict(scope=f'seed{s}',seed=s,window=w,kind=kind,n=len(q),coverage=float(ok.mean()),seasonal_rmse=rm(y,p),base_rmse=rm(y,b),blend05=rm(y,np.where(ok,.95*b+.05*p,b)),blend10=rm(y,np.where(ok,.9*b+.1*p,b)),blend20=rm(y,np.where(ok,.8*b+.2*p,b))))
  print('seed',s,'done',flush=True)
 combo=pd.concat([tr,pr],ignore_index=True,sort=False); keys=gt[[ID,DATE]]
 for w in [7,14,21,30,45]:
  for k in ['mean','median','trimmed']:
   p=seasonal(combo,keys,w,k); z=keys.copy(); z['sp']=p; z=z.merge(gt,on=[ID,DATE],validate='one_to_one').merge(base,on=[ID,DATE],validate='one_to_one'); y=z.primary_ndvi_true.to_numpy(float); b=z.primary_ndvi_pred.to_numpy(float); q=dict(scope='released',seed=-1,window=w,kind=k,n=len(z),coverage=float(np.isfinite(p).mean()),seasonal_rmse=rm(y,p),base_rmse=rm(y,b),blend05=rm(y,np.where(np.isfinite(p),.95*b+.05*p,b)),blend10=rm(y,np.where(np.isfinite(p),.9*b+.1*p,b)),blend20=rm(y,np.where(np.isfinite(p),.8*b+.2*p,b))); rows.append(q)
 # pooled outer from stored per seed cannot combine RMSE directly; rerun chosen grid
 for w in [7,14,21,30,45]:
  for k in ['mean','median','trimmed']:
   ps=[];ys=[];bs=[]
   for s in SEEDS:
    f,m=_mask_private(pr,s); q=rr[rr.seed.eq(s)]; p=seasonal(pd.concat([tr,f],ignore_index=True,sort=False),q[[ID,DATE]],w,k); ps.extend(p);ys.extend(q.truth);bs.extend(q.route)
   p=np.asarray(ps); y=np.asarray(ys,float); b=np.asarray(bs,float); ok=np.isfinite(p); rows.append(dict(scope='pooled',seed=-1,window=w,kind=k,n=len(y),coverage=float(ok.mean()),seasonal_rmse=rm(y,p),base_rmse=rm(y,b),blend05=rm(y,np.where(ok,.95*b+.05*p,b)),blend10=rm(y,np.where(ok,.9*b+.1*p,b)),blend20=rm(y,np.where(ok,.8*b+.2*p,b))))
 met=pd.DataFrame(rows); met.to_csv(R/'seasonal_route_probe_20260905.csv',index=False,float_format='%.9f'); print(met[met.scope.isin(['pooled','released'])].sort_values('blend10').to_string(index=False)); (R/'seasonal_route_probe_20260905_report.md').write_text('# Seasonal AOI route probe\n\nLeakage-safe AOI by day-of-year template, excluding query year.\n\n'+met.to_string(index=False)+'\n\nNo candidate materialized; no upload.\n',encoding='utf-8')
if __name__=='__main__':main()

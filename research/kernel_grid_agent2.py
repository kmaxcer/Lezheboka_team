"""Kernel/robust smoothers for local latent NDVI interpolation."""
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import SOURCES,_prepare,_fit_source_maps,_mode_posteriors,_query_posterior
from validate import make_fold

def kval(dx,yy,kind,h):
    d=np.abs(dx).astype(float); z=yy.astype(float)
    if len(z)==0:return np.nan
    if kind=='gauss':w=np.exp(-0.5*(d/h)**2)
    elif kind=='laplace':w=np.exp(-d/h)
    elif kind=='tricube':
      u=d/h;w=np.where(u<1,(1-u**3)**3,0.)
    elif kind=='inv':w=1/(1+d/h)**2
    else:w=1/(1+d/h)
    if not np.any(w>1e-8):
      j=np.argmin(d);return float(z[j])
    # local constant robust weighted center; trim gross tails once
    v=float(np.average(z,weights=w))
    if len(z)>=6:
      r=np.abs(z-v); med=np.median(r); good=r<=max(3*med,0.04)
      if good.sum()>=2:v=float(np.average(z[good],weights=w[good]))
    return v

def collect(fold,kinds,hs):
 d=_prepare(fold); syn=fold.is_synthetic_gap.astype(bool).to_numpy(); known=np.isfinite(d.primary_ndvi.to_numpy(float)); y=d.primary_ndvi.to_numpy(float);x=d._ord.to_numpy(float);src=d._src.to_numpy(object);maps=_fit_source_maps(d,known,bin_days=30);aoi,crop,glob,date=_mode_posteriors(d,known);names=[f'{kind}_{h}' for kind in kinds for h in hs]; pred={n:np.full(len(d),np.nan) for n in names};
 for _,idx in d.groupby(['anon_polygon_id','_year'],sort=False).groups.items():
  ii=np.asarray(idx,dtype=int);kk=ii[known[ii]]
  for q in ii[syn[ii]]:
   p=_query_posterior(d,int(q),aoi,crop,glob,date); arr={n:[] for n in names}
   for s,w in zip(SOURCES,p):
    yy=[];xx=[]
    for j in kk:
      a,b=maps.get((s,str(src[j]),int(d._doy.iat[q]//30)),maps.get((s,str(src[j]),'g'),(0,1)));yy.append(a+b*y[j]);xx.append(x[j]-x[q])
    yy=np.array(yy);xx=np.array(xx)
    for kind in kinds:
     for h in hs:
      v=kval(xx,yy,kind,float(h));
      if np.isfinite(v):arr[f'{kind}_{h}'].append((v,w,s))
   for n,vals in arr.items():
    if vals:pred[n][q]=float(np.average([v for v,w,s in vals],weights=[w for v,w,s in vals]))
 for n,pp in pred.items():
  for q in np.flatnonzero(syn&~np.isfinite(pp)):
   same=np.flatnonzero(known&(d.anon_polygon_id.to_numpy()==d.anon_polygon_id.iat[q]));pp[q]=y[same[np.argmin(np.abs(x[same]-x[q]))]] if len(same) else np.nanmedian(y[known])
  pred[n]=pp[syn]
 return pred,fold.loc[syn,'_truth'].to_numpy(float)

def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);kinds=['gauss','laplace','tricube','inv'];hs=[2,3,4,5,7,10,14,21,30,45,60,90];rec={f'{k}_{h}':[] for k in kinds for h in hs}
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr);pp,yy=collect(f,kinds,hs)
  for n,p in pp.items():rec[n].extend((p-yy).tolist())
 out=[]
 for n,e in rec.items():e=np.array(e);out.append((n,len(e),float(np.sqrt(np.mean(e*e))),float(np.mean(np.abs(e)))))
 out=pd.DataFrame(out,columns=['method','n','rmse','mae']).sort_values('rmse');out.to_csv(ROOT/'research'/'kernel_grid_agent2.csv',index=False);print(out.head(30).to_string(index=False))
if __name__=='__main__':main()

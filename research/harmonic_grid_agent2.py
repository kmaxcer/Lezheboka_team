"""Harmonic/seasonal regressions as a low-cost cross-year fallback."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import predict_private
from validate import make_fold

def design(d, harmonics=4, aoi=True, crop=True, year=True, interaction=False):
    doy=d.date.dt.dayofyear.to_numpy(float); ang=2*np.pi*doy/365.25; X=[np.ones(len(d))]
    # centered linear/quadratic seasonal trend plus Fourier
    for k in range(1,harmonics+1): X.extend([np.sin(k*ang),np.cos(k*ang)])
    if harmonics>=6: X.extend([((doy-200)/100),((doy-200)/100)**2])
    if crop:
      cs=pd.Categorical(d.crop_type); C=np.eye(len(cs.categories))[cs.codes]; X.extend([C[:,j] for j in range(1,C.shape[1])])
    if aoi:
      cs=pd.Categorical(d.anon_polygon_id); C=np.eye(len(cs.categories))[cs.codes]; X.extend([C[:,j] for j in range(1,C.shape[1])])
    if year:
      cs=pd.Categorical(d.date.dt.year); C=np.eye(len(cs.categories))[cs.codes]; X.extend([C[:,j] for j in range(1,C.shape[1])])
    if interaction:
      # AOI-specific seasonal phase/amplitude, compact first two harmonics
      cs=pd.Categorical(d.anon_polygon_id); C=np.eye(len(cs.categories))[cs.codes]
      for j in range(C.shape[1]):
       for f in [np.sin(ang),np.cos(ang),np.sin(2*ang),np.cos(2*ang)]: X.append(C[:,j]*f)
    return np.column_stack(X)

def pred(fold, harmonics=4, ridge=1e-2, aoi=True,crop=True,year=True,interaction=False):
 d=fold.copy(); syn=d.is_synthetic_gap.astype(bool).to_numpy(); known=d.primary_ndvi.notna().to_numpy(); y=d.primary_ndvi.to_numpy(float); X=design(d,harmonics,aoi,crop,year,interaction); # avoid NaNs
 X0=np.nan_to_num(X); idx=np.flatnonzero(known); q=np.flatnonzero(syn); A=X0[idx]; b=y[idx]; reg=ridge*np.eye(A.shape[1]); reg[0,0]=0
 coef=np.linalg.solve(A.T@A+reg,A.T@b); ph=X0[q]@coef; return ph

def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 folds=[]
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr); folds.append((f,t.to_numpy(float),predict_private(f).primary_ndvi_pred.to_numpy(float)))
 rec=[]
 for h in [1,2,3,4,6,8,10]:
  for inter in [False,True]:
   for r in [.001,.01,.1,1]:
    es=[]; blends={a:[] for a in [0,.1,.25,.5,.75,1]}
    for f,yy,base in folds:
      ph=pred(f,h,r,interaction=inter); es.extend((ph-yy).tolist());
      for a in blends: blends[a].extend(((1-a)*base+a*ph-yy).tolist())
    rec.append((h,inter,r,'harm',len(es),float(np.sqrt(np.mean(np.array(es)**2))),float(np.mean(np.abs(es)))))
    for a,e in blends.items(): rec.append((h,inter,r,f'blend{a}',len(e),float(np.sqrt(np.mean(np.array(e)**2))),float(np.mean(np.abs(e)))))
 out=pd.DataFrame(rec,columns=['h','interaction','ridge','method','n','rmse','mae']).sort_values('rmse');out.to_csv(ROOT/'research'/'harmonic_grid_agent2.csv',index=False);print(out.head(40).to_string(index=False))
if __name__=='__main__':main()

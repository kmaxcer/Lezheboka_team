"""Supervised residual/meta-model over interpolation context.

The challenge masks every dynamic field on a query row.  This experiment
therefore derives features strictly from date, crop/ID and *other* observed
rows, then learns a small ridge correction to the calibrated interpolator.
It is kept separate from production until time-block CV demonstrates a gain.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from infer import (SOURCES,_prepare,_fit_source_maps,_local_source_prediction,
                   _mode_posteriors,_query_posterior,predict_private)
from validate import make_fold

def _one_features(fold: pd.DataFrame) -> tuple[pd.DataFrame,np.ndarray]:
    d=_prepare(fold); syn=fold.is_synthetic_gap.astype(bool).to_numpy(); known=np.isfinite(d.primary_ndvi.to_numpy(float)); y=d.primary_ndvi.to_numpy(float); x=d._ord.to_numpy(float); src=d._src.to_numpy(object)
    maps=_fit_source_maps(d,known,30); aoi,crop,glob,date=_mode_posteriors(d,known)
    # Per-date peer summaries, excluding masked rows by construction.
    z=pd.DataFrame({'date':d.date,'pid':d.anon_polygon_id,'y':y,'known':known})
    obs=z[known]; sm=obs.groupby('date').y.sum(); ct=obs.groupby('date').y.count();
    asum=obs.groupby('pid').y.sum(); acnt=obs.groupby('pid').y.count()
    # all-year AOI robust levels
    amed=obs.groupby('pid').y.median();
    qrows=[]; qtruth=[]
    for _,idx in d.groupby(['anon_polygon_id','_year'],sort=False).groups.items():
      ii=np.asarray(idx,dtype=int); kk=ii[known[ii]]
      for q in ii[syn[ii]]:
        p=_query_posterior(d,int(q),aoi,crop,glob,date)
        vals=[]; candidates=[]
        for s,w in zip(SOURCES,p):
          v=_local_source_prediction(x[q],kk,x,y,src,s,maps,int(d._doy.iat[q]),30,8)
          candidates.append(v)
          if np.isfinite(v): vals.append((v,float(w)))
        prod=float(np.average([v for v,w in vals],weights=[w for v,w in vals])) if vals else np.nan
        # nearest left/right primary values and local geometry
        if len(kk):
          dx=x[kk]-x[q]; left=kk[dx<0]; right=kk[dx>0]
          il=left[np.argmin(abs(x[left]-x[q]))] if len(left) else -1
          ir=right[np.argmin(abs(x[right]-x[q]))] if len(right) else -1
          lv=y[il] if il>=0 else np.nan; rv=y[ir] if ir>=0 else np.nan
          dl=abs(x[il]-x[q]) if il>=0 else np.nan; dr=abs(x[ir]-x[q]) if ir>=0 else np.nan
          if np.isfinite(lv) and np.isfinite(rv): br=(lv*dr+rv*dl)/(dl+dr)
          elif np.isfinite(lv): br=lv
          elif np.isfinite(rv): br=rv
          else: br=prod
          # local observed dispersion / slope
          near=kk[np.argsort(abs(dx))[:min(8,len(kk))]]; sd=float(np.nanstd(y[near])) if len(near)>1 else 0.
          slope=((rv-lv)/(dl+dr)) if np.isfinite(lv) and np.isfinite(rv) and dl+dr>0 else 0.
        else: lv=rv=dl=dr=br=sd=slope=np.nan
        dt=d.date.iat[q]; date_sum=sm.get(dt,np.nan)-asum.get(d.anon_polygon_id.iat[q],0.); date_n=ct.get(dt,0)-acnt.get(d.anon_polygon_id.iat[q],0.); peer=date_sum/date_n if date_n>0 else np.nan
        aoi_med=amed.get(d.anon_polygon_id.iat[q],np.nan)
        doy=float(d._doy.iat[q]); ang=2*np.pi*doy/365.25
        # Keep a compact, generalizable feature set; categorical strings are
        # encoded by stable integer IDs and one-hot later in fit().
        qrows.append([prod,*candidates,*p,br,lv,rv,dl,dr,sd,slope,peer,aoi_med,
                      np.sin(ang),np.cos(ang),np.sin(2*ang),np.cos(2*ang),
                      float(d._year.iat[q]),float(str(d.anon_polygon_id.iat[q]).split('-')[-1]),
                      str(d.crop_type.iat[q])])
        # ``primary_ndvi`` is masked on the query; validation truth is kept
        # in the private helper column and never used as a feature.
        qtruth.append(float(fold['_truth'].iat[q]) if '_truth' in fold else np.nan)
    cols=['prod','p_s2','p_ls','p_md','post_s2','post_ls','post_md','bracket','left','right','dl','dr','local_sd','slope','peer','aoi_med','sin1','cos1','sin2','cos2','year','aoi_num','crop']
    return pd.DataFrame(qrows,columns=cols),np.asarray(qtruth,float)

def _encode(frames):
    allf=pd.concat(frames,ignore_index=True); crops=sorted(allf.crop.astype(str).unique());
    out=[]
    numeric=['prod','p_s2','p_ls','p_md','post_s2','post_ls','post_md','bracket','left','right','dl','dr','local_sd','slope','peer','aoi_med','sin1','cos1','sin2','cos2','year','aoi_num']
    for _,r in allf.iterrows():
      vals=[float(r[c]) if np.isfinite(r[c]) else 0.0 for c in numeric]
      vals += [1.0 if str(r.crop)==c else 0.0 for c in crops]
      out.append(vals)
    return np.asarray(out,float),numeric,crops

def ridge_fit(X,y,lam):
    X=np.column_stack([np.ones(len(X)),X]); A=X.T@X+lam*np.eye(X.shape[1]);A[0,0]=1e-8
    return np.linalg.solve(A,X.T@y)
def ridge_pred(X,b): return np.column_stack([np.ones(len(X)),X])@b

def main():
    b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']);pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']); years=[2019,2020,2021,2022,2023,2024]; fs=[]; ys=[]
    for yr in years:
      f,t=make_fold(tr,pr,yr); q,yy=_one_features(f);q['fold']=yr;fs.append(q);ys.append(yy);print('fold',yr,len(q),flush=True)
    allf=pd.concat(fs,ignore_index=True); y=np.concatenate(ys); X,nums,crops=_encode([allf.drop(columns='fold')]);
    # Evaluate time-block ridge (train on other years) and direct baseline.
    rec=[]
    for lam in [0.001,.01,.1,1,10,100,1000]:
      for yr in years:
       te=allf.fold.to_numpy()==yr; b0=ridge_fit(X[~te],y[~te],lam); ph=ridge_pred(X[te],b0); e=ph-y[te]; rec.append((lam,yr,len(e),float(np.sqrt(np.mean(e*e)))))
      b0=ridge_fit(X,y,lam);ph=ridge_pred(X,b0);rec.append((lam,'all',len(y),float(np.sqrt(np.mean((ph-y)**2)))))
    out=pd.DataFrame(rec,columns=['lambda','year','n','rmse']);print(out.to_string(index=False));out.to_csv(ROOT/'research'/'meta_model_agent2.csv',index=False);allf.to_csv(ROOT/'research'/'meta_features_agent2.csv',index=False)
if __name__=='__main__':main()

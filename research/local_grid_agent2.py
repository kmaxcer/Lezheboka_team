"""Grid search local source-aware temporal estimators on pseudo masks."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from infer import SOURCES,_prepare,_fit_source_maps,_mode_posteriors,_query_posterior
from validate import make_fold

def cand(xq,dx,yy,kind,k):
    if len(yy)==0:return np.nan
    o=np.argsort(np.abs(dx))[:min(k,len(yy))]; zdx=dx[o].astype(float); z=yy[o].astype(float)
    if len(z)==1:return float(z[0])
    scale=max(1.,float(np.max(np.abs(zdx)))); zz=zdx/scale
    if kind=='mean': v=np.average(z,weights=1/(1+np.abs(zz)))
    elif kind.startswith('pow'):
        pw=float(kind[3:]); v=np.average(z,weights=1/(1+np.abs(zdx))**pw)
    elif kind=='median': v=np.median(z)
    elif kind=='bracket':
        left=np.flatnonzero(zdx<0); right=np.flatnonzero(zdx>0)
        if len(left) and len(right):
            il=left[np.argmin(np.abs(zdx[left]))]; ir=right[np.argmin(np.abs(zdx[right]))]; dl=-zdx[il]; dr=zdx[ir]; v=(z[il]*dr+z[ir]*dl)/(dl+dr)
        else:v=z[np.argmin(np.abs(zdx))]
    else:
        deg=int(kind[4:]) if kind.startswith('poly') else 1
        try:
            # Small ridge stabilizes high-order fits on near-duplicate dates.
            X=np.vstack([zz**j for j in range(deg+1)]).T; w=1/(1+2*np.abs(zz)); A=X.T@(w[:,None]*X); A[1:,1:]+=1e-3*np.eye(deg); b=X.T@(w*z); v=np.linalg.solve(A,b)[0]
        except Exception:v=np.average(z,weights=1/(1+np.abs(zz)))
    if kind not in ('bracket','median') and len(z)>=4:
        lo,hi=np.quantile(z,[.05,.95]); v=np.clip(v,lo-.04,hi+.04)
    return float(v)

def predict(fold,kind='linear',k=8,posterior='weighted',bin_days=30,date_weight=1.0):
    d=_prepare(fold); syn=fold.is_synthetic_gap.astype(bool).to_numpy(); known=np.isfinite(d.primary_ndvi.to_numpy(float)); y=d.primary_ndvi.to_numpy(float); x=d._ord.to_numpy(float); src=d._src.to_numpy(object); maps=_fit_source_maps(d,known,bin_days=bin_days); aoi,crop,glob,date=_mode_posteriors(d,known); out=[]
    for _,idx in d.groupby(['anon_polygon_id','_year'],sort=False).groups.items():
      ii=np.asarray(idx,dtype=int); kk=ii[known[ii]]
      for q in ii[syn[ii]]:
        p=_query_posterior(d,int(q),aoi,crop,glob,date,date_weight=date_weight); vals=[]
        for s,w in zip(SOURCES,p):
          yy=[]; xx=[]
          for j in kk:
            a,b=maps.get((s,str(src[j]),int(d._doy.iat[q]//bin_days)),maps.get((s,str(src[j]),'g'),(0.,1.))); yy.append(a+b*y[j]); xx.append(x[j]-x[q])
          v=cand(0,np.array(xx),np.array(yy),kind,k)
          if np.isfinite(v):vals.append((v,w,s))
        if vals:
          if posterior=='mode': out.append((q,max(vals,key=lambda t:(t[1],-SOURCES.index(t[2])))[0]))
          elif posterior=='top2':
            vv=sorted(vals,key=lambda t:t[1],reverse=True)[:2]; out.append((q,float(np.average([t[0] for t in vv],weights=[t[1] for t in vv]))))
          else:out.append((q,float(np.average([t[0] for t in vals],weights=[t[1] for t in vals]))) )
    pred=np.full(len(d),np.nan); pred[[q for q,v in out]]=[v for q,v in out]
    # nearest fallback
    for q in np.flatnonzero(syn&~np.isfinite(pred)):
      same=np.flatnonzero(known&(d.anon_polygon_id.to_numpy()==d.anon_polygon_id.iat[q])); pred[q]=y[same[np.argmin(np.abs(x[same]-x[q]))]] if len(same) else np.nanmedian(y[known])
    return pred[syn]

def collect(fold, kinds, ks, posterior='weighted', bin_days=30, date_weight=1.0):
    """One expensive pass over a fold; return predictions for all grids."""
    d=_prepare(fold); syn=fold.is_synthetic_gap.astype(bool).to_numpy(); known=np.isfinite(d.primary_ndvi.to_numpy(float)); y=d.primary_ndvi.to_numpy(float); x=d._ord.to_numpy(float); src=d._src.to_numpy(object); maps=_fit_source_maps(d,known,bin_days=bin_days); aoi,crop,glob,date=_mode_posteriors(d,known)
    names=[f'{kind}_k{k}' for kind in kinds for k in ks]; pred={n:np.full(len(d),np.nan) for n in names}
    for _,idx in d.groupby(['anon_polygon_id','_year'],sort=False).groups.items():
      ii=np.asarray(idx,dtype=int); kk=ii[known[ii]]
      for q in ii[syn[ii]]:
        p=_query_posterior(d,int(q),aoi,crop,glob,date,date_weight=date_weight); arr={name:[] for name in names}
        for s,w in zip(SOURCES,p):
          yy=[]; xx=[]
          for j in kk:
            a,b=maps.get((s,str(src[j]),int(d._doy.iat[q]//bin_days)),maps.get((s,str(src[j]),'g'),(0.,1.))); yy.append(a+b*y[j]); xx.append(x[j]-x[q])
          for kind in kinds:
            for k in ks:
              v=cand(0,np.array(xx),np.array(yy),kind,k)
              if np.isfinite(v):arr[f'{kind}_k{k}'].append((v,w,s))
        for name,vals in arr.items():
          if not vals: continue
          if posterior=='mode': v=max(vals,key=lambda t:(t[1],-SOURCES.index(t[2])))[0]
          elif posterior=='top2':
            vv=sorted(vals,key=lambda t:t[1],reverse=True)[:2]; v=float(np.average([t[0] for t in vv],weights=[t[1] for t in vv]))
          else:v=float(np.average([t[0] for t in vals],weights=[t[1] for t in vals]))
          pred[name][q]=v
    for name,pp in pred.items():
      for q in np.flatnonzero(syn&~np.isfinite(pp)):
        same=np.flatnonzero(known&(d.anon_polygon_id.to_numpy()==d.anon_polygon_id.iat[q])); pp[q]=y[same[np.argmin(np.abs(x[same]-x[q]))]] if len(same) else np.nanmedian(y[known])
      pred[name]=pp[syn]
    return pred, fold.loc[syn,'_truth'].to_numpy(float)

def main():
 b=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); tr=pd.read_csv(b/'train_dataset.csv',low_memory=False,parse_dates=['date']); pr=pd.read_csv(b/'private_features.csv',low_memory=False,parse_dates=['date']);
 methods={}
 for kind in ['mean','pow0.5','pow1','pow2','pow3','median','bracket','poly1','poly2','poly3']:
  for k in [2,3,4,6,8,12,16,24]: methods[f'{kind}_k{k}']=(kind,k)
 # one pass per fold, all candidates in memory
 rec={name:[] for name in methods}
 kinds=['mean','pow0.5','pow1','pow2','pow3','median','bracket','poly1','poly2','poly3']; ks=[2,3,4,6,8,12,16,24]
 for yr in [2019,2020,2021,2022,2023,2024]:
  f,t=make_fold(tr,pr,yr); pp,yy=collect(f,kinds,ks)
  for name,ph in pp.items(): rec[name].extend((ph-yy).tolist())
 rows=[]
 for name,e in rec.items():
  e=np.asarray(e); rows.append((name,len(e),float(np.sqrt(np.mean(e*e))),float(np.mean(np.abs(e)))))
 out=pd.DataFrame(rows,columns=['method','n','rmse','mae']).sort_values('rmse'); out.to_csv(ROOT/'research'/'local_grid_agent2.csv',index=False); print(out.head(30).to_string(index=False))
if __name__=='__main__':main()

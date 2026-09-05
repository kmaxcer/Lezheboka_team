import pandas as pd, numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent
def rmse(q,p): return np.sqrt(np.mean((p-q.truth.to_numpy())**2))
def load(fn,seed):
 q=pd.read_csv(R/fn); q=q[q.mask_seed.eq(seed)].copy() if 'mask_seed' in q else q
 if 'span' not in q:
  pr=pd.read_csv(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904/private_features.csv',parse_dates=['date']); pr['_gap']=pr.is_synthetic_gap.fillna(False).astype(bool)
  hk=set(zip(q.anon_polygon_id,pd.to_datetime(q.date))); pr['_hold']=[(i,d) in hk for i,d in zip(pr.anon_polygon_id,pr.date)]; known=pr.primary_ndvi.notna()&~pr._gap&~pr._hold
  spans=[]
  for i,d in zip(q.anon_polygon_id,pd.to_datetime(q.date)):
   x=pr[(pr.anon_polygon_id==i)&(pr.date.dt.year==d.year)&known]; ds=(x.date-d).dt.days; a=ds[ds<0].abs().min() if (ds<0).any() else np.nan; b=ds[ds>0].min() if (ds>0).any() else np.nan; spans.append(np.nanmin([a,b]))
  q['span']=spans
 return q
def add_scheme(q,name,ws):
 # peer correction is zero where peer unavailable
 d=np.nan_to_num(q.n16_c60_r125_k2.to_numpy()-q.ext40.to_numpy())
 if name=='year': w=np.where(q.year.to_numpy()<2025,ws[0],ws[1])
 elif name=='span':
  s=q.span.to_numpy(); w=np.select([s<=2,s<=5],[ws[0],ws[1]],default=ws[2])
 elif name=='cohort': w=np.where(q.cohort.eq('new'),ws[0],ws[1])
 elif name=='yearspan':
  s=q.span.to_numpy(); hist=q.year.to_numpy()<2025
  w=np.select([hist&(s<=2),hist&(s<=5),hist,~hist&(s<=2),~hist&(s<=5)],[ws[0],ws[1],ws[2],ws[3],ws[4]],default=ws[5])
 elif name=='cohortyear':
  new=q.cohort.eq('new').to_numpy(); hist=q.year.to_numpy()<2025
  w=np.select([new&hist,new&~hist,~new&hist,~new&~hist],[ws[0],ws[1],ws[2],ws[3]],default=0)
 return q.ext40.to_numpy()+w*d
def main():
 files=[('private_cohort_blend_holdout_predictions.csv',70404),('meta_residual_v2_independent_predictions.csv',0),('meta_residual_v2_independent_predictions.csv',1)]
 qs=[load(*x) for x in files]
 # Holdout-independent pooled score for fixed simple schemes.
 schemes=[]
 for w in [.05,.08,.10,.12,.15]: schemes.append(('year',(w,0)))
 for w in [.05,.08,.10,.12,.15]: schemes.append(('span',(w,w,0)))
 for w in [.05,.08,.10]: schemes.append(('cohort',(w,0)))
 for s,ws in schemes:
  vals=[rmse(q,add_scheme(q,s,ws)) for q in qs]; pooled=rmse(pd.concat(qs,ignore_index=True),np.concatenate([add_scheme(q,s,ws) for q in qs]))
  print(s,ws,'masks',*[round(v,6) for v in vals],'pooled',round(pooled,6))
 # A few pre-registered year/span routes (history gets peer; 2025 stays ext40).
 for ws in [(0.1,0.1,0,0,0,0),(0.12,0.1,0,0,0,0),(0.15,0.12,0,0,0,0),(0.1,0.15,0.05,0,0,0)]:
  vals=[rmse(q,add_scheme(q,'yearspan',ws)) for q in qs]; pooled=rmse(pd.concat(qs,ignore_index=True),np.concatenate([add_scheme(q,'yearspan',ws) for q in qs]))
  print('yearspan',ws,'masks',*[round(v,6) for v in vals],'pooled',round(pooled,6))
if __name__=='__main__': main()

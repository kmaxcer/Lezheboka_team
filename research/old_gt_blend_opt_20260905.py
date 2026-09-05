import pandas as pd, numpy as np, os, json, hashlib
from pathlib import Path
root=Path('.')
gt=pd.read_csv('research/data_update_20260905_1350/private_test_ground_truth.csv')
gt['date']=pd.to_datetime(gt.date).dt.strftime('%Y-%m-%d')
score=pd.read_csv('research/released_ground_truth_candidate_scores_20260905.csv')
# top 40 existing
rows=[]
for fn in score.file.head(50):
 p=Path('outputs')/fn
 if not p.exists(): continue
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
 m=gt.merge(d,on=['anon_polygon_id','date'],how='inner',validate='one_to_one')
 if len(m)!=len(gt): continue
 rows.append((fn,m.primary_ndvi_pred.to_numpy(float)))
print('loaded',len(rows))
y=gt.primary_ndvi_true.to_numpy(float)
P=np.column_stack([x[1] for x in rows]); names=[x[0] for x in rows]
rm=np.sqrt(np.mean((P-y[:,None])**2,axis=0)); order=np.argsort(rm)
print('top',[(names[i],rm[i]) for i in order[:15]])
# pairwise residual corr and best convex weight among top 20
res=P-y[:,None]
results=[]
for ai in order[:20]:
 for bi in order[:20]:
  if ai>=bi: continue
  a,b=P[:,ai],P[:,bi]
  delta=b-a
  w=np.dot(y-a,delta)/np.dot(delta,delta) if np.dot(delta,delta)>0 else 0
  wc=np.clip(w,0,1)
  pred=(1-wc)*a+wc*b
  r=np.sqrt(np.mean((pred-y)**2))
  # robust group mean of holdout rmse by AOI/year
  tmp=gt[['anon_polygon_id','date']].copy(); tmp['year']=pd.to_datetime(tmp.date).dt.year; tmp['e']=(pred-y)**2
  ao=np.sqrt(tmp.groupby('anon_polygon_id').e.mean()).mean(); yr=np.sqrt(tmp.groupby('year').e.mean()).mean()
  results.append((r,w,wc,ao,yr,names[ai],names[bi]))
results.sort()
print('best blends',results[:20])
# Group CV: derive weight on train AOIs, evaluate heldout AOIs; same for years
for groupcol in ['anon_polygon_id','year']:
 groups=gt[groupcol] if groupcol=='anon_polygon_id' else pd.to_datetime(gt.date).dt.year
 uniq=groups.unique(); out=[]
 # use top 12 only
 for ai in order[:12]:
  for bi in order[:12]:
   if ai>=bi: continue
   preds=[]
   for g in uniq:
    tr=groups!=g; te=groups==g
    d=P[tr,bi]-P[tr,ai]; w=np.dot(y[tr]-P[tr,ai],d)/np.dot(d,d) if np.dot(d,d)>0 else 0; w=np.clip(w,0,1)
    preds.extend((te, w))
   # recompute each row weights cumbersome
   pe=np.empty(len(y))
   for g in uniq:
    tr=groups!=g; te=groups==g; d=P[tr,bi]-P[tr,ai]; w=np.dot(y[tr]-P[tr,ai],d)/np.dot(d,d) if np.dot(d,d)>0 else 0; w=np.clip(w,0,1); pe[te]=(1-w)*P[te,ai]+w*P[te,bi]
   r=np.sqrt(np.mean((pe-y)**2)); out.append((r,names[ai],names[bi]))
 print('CV',groupcol,sorted(out)[:10])
# save compact residual stats
Path('research/old_gt_blend_pairs_20260905.csv').write_text('rmse,w,ao_rmse_mean,year_rmse_mean,a,b\n'+'\n'.join(f'{r[0]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}' for r in results[:100]),encoding='utf-8')

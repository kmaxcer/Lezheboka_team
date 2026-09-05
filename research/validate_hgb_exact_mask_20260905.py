from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'research'; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(R))
from eval_paired_w16_r4_mean_a025_20260905 import route_rows, local_base, rmse, SEEDS
tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False)
pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
rr=route_rows(); frames=[]
for s in [0,1,2,70404]:
 q,f,m=local_base(s,tr,pr,rr)
 # pair08 exact from existing pair artifacts
 if s<70404:
  old=pd.read_csv(R/'paired_aoi_v2_predictions.csv',parse_dates=['date'],low_memory=False); old=old[old.partition.eq('random_'+str(s))]
  pp=old[['anon_polygon_id','date','n12_c40_r100_k2']]
 else:
  old=pd.read_csv(R/'paired_aoi_v2_seed70404_private_only_predictions_20260905.csv',parse_dates=['date'],low_memory=False)
  pp=old[old.config.eq('n12_c40_r100_k2')][['anon_polygon_id','date','peer']].rename(columns={'peer':'n12_c40_r100_k2'})
 q=q.merge(pp,on=['anon_polygon_id','date'],how='left',validate='one_to_one')
 hp=pd.read_csv(R/f'hgb_robust_seed{s}_predictions.csv',parse_dates=['date'],low_memory=False)
 # exact mask key matching only
 hp=hp[['anon_polygon_id','date','sq_clip']].rename(columns={'sq_clip':'hgb_sq_clip'})
 before=len(q); q=q.merge(hp,on=['anon_polygon_id','date'],how='left',validate='one_to_one')
 print('seed',s,'q',before,'hgb matched',q.hgb_sq_clip.notna().sum(),'pair',q.n12_c40_r100_k2.notna().sum(), 'hgb unique',len(hp))
 q['seed']=s; frames.append(q)
allq=pd.concat(frames,ignore_index=True)
# pair08 base25 was not stored in q? local_base adds base25 only after merge pp. yes.
rows=[]
for comp in ['base25','pair08']:
 for w in [0,.01,.02,.03,.05,.08,.10,.15,.20]:
  for scope,g in [('pooled',allq)]+[(f'seed{s}',allq[allq.seed.eq(s)]) for s in [0,1,2,70404]]:
   b=g.base25.to_numpy(float); pair=g.n12_c40_r100_k2.to_numpy(float); h=g.hgb_sq_clip.to_numpy(float); y=g.truth.to_numpy(float)
   if comp=='base25': d=b
   else: d=np.where(np.isfinite(pair),.92*b+.08*pair,b)
   ok=np.isfinite(h)&np.isfinite(y); p=np.where(ok,(1-w)*d+w*h,d); rows.append({'comp':comp,'weight':w,'scope':scope,'n':len(g),'hgb_n':int(ok.sum()),'coverage':float(ok.mean()),'rmse':rmse(y,p),'base_rmse':rmse(y,d),'delta':rmse(y,p)-rmse(y,d)})
# slices for best pooled
met=pd.DataFrame(rows); print('\nBEST POOLED'); print(met[met.scope.eq('pooled')].sort_values('rmse').head(20).to_string(index=False)); print('\nBY SEED pair08 w=.15'); print(met[(met.comp=='pair08')&(met.weight==.15)&met.scope.str.startswith('seed')].to_string(index=False))
met.to_csv(R/'hgb_exact_mask_validation_20260905_metrics.csv',index=False,float_format='%.10f')
# save per-row predictions sidecar exact matches
allq.to_csv(R/'hgb_exact_mask_validation_20260905_rows.csv',index=False,float_format='%.10f')
report=['# Exact-mask HGB blend validation (2026-09-05)','', 'Each robust HGB prediction is joined only to the identical `(anon_polygon_id,date)` holdout mask that generated it (seed 0, 1, 2, 70404). No cross-seed assignment.', '', 'Pair08 base = source-route + local w16/r4/mean alpha=.25 with n12_c40_r100_k2 paired overlay weight .08.', '', met[met.scope.eq('pooled')].sort_values('rmse').head(20).to_string(index=False), '', 'Per-seed pair08 + HGB at w=.15:', '', met[(met.comp=="pair08")&(met.weight==.15)&met.scope.str.startswith("seed")].to_string(index=False)]
(ROOT/'reports'/'hgb_exact_mask_validation_20260905.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

from pathlib import Path
import sys,time,json
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
ROOT=Path('.'); R=ROOT/'research'; D=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(ROOT/'_archive_inspect'/'agropulse_max_score'/'src')); from agropulse.pipeline import build_features
sys.path.insert(0,str(R)); import meta_residual_v2 as mr; from feature_hgb_v2 import _clear, extra_features
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
mr.TRAIN_IDS=set(pd.read_csv(D/'train_dataset.csv',usecols=[ID])[ID].astype(str))
def mat(d,obs,m):
 fr=_clear(d,np.asarray(m,bool)); mm=np.asarray(m,bool)
 bx=extra_features(fr,obs,mm)
 return bx.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
def feat_ctx(d,mask,qmask):
 c=mr.context_features(d,np.asarray(mask,bool),np.asarray(qmask,bool));
 keep=[x for x in c.columns if x not in [ID,DATE]]
 return c[keep].reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
def fill(X):
 X=X.astype(float); med=np.nanmedian(X,axis=0); med=np.where(np.isfinite(med),med,0.0); ii=np.where(~np.isfinite(X)); X[ii]=np.take(med,ii[1]); return X

def main():
 tr=pd.read_csv(D/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/'private_features.csv',parse_dates=[DATE],low_memory=False)
 gt=pd.read_csv(R/'data_update_20260905_1350/private_test_ground_truth.csv',parse_dates=[DATE]).rename(columns={'primary_ndvi_true':TARGET})
 pr[GAP]=pr[GAP].fillna(False).astype(bool); hidden=pr[GAP].to_numpy(bool); pr.loc[hidden,TARGET]=np.nan
 for f,orig in [(tr,'train'),(pr,'private')]:
  f[GAP]=False; f['_origin']=orig; f['date']=pd.to_datetime(f.date); f['year']=f.year.fillna(f.date.dt.year).astype(int); f['doy']=f.doy.fillna(f.date.dt.dayofyear).astype(int)
 d=pd.concat([tr,pr],ignore_index=True,sort=False); d['_truth']=pd.to_numeric(d[TARGET],errors='coerce'); known=d[TARGET].notna().to_numpy(bool); hm=np.r_[np.zeros(len(tr),bool),hidden]; ids=d[ID].astype(str).to_numpy(); yrs=d.date.dt.year.to_numpy(int)
 qkeys=pr.loc[hidden,[ID,DATE]]; truth=qkeys.merge(gt[[ID,DATE,TARGET]],on=[ID,DATE],validate='one_to_one')[TARGET].to_numpy(float)
 blocks=[]; yblocks=[]; cblocks=[]; baseblocks=[]; groups=[]
 for rep in range(3):
  rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(d),bool); tab=pd.DataFrame({'id':ids,'year':yrs})
  for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
   ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
  comb=hm|pm; obs=d[TARGET].where(~comb); print('block',rep,int(pm.sum()),flush=True)
  x=mat(d,obs,comb); Xq=x.loc[pm].reset_index(drop=True); Xall=x
  m=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=8,random_state=42); m.fit(fill(Xall.loc[known & ~pm].to_numpy(float)),d.loc[known & ~pm,TARGET].to_numpy(float)); bp=np.clip(m.predict(fill(Xq.to_numpy(float))),-.2,1.1)
  ctx=feat_ctx(d,comb,pm); blocks.append(ctx); cblocks.append(ctx); baseblocks.append(bp); yblocks.append(d.loc[pm,'_truth'].to_numpy(float)); groups.extend(ids[pm])
 Xc=fill(pd.concat(cblocks,ignore_index=True).to_numpy(float)); y=np.concatenate(yblocks); b=np.concatenate(baseblocks); groups=np.array(groups)
 # group-safe OOF meta evaluation on pseudo blocks by AOI: residual ridge; no query labels used
 for alpha in [10,30,100]:
  pred=np.zeros(len(y))
  for g in np.unique(groups):
   ti=groups!=g; te=~ti; mm=Ridge(alpha=alpha); mm.fit(Xc[ti],(y-b)[ti]); pred[te]=np.clip(b[te]+np.clip(mm.predict(Xc[te]),-.02,.02),-.2,1.1)
  print('pseudo ridge',alpha,'base',np.sqrt(np.mean((b-y)**2)),'meta',np.sqrt(np.mean((pred-y)**2)),flush=True)
 # build actual query context and HGB base trained on all visible context
 obs=d[TARGET].where(~hm); x=mat(d,obs,hm); m=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=8,random_state=42); m.fit(fill(x.loc[known].to_numpy(float)),d.loc[known,TARGET].to_numpy(float)); baseq=np.clip(m.predict(fill(x.loc[hm].to_numpy(float))),-.2,1.1)
 ctxq=feat_ctx(d,hm,hm); Xq=fill(ctxq.to_numpy(float));
 rec=[]
 for alpha in [10,30,100]:
  mm=Ridge(alpha=alpha); mm.fit(Xc,(y-b)); corr=np.clip(mm.predict(Xq),-.02,.02); final=np.clip(baseq+corr,-.2,1.1); rm=float(np.sqrt(np.mean((final-truth)**2))); rb=float(np.sqrt(np.mean((baseq-truth)**2))); rec.append({'alpha':alpha,'base_hgb_rmse':rb,'meta_rmse':rm,'delta':rm-rb,'coverage':len(final),'mean_corr':float(np.mean(corr))}); print('query',rec[-1],flush=True)
 # apply same correction to strongest existing robust candidate for GT audit, as a diagnostic only
 cand=pd.read_csv(ROOT/'outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv',parse_dates=[DATE]); cand=cand.merge(qkeys,on=[ID,DATE],how='right',validate='one_to_one'); cb=cand.primary_ndvi_pred.to_numpy(float)
 for alpha in [10,30,100]:
  mm=Ridge(alpha=alpha); mm.fit(Xc,(y-b)); corr=np.clip(mm.predict(Xq),-.02,.02); final=np.clip(cb+corr,-.2,1.2); rm=float(np.sqrt(np.mean((final-truth)**2))); print('robust diag',alpha,rm,'delta',rm-np.sqrt(np.mean((cb-truth)**2)))
 out={'script':'killer_temporal_meta_20260905.py','pseudo_rows':len(y),'query_rows':len(truth),'results':rec,'no_existing_overwrite':True}
 (R/'killer_temporal_meta_20260905_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');
 (R/'killer_temporal_meta_20260905_report.md').write_text('# Temporal/context residual probe\n\n'+json.dumps(out,indent=2)+'\n\nNo submission candidate was overwritten. Query labels were used only for retrospective released-GT scoring.\n',encoding='utf-8')
if __name__=='__main__': main()

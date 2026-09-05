"""Materialize leakage-safe HGB baseline for released new test features.
Uses old train + visible old private and released old-gap labels as training context;
new test synthetic-gap labels are never read. Writes only outputs/test_20260905_1350.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'research'
sys.path.insert(0,str(ROOT/'_archive_inspect'/'agropulse_max_score'/'src'))
from agropulse.pipeline import build_features
sys.path.insert(0,str(R)); from feature_hgb_v2 import _clear, extra_features
TARGET='primary_ndvi'; ID='anon_polygon_id'; DATE='date'; GAP='is_synthetic_gap'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(d,obs,mask):
 fr=_clear(d,mask); bx=build_features(fr,obs,pd.Series(np.asarray(mask,bool))); ex=extra_features(fr,obs,np.asarray(mask,bool)); return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True); ap.add_argument('--private',required=True); ap.add_argument('--test',required=True); ap.add_argument('--old_truth',required=True); ap.add_argument('--outdir',required=True); a=ap.parse_args(); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True); t0=time.time()
 tr=pd.read_csv(a.train,parse_dates=[DATE],low_memory=False); pr=pd.read_csv(a.private,parse_dates=[DATE],low_memory=False); te=pd.read_csv(a.test,parse_dates=[DATE],low_memory=False); gt=pd.read_csv(a.old_truth,parse_dates=[DATE],low_memory=False)
 # released old labels are explicitly permitted training data
 if 'primary_ndvi_true' in gt.columns: gt=gt.rename(columns={'primary_ndvi_true':TARGET})
 oldgap=pr[GAP].fillna(False).astype(bool) if GAP in pr else pd.Series(False,index=pr.index)
 pr=pr.copy().merge(gt[[ID,DATE,TARGET]].rename(columns={TARGET:'_released_truth'}),on=[ID,DATE],how='left',validate='one_to_one'); pr[TARGET]=pr[TARGET].where(pr[TARGET].notna(),pr['_released_truth']); pr=pr.drop(columns=['_released_truth'])
 for f in (tr,pr,te): f[GAP]=f.get(GAP,False); f[GAP]=f[GAP].fillna(False).astype(bool); f['_origin']='train' if f is tr else ('private' if f is pr else 'test'); f['date']=pd.to_datetime(f.date); f['year']=f['year'].fillna(f.date.dt.year).astype(int); f['doy']=f['doy'].fillna(f.date.dt.dayofyear).astype(int)
 d=pd.concat([tr,pr,te],ignore_index=True,sort=False); d['_truth']=pd.to_numeric(d[TARGET],errors='coerce'); hidden=(d['_origin'].eq('test') & d[GAP]).to_numpy(bool); qi=np.flatnonzero(hidden); known=d[TARGET].notna().to_numpy(bool)&~hidden
 # train pseudo masked rows from all known contexts; no test gap labels enter
 blocks=[]; ys=[]; years=d.date.dt.year.to_numpy(int); ids=d[ID].astype(str).to_numpy()
 for rep in range(3):
  rng=np.random.default_rng(20260905+rep); pm=np.zeros(len(d),bool)
  tab=pd.DataFrame({'id':ids,'year':years});
  for _,ix0 in tab.loc[known].groupby(['id','year'],sort=False).groups.items():
   ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.18*len(ix)))); pm[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
  comb=hidden|pm; obs=d[TARGET].where(~comb); print('features block',rep+1,'masked',int(pm.sum()),flush=True); x=matrix(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True))
 vf=_clear(d,hidden); obs=vf[TARGET].where(~hidden); print('features query',len(qi),flush=True); qx=matrix(d,obs,hidden).loc[hidden].reset_index(drop=True); X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).astype(float)
 keys=d.loc[hidden,[ID,DATE]].copy().reset_index(drop=True); outputs=[]
 for kind,spec in [('regular',dict(learning_rate=.03,max_iter=350,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.)),('wide',dict(learning_rate=.03,max_iter=350,max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=8.))]:
  m=HistGradientBoostingRegressor(loss='squared_error',random_state=42,**spec); print('fit',kind,X.shape,flush=True); m.fit(X,y); pred=np.clip(m.predict(qx),-.2,1.1); z=keys.copy(); z['primary_ndvi_pred']=pred; z[DATE]=z[DATE].dt.strftime('%Y-%m-%d'); z=z[[ID,DATE,'primary_ndvi_pred']]; path=outdir/f'model_newtest_extended_hgb_{kind}_20260905.csv'; z.to_csv(path,index=False,float_format='%.9f'); chk=pd.read_csv(path); ok=(len(chk)==len(qi) and list(chk.columns)==[ID,DATE,'primary_ndvi_pred'] and chk[[ID,DATE]].drop_duplicates().shape[0]==len(chk) and np.isfinite(chk.primary_ndvi_pred).all()); meta={'candidate':str(path),'formula':'HistGradientBoosting on old train + old private with released old-gap labels, 3x18% AOI-year pseudo-mask OOF blocks; feature_hgb_v2 build_features + extra_features; query=new test is_synthetic_gap rows','rows':len(z),'finite':bool(ok),'unique_keys':int(chk[[ID,DATE]].drop_duplicates().shape[0]),'feature_count':int(X.shape[1]),'train_rows':int(len(X)),'released_old_labels':int(pr['_origin'].eq('private').sum() and pr[TARGET].notna().sum()),'new_test_gap_rows':int(len(qi)),'test_sha256':sha(a.test),'train_sha256':sha(a.train),'private_sha256':sha(a.private),'old_truth_sha256':sha(a.old_truth),'candidate_sha256':sha(path),'no_upload':True,'production_baseline_overwritten':False,'seconds':round(time.time()-t0,1)}; (path.with_suffix('.json')).write_text(json.dumps(meta,indent=2),encoding='utf8'); outputs.append(meta)
 report={'dataset':'data_update_20260905_1350','test_rows':int(len(te)),'test_gap_rows':int(len(qi)),'test_aoi':int(te[ID].nunique()),'date_min':str(te.date.min().date()),'date_max':str(te.date.max().date()),'label_provenance':'new test gap labels never read; old private released ground truth joined only as training target','outputs':outputs}; (outdir/'REPORT.md').write_text('# New test HGB baseline\n\n'+json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

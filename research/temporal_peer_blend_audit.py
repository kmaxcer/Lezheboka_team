"""Audit a small adaptive temporal/spatial peer correction over ext40.

Uses existing leakage-safe holdout predictions (seeds 0/1 and 70404) and
the exact peer component used for the private submission.  Also emits
candidate files that differ from ext40 only by w*(peer-ext40).
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; R=ROOT/'research'
def rmse(y,p):
    q=np.isfinite(y)&np.isfinite(p); return float(np.sqrt(np.mean((p[q]-y[q])**2)))
def cohorts(g):
    return {'all':g,'history':g[g.year<2025],'2025':g[g.year==2025],
      'new2025':g[(g.year==2025)&(g.cohort=='new')],
      'shared2025':g[(g.year==2025)&(g.cohort=='shared')]}
def main():
    rows=[]; qs=[]
    for fn,seed in [('private_cohort_blend_holdout_predictions.csv',70404),('meta_residual_v2_independent_predictions.csv',0),('meta_residual_v2_independent_predictions.csv',1)]:
      q=pd.read_csv(R/fn); q=q[q.mask_seed.eq(seed)] if 'mask_seed' in q else q; qs.append(q)
      for name,g in cohorts(q).items():
       y=g.truth.to_numpy(float); b=g.ext40.to_numpy(float); peer=g['n16_c60_r125_k2'].to_numpy(float)
       for w in [0,.03,.05,.07,.08,.10,.12,.15]:
        p=b+w*np.nan_to_num(peer-b,nan=0.0)
        rows.append({'audit':fn,'seed':seed,'cohort':name,'n':len(g),'peer_coverage':float(np.isfinite(peer).mean()),'w':w,'rmse':rmse(y,p),'delta':rmse(y,p)-rmse(y,b)})
    d=pd.DataFrame(rows); d.to_csv(R/'temporal_peer_blend_audit.csv',index=False,float_format='%.10f')
    print(d.to_string(index=False))
    # concise pooled summary across the three masks
    pool=[]
    for w in [0,.10,.12]:
      vals=[]
      for q in qs:
       dd=np.nan_to_num(q.n16_c60_r125_k2.to_numpy()-q.ext40.to_numpy()); ww=np.where(q.year.to_numpy()<2025,w,0); vals.extend(q.ext40.to_numpy()+ww*dd-q.truth.to_numpy())
      pool.append({'route':'ext40' if w==0 else f'history_peer_{w:.2f}','pooled_rmse':float(np.sqrt(np.mean(np.asarray(vals)**2))),'pooled_gapscore':float(30*max(0,1-np.sqrt(np.mean(np.asarray(vals)**2))/.10))})
    pd.DataFrame(pool).to_csv(R/'temporal_peer_blend_pooled_summary.csv',index=False,float_format='%.10f'); print(pd.DataFrame(pool).to_string(index=False))
    # Actual private candidates based on ext40 and exact peer apply rows.
    c=pd.read_csv(OUT/'model_dani_lag40_peer10_extwide40_v3_30_submission.csv')
    peer=pd.read_csv(R/'ensemble_cv_v2_peer_apply_rows.csv',usecols=['anon_polygon_id','date','peer'])
    m=c.merge(peer,on=['anon_polygon_id','date'],how='left',validate='one_to_one')
    for w in [.07,.08,.10]:
      z=c.copy(); z['primary_ndvi_pred']=np.clip(m.primary_ndvi_pred.to_numpy(float)+w*(m.peer.to_numpy(float)-m.primary_ndvi_pred.to_numpy(float)),0,1)
      out=OUT/f'model_dani_extwide40_v3_30_peerblend{int(w*100):02d}_submission.csv'; z.to_csv(out,index=False,float_format='%.8f')
      print(out.name, 'rows',len(z),'peerfinite',int(np.isfinite(m.peer).sum()),'minmax',z.primary_ndvi_pred.min(),z.primary_ndvi_pred.max())
    # Cohort/year-adaptive candidate: peer correction only for history rows.
    d=m.peer.to_numpy(float)-m.primary_ndvi_pred.to_numpy(float); yrs=pd.to_datetime(m.date).dt.year.to_numpy()
    import hashlib, json
    def sha(path):
      h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
    for wh in [.10,.12]:
      z=c.copy(); z['primary_ndvi_pred']=np.clip(m.primary_ndvi_pred.to_numpy(float)+np.where(yrs<2025,wh,0)*np.nan_to_num(d),0,1)
      out=OUT/f'model_dani_extwide40_v3_30_peerblend{int(wh*100):02d}_history_submission.csv'; z.to_csv(out,index=False,float_format='%.8f')
      meta={'candidate':out.name,'formula':f'ext40 + {wh:.2f}*(peer-ext40) for year<2025; 2025 untouched','rows':len(z),'history_rows':int((yrs<2025).sum()),'year2025_rows':int((yrs==2025).sum()),'peer_rows':int(np.isfinite(m.peer).sum()),'base_sha256':sha(OUT/'model_dani_lag40_peer10_extwide40_v3_30_submission.csv'),'peer_apply_sha256':sha(R/'ensemble_cv_v2_peer_apply_rows.csv'),'candidate_sha256':sha(out),'production_baseline_overwritten':False}
      (out.with_name(out.stem+'_metadata.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(out.name,'rows',len(z),'sha',meta['candidate_sha256'])
if __name__=='__main__': main()

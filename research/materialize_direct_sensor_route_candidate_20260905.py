"""Materialise an optional direct-sensor residual variant of the route candidate.

The parent candidate is untouched.  For actual private gaps, direct sensor
summaries are computed from visible train + private rows at the same date and
crop (radius 2), then a tiny residual beta=.02 is applied only to far/no-peer
queries; near peers keep the route prediction.  This policy was selected by
four-mask LOO audit and is intentionally conservative.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');R=ROOT/'research';OUT=ROOT/'outputs'
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
from direct_spatial_sensor_eval_fast_20260905 import spatial_values,posterior,fit_affine
from source_expert_route_v2 import _masked_private,_neighbor_counts

def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()

def main():
 t0=time.time();tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
 tr[GAP]=False;pr[GAP]=pr[GAP].fillna(False).astype(bool);actual=pr[GAP].to_numpy(bool);qi=np.flatnonzero(actual)
 basepath=OUT/'model_dani_source_expert_route_v2_cohort_year_dist_submission.csv'
 if not basepath.exists(): raise FileNotFoundError(basepath)
 base=pd.read_csv(basepath,parse_dates=[DATE],low_memory=False); bm=base.set_index([ID,DATE])['primary_ndvi_pred']
 qkeys=pr.loc[actual,[ID,DATE,'crop_type']].copy().reset_index(drop=True);qkeys[DATE]=pd.to_datetime(qkeys[DATE])
 P=np.asarray([bm.get((i,d),np.nan) for i,d in qkeys[[ID,DATE]].itertuples(index=False,name=None)],float)
 if not np.isfinite(P).all(): raise RuntimeError('parent candidate key alignment failed')
 # Mask actual gaps; hold is all-false, so no hidden values enter the peer pool.
 pm,gaps=_masked_private(pr,np.zeros(len(pr),bool));
 V,C,near_direct=spatial_values(pm,tr,gaps,qi,radius=2,crop=True,method='median')
 W=posterior(pm,tr)[qi];cal=fit_affine(tr);V=V*cal[:,0][None,:]+cal[:,1][None,:];ok=np.isfinite(V);WW=W*ok;den=WW.sum(1)
 D=np.divide(np.nan_to_num(V)*WW,den[:,None],out=np.full_like(V,np.nan),where=den[:,None]>0).sum(1)
 # Route near distance is computed by the same observable source-count routine
 # used by the parent candidate, not by hidden labels.
 _,_,near_route=_neighbor_counts(pm,gaps,qkeys); beta=np.where(np.isfinite(near_route)&(near_route<=2),0.,.02)
 pred=np.where(np.isfinite(D),P+beta*(D-P),P);pred=np.clip(pred,-.2,1.1)
 out=qkeys[[ID,DATE]].copy();out['primary_ndvi_pred']=pred;out[DATE]=pd.to_datetime(out[DATE]).dt.strftime('%Y-%m-%d')
 path=OUT/'model_dani_source_expert_route_v2_cohort_year_dist_directsensor_r2b002_submission.csv';out.to_csv(path,index=False,float_format='%.9f')
 if len(out)!=3112 or out[[ID,DATE]].drop_duplicates().shape[0]!=len(out) or not np.isfinite(pred).all(): raise RuntimeError('candidate contract failed')
 counts={'rows':len(out),'direct_coverage':int(np.isfinite(D).sum()),'near_beta0':int((beta==0).sum()),'far_beta002':int((beta>.0).sum()),'near_direct_dist_le2':int((np.isfinite(near_direct)&(near_direct<=2)).sum()),'near_route_dist_le2':int((np.isfinite(near_route)&(near_route<=2)).sum())}
 meta={'candidate':path.name,'parent':basepath.name,'formula':'parent cohort/year/dist route prediction + beta*(direct same-date crop sensor mix - parent), beta=0 for route near_dist<=2, beta=.02 otherwise; direct values affine-calibrated per sensor and mixed by observable schedule posterior','rows':len(out),'finite':bool(np.isfinite(pred).all()),'unique_keys':int(out[[ID,DATE]].drop_duplicates().shape[0]),'sha256':sha(path),'coverage':counts,'seconds':round(time.time()-t0,1)}
 path.with_name(path.stem+'_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 (R/'direct_sensor_route_candidate_report_20260905.md').write_text('# Optional direct-sensor route candidate\n\nParent candidate was not overwritten. '+json.dumps(meta,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()

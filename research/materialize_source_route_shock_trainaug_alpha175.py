"""Materialise alpha=.175 train-augmented shock candidate (separate file)."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';O=ROOT/'outputs';D=ROOT/'_archive_inspect'/'agropulse_max_score'/'data';sys.path.insert(0,str(R))
from shock_bin_sweep_v1 import _features  # noqa: E402


def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()


def run():
 tr=pd.read_csv(D/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(D/'private_features.csv',parse_dates=['date'],low_memory=False);m=pr.is_synthetic_gap.fillna(False).astype(bool).to_numpy();combo=pd.concat([tr,pr],ignore_index=True,sort=False);combo['_truth']=pd.to_numeric(combo.primary_ndvi,errors='coerce');ft=_features(combo,np.r_[np.zeros(len(tr),bool),m],24);sm=ft.set_index(['anon_polygon_id','date'])['crop_shock'];basep=O/'model_dani_source_expert_route_v2_cohort_year_dist_submission.csv';bdf=pd.read_csv(basep,parse_dates=['date']).set_index(['anon_polygon_id','date']);keys=pr.loc[m,['anon_polygon_id','date']].copy();keys.date=pd.to_datetime(keys.date);ki=pd.MultiIndex.from_frame(keys);b=bdf.loc[ki,'primary_ndvi_pred'].to_numpy(float);s=np.asarray([sm.get(k,np.nan) for k in ki],float);p=np.clip(b+.175*np.nan_to_num(s,nan=0),-.2,1.1);out=keys.copy();out['primary_ndvi_pred']=p;path=O/'model_dani_source_expert_route_v2_cohort_year_dist_shock_trainaug_global175_submission.csv';
 if path.exists():raise RuntimeError(path)
 out.to_csv(path,index=False,float_format='%.9f');chk=pd.read_csv(path);ok=len(chk)==int(m.sum()) and chk[['anon_polygon_id','date']].drop_duplicates().shape[0]==len(chk) and np.isfinite(chk.primary_ndvi_pred).all();meta={'candidate':path.name,'formula':'base=source_route_cohort_year_dist; pred=clip(base+0.175*visible_train_augmented_24day_date_crop_shock)','rows':len(out),'finite':bool(ok),'shock_finite':int(np.isfinite(s).sum()),'base_sha256':sha(basep),'candidate_sha256':sha(path),'production_baseline_overwritten':False,'no_upload':True};path.with_name(path.stem+'_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8');(R/'source_route_shock_trainaug_alpha175_report.md').write_text('# Alpha .175 train-augmented source-route candidate\n\n'+json.dumps(meta,indent=2)+'\n',encoding='utf8');print(json.dumps(meta,indent=2))


if __name__=='__main__':run()

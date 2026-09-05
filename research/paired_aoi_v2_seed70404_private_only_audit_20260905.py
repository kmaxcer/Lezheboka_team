"""Audit paired_aoi_v2 on a private-only random-70404 mask.

The peer map sees only visible private rows (no train rows), while the base is
the stronger train-augmented r2 + local-peer(.20) route sidecar.  This isolates
whether private-only affine peers add any safe residual signal.  No output
candidate is overwritten or uploaded.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R=ROOT/'research'; REP=ROOT/'reports'
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
SEED=70404
import sys; sys.path.insert(0,str(R))
from paired_aoi_v2 import peer_predictions, _random_mask, _config_name, CONFIGS

def rmse(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);ok=np.isfinite(y)&np.isfinite(p)
 return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan

def build_base():
 """Reconstruct trainaug-r2 cohort/year route + localpeer .20 for seed 70404."""
 rows=pd.read_csv(R/'source_expert_route_v2_fixed_radius_trainaug_rows.csv',parse_dates=[DATE],low_memory=False)
 rows=rows[rows.seed.astype(int)==SEED].copy()
 probe=pd.read_csv(R/'source_schedule_route_probe_rows.csv',parse_dates=[DATE],low_memory=False)
 keys=[ID,DATE,'seed']
 rows=rows.merge(probe[keys+['sp_crop_2_n','sp_crop_8_n']],on=keys,how='left',validate='one_to_one')
 n2=rows.sp_crop_2_n.fillna(0).to_numpy(float);n8=rows.sp_crop_8_n.fillna(0).to_numpy(float)
 near=n2>0;mid=(~near)&(n8>0);a=np.where(near,.5,np.where(mid,.4,.3));yy=rows.year.to_numpy(int);co=rows.cohort.astype(str).to_numpy();a=np.where((co=='new')&(yy==2025),.6,a);a=np.where((co=='shared')&(yy==2025),.35,a)
 route=rows.baseline.to_numpy(float)+a*(rows.expert_trainaug_r2.to_numpy(float)-rows.baseline.to_numpy(float))
 lf=pd.read_csv(R/'local_peer_residual_v1_features.csv',parse_dates=[DATE],low_memory=False)
 lf=lf[lf.seed.astype(int)==SEED][keys+['r8_crop_resmean']]
 z=rows[keys+['truth','year','cohort','true_src','baseline']].copy();z['route']=route;z['near_trainaug']=near;z=z.merge(lf,on=keys,how='left',validate='one_to_one');z['base_local']=np.clip(z.route.to_numpy(float)+.20*z.r8_crop_resmean.fillna(0).to_numpy(float),-.2,1.1)
 return z

def main():
 private=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
 frame,mask=_random_mask(private,SEED)
 print('query count',int(mask.sum()),'frame',frame.shape,flush=True)
 # `_random_mask` leaves `_truth` as sidecar and clears every dynamic field.
 peer,pairs=peer_predictions(frame,mask,partition='private_random70404')
 keys=frame.loc[mask,[ID,DATE]].copy().reset_index(drop=True);keys[DATE]=pd.to_datetime(keys[DATE]);keys['_truth']=frame.loc[mask,'_truth'].to_numpy(float)
 peer=peer.drop(columns=['_row'],errors='ignore');q=keys.merge(peer,on=[ID,DATE],how='left',validate='one_to_one')
 base=build_base();q=q.merge(base,on=[ID,DATE],how='left',validate='one_to_one');
 if q.base_local.isna().any(): raise RuntimeError('base alignment failed')
 # Requested configurations plus the full grid for quick ranking.
 requested=['n12_c40_r100_k2','n8_c40_r125_k2','n16_c60_r125_k2']
 configs=requested+[_config_name(*c) for c in CONFIGS if _config_name(*c) not in requested]
 rows=[];pred_rows=[]
 y=q['_truth'].to_numpy(float);b=q.base_local.to_numpy(float)
 for cfg in configs:
  if cfg not in q:continue
  d=q[cfg].to_numpy(float);ok=np.isfinite(d); 
  for w in [0,.01,.02,.03,.05,.08,.10,.12,.15,.20,.25,.30]:
   p=np.where(ok,(1-w)*b+w*d,b);rows.append({'seed':SEED,'config':cfg,'weight':w,'n':len(y),'peer_n':int(ok.sum()),'coverage':float(ok.mean()),'rmse':rmse(y,p),'base_rmse':rmse(y,b),'delta_rmse':rmse(y,p)-rmse(y,b)})
  # Save requested per-query rows only to keep artifact compact.
  if cfg in requested:
   z=q[[ID,DATE,'_truth','base_local','route','year','cohort','true_src','near_trainaug',cfg]].copy();z=z.rename(columns={'_truth':'truth',cfg:'peer'});z['config']=cfg;z['seed']=SEED;pred_rows.append(z)
 m=pd.DataFrame(rows).sort_values('rmse');m.to_csv(R/'paired_aoi_v2_seed70404_private_only_metrics_20260905.csv',index=False,float_format='%.10f')
 pp=pd.concat(pred_rows,ignore_index=True) if pred_rows else pd.DataFrame();pp.to_csv(R/'paired_aoi_v2_seed70404_private_only_predictions_20260905.csv',index=False,float_format='%.9f')
 pairs.to_csv(R/'paired_aoi_v2_seed70404_private_only_pairs_20260905.csv',index=False)
 print(m.head(40).to_string(index=False),flush=True)
 best=m.iloc[0].to_dict() if len(m) else {}; REP.mkdir(exist_ok=True)
 (REP/'paired_aoi_v2_seed70404_private_only_report_20260905.md').write_text('# paired_aoi_v2 private-only random seed 70404\n\nPeer affine maps fit only on visible private rows; no train rows enter peer selection. Base is trainaug-r2 cohort/year route plus local peer correction alpha=.20.\n\nBest rows:\n\n'+m.head(40).to_string(index=False)+'\n\nArtifacts: `research/paired_aoi_v2_seed70404_private_only_metrics_20260905.csv`, `research/paired_aoi_v2_seed70404_private_only_predictions_20260905.csv`, `research/paired_aoi_v2_seed70404_private_only_pairs_20260905.csv`.\n',encoding='utf-8')
 print('best',json.dumps(best),flush=True)

if __name__=='__main__':main()

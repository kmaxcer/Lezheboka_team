from pathlib import Path
import sys, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'research'; DATA = Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
ID, DATE, GAP = 'anon_polygon_id','date','is_synthetic_gap'; SEEDS=(0,1,2,70404)
sys.path.insert(0,str(R))
from teammate_sweep_postcorr import _mask_private
from local_peer_residual_sweep_v1 import feature
from paired_aoi_v2 import peer_predictions

def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan

def route_rows():
    rows=pd.read_csv(R/'source_expert_route_v2_fixed_radius_trainaug_rows.csv',parse_dates=[DATE],low_memory=False)
    probe=pd.read_csv(R/'source_schedule_route_probe_rows.csv',parse_dates=[DATE],low_memory=False)
    keys=[ID,DATE,'seed']; rows=rows.merge(probe[keys+['sp_crop_2_n','sp_crop_8_n']],on=keys,how='left',validate='one_to_one')
    n2=rows.sp_crop_2_n.fillna(0).to_numpy(float); n8=rows.sp_crop_8_n.fillna(0).to_numpy(float)
    near=n2>0; mid=(~near)&(n8>0); a=np.where(near,.5,np.where(mid,.4,.3)); yy=rows.year.to_numpy(int); co=rows.cohort.astype(str).to_numpy()
    a=np.where((co=='new')&(yy==2025),.6,a); a=np.where((co=='shared')&(yy==2025),.35,a)
    rows['route']=rows.baseline.to_numpy(float)+a*(rows.expert_trainaug_r2.to_numpy(float)-rows.baseline.to_numpy(float)); rows['near_trainaug']=near
    return rows

def local_base(seed, tr, pr, rr):
    f,m=_mask_private(pr,int(seed)); tr0=tr.copy(); tr0['_truth']=pd.to_numeric(tr0.primary_ndvi,errors='coerce'); tr0['_hidden']=False
    f=f.copy(); f['_truth']=pd.to_numeric(f.primary_ndvi,errors='coerce'); f['_hidden']=m
    combo=pd.concat([tr0,f],ignore_index=True,sort=False); known=combo.primary_ndvi.notna().to_numpy(bool)&~combo._hidden.to_numpy(bool)
    qidx=np.flatnonzero(np.r_[np.zeros(len(tr),bool),m]); x=feature(combo,known,qidx,width=16,radius=4,source_level=False,agg='mean')
    keys=f.loc[m,[ID,DATE]].copy().reset_index(drop=True); keys[DATE]=pd.to_datetime(keys[DATE]); keys['local_feature']=x
    q=rr[rr.seed.astype(int)==int(seed)].copy().merge(keys,on=[ID,DATE],how='inner',validate='one_to_one')
    q['base25']=np.clip(q.route.to_numpy(float)+.25*q.local_feature.fillna(0).to_numpy(float),-.2,1.1)
    q['truth']=q.truth.astype(float); q['seed']=int(seed)
    # Static nearest-AOI distance (diagnostic only; paired/local features
    # remain leakage-safe).  This sidecar was computed without held labels.
    nf=R/'local_peer_residual_v1_features.csv'
    if nf.exists():
        nd=pd.read_csv(nf,usecols=[ID,DATE,'seed','near_dist'],parse_dates=[DATE],low_memory=False)
        nd=nd[nd.seed.astype(int).eq(int(seed))].drop_duplicates([ID,DATE])
        q=q.merge(nd[[ID,DATE,'near_dist']],on=[ID,DATE],how='left',validate='one_to_one')
    else: q['near_dist']=np.nan
    return q, f, m

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False); rr=route_rows(); frames=[]
    for s in SEEDS:
        q,f,m=local_base(s,tr,pr,rr); print('seed',s,'base rows',len(q),'local cov',np.isfinite(q.local_feature).mean(),flush=True)
        if s<70404:
            old=pd.read_csv(R/'paired_aoi_v2_predictions.csv',parse_dates=[DATE],low_memory=False); old=old[old.partition.eq('random_'+str(s))].copy(); cfgs=['n12_c40_r100_k2','n12_c80_r100_k2','n16_c60_r125_k2']; pp=old[[ID,DATE]+cfgs]
        else:
            old=pd.read_csv(R/'paired_aoi_v2_seed70404_private_only_predictions_20260905.csv',parse_dates=[DATE],low_memory=False); wide=old.pivot(index=[ID,DATE],columns='config',values='peer').reset_index(); cfgs=['n12_c40_r100_k2','n12_c80_r100_k2','n16_c60_r125_k2'];
            # Compute missing c80 map on this mask once; all values are fit only on visible private rows.
            extra_path=R/'paired_aoi_v2_seed70404_private_only_predictions_extra_20260905.csv'
            if extra_path.exists():
                wide=pd.read_csv(extra_path,parse_dates=[DATE],low_memory=False)
            else:
                frame,m2=_mask_private(pr,s); peer,_=peer_predictions(frame,m2,partition='private_random70404_extra'); ex=peer.drop(columns=['_row'],errors='ignore')[[ID,DATE,'n12_c80_r100_k2']]; wide=wide.merge(ex,on=[ID,DATE],how='outer',validate='one_to_one'); wide.to_csv(extra_path,index=False,float_format='%.9f')
            pp=wide
        q=q.merge(pp,on=[ID,DATE],how='left',validate='one_to_one'); frames.append(q)
    allq=pd.concat(frames,ignore_index=True,sort=False); weights=(.03,.05,.08,.10); rows=[]
    for cfg in ['n12_c40_r100_k2','n12_c80_r100_k2','n16_c60_r125_k2']:
        for w in weights:
            for scope,g in [('pooled',allq)]+[(f'seed{s}',allq[allq.seed.eq(s)]) for s in SEEDS]:
                b=g.base25.to_numpy(float); d=g[cfg].to_numpy(float); y=g.truth.to_numpy(float); ok=np.isfinite(d); p=np.where(ok,(1-w)*b+w*d,b)
                rows.append({'scope':scope,'seed':-1 if scope=='pooled' else int(scope[4:]),'config':cfg,'weight':w,'n':len(g),'peer_n':int(ok.sum()),'coverage':float(ok.mean()),'rmse':rmse(y,p),'base_rmse':rmse(y,b),'delta_rmse':rmse(y,p)-rmse(y,b)})
    met=pd.DataFrame(rows); met.to_csv(R/'paired_aoi_w16_r4_mean_a025_metrics_20260905.csv',index=False,float_format='%.10f')
    # Best per config/weight slice diagnostics (year/cohort/source and distance quartile).
    best=met[met.scope.eq('pooled')].sort_values('rmse').groupby('config',as_index=False).first(); sl=[]
    for _,z in best.iterrows():
        cfg,w=z.config,float(z.weight)
        for dim in ['year','cohort','true_src']:
            for val,g in allq.groupby(dim,dropna=False):
                b=g.base25.to_numpy(float); d=g[cfg].to_numpy(float); y=g.truth.to_numpy(float); p=np.where(np.isfinite(d),(1-w)*b+w*d,b); sl.append({'config':cfg,'weight':w,'slice':dim,'value':str(val),'n':len(g),'peer_cov':float(np.isfinite(d).mean()),'base_rmse':rmse(y,b),'rmse':rmse(y,p),'delta_rmse':rmse(y,p)-rmse(y,b)})
    allq['dist_bin']=pd.qcut(allq['near_dist'].rank(method='first'),4,labels=['q1','q2','q3','q4']) if allq['near_dist'].notna().any() else 'all'
    for _,z in best.iterrows():
        cfg,w=z.config,float(z.weight)
        for val,g in allq.groupby('dist_bin',dropna=False):
            b=g.base25.to_numpy(float); d=g[cfg].to_numpy(float); y=g.truth.to_numpy(float); p=np.where(np.isfinite(d),(1-w)*b+w*d,b); sl.append({'config':cfg,'weight':w,'slice':'distance_q','value':str(val),'n':len(g),'peer_cov':float(np.isfinite(d).mean()),'base_rmse':rmse(y,b),'rmse':rmse(y,p),'delta_rmse':rmse(y,p)-rmse(y,b)})
    pd.DataFrame(sl).to_csv(R/'paired_aoi_w16_r4_mean_a025_slices_20260905.csv',index=False,float_format='%.10f')
    report=['# Paired-AOI overlay on w16/r4/mean alpha=.25 base','', 'Base = source trainaug-r2 cohort/year route + 0.25 * leakage-safe visible 16-day same-crop peer residual mean (numeric AOI radius 4). Paired correction blends base with affine peer prediction: (1-w)*base + w*peer; missing peers fall back to base.', '', 'Pooled and per-seed metrics:', '', met.sort_values(['scope','rmse']).to_string(index=False), '', 'Pooled winners by config:', '', best.to_string(index=False), '', 'Slice diagnostics saved to `research/paired_aoi_w16_r4_mean_a025_slices_20260905.csv`.', '', 'No submission uploaded; no existing candidate overwritten.']
    (ROOT/'reports'/'paired_aoi_w16_r4_mean_a025_20260905.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(met[met.scope.eq('pooled')].sort_values('rmse').to_string(index=False)); print('best',best.to_string(index=False))

if __name__=='__main__': main()

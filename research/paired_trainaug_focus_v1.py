"""Focused train-augmented paired-AOI predictor for one peer configuration.

The full paired_aoi_v2 grid is expensive because it evaluates 324 configs.
This helper computes only n16/corr.60/max-cv-RMSE.125/top-2, using train plus
visible private rows, for quick overlay audits.  Hidden labels are never used
in pair fitting.
"""
from __future__ import annotations
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904');sys.path.insert(0,str(R))
from paired_aoi_v2 import _pair_model  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402

ID,DATE='anon_polygon_id','date'; SEEDS=(0,1,2,70404)

def focused(frame,qmask,config=(16,.60,.125,2)):
    d=frame.copy();d[DATE]=pd.to_datetime(d[DATE]);d['_yr']=d[DATE].dt.year.astype(int);qmask=np.asarray(qmask,bool);known=d.primary_ndvi.notna().to_numpy(bool)&~qmask
    q=d.loc[qmask,[ID,DATE]].copy().reset_index().rename(columns={'index':'_row'});q['_yr']=q[DATE].dt.year.astype(int);q['_qpos']=np.arange(len(q));out=np.full(len(q),np.nan)
    min_n,min_corr,max_rmse,topk=config
    for year,qy in q.groupby('_yr',sort=False):
        obs=d.loc[known & d['_yr'].eq(int(year)).to_numpy(),[DATE,ID,'primary_ndvi']]
        if obs.empty:continue
        pivot=obs.pivot_table(index=DATE,columns=ID,values='primary_ndvi',aggfunc='first'); ids=list(pivot.columns)
        # Fit each target AOI's peer maps once, then apply to its query dates.
        for target_id,qt in qy.groupby(ID,sort=False):
            if target_id not in pivot.columns:continue
            yt=pivot[target_id]; models={}
            for peer_id in ids:
                if peer_id==target_id:continue
                xy=pd.concat([pivot[peer_id],yt],axis=1,keys=['x','y']).dropna(); m=_pair_model(xy.x.to_numpy(float),xy.y.to_numpy(float),min_fit=min_n)
                if m is None or m.n<min_n or m.corr<min_corr or m.cv_rmse>max_rmse:continue
                models[str(peer_id)]=m
            if not models:continue
            for _,qr in qt.iterrows():
                dt=qr[DATE]
                if dt not in pivot.index:continue
                row=pivot.loc[dt];cand=[]
                for pid,m in models.items():
                    if pid not in row.index:continue
                    v=row[pid]
                    if np.isfinite(v):cand.append((m.cv_rmse,-m.n,m.intercept+m.slope*float(v),m))
                if not cand:continue
                cand.sort(key=lambda z:(z[0],z[1]));cand=cand[:topk]; vals=np.array([z[2] for z in cand]);w=np.array([min(1.,z[3].n/24.)/(z[3].cv_rmse**2+.0009) for z in cand]);out[int(qr['_qpos'])]=np.clip(np.average(vals,weights=w),-.2,1.1)
    ans=q[[ID,DATE]].copy();ans['peer_pred']=out;return ans

def run():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False);parts=[]
    for seed in SEEDS:
        f,m=_mask_private(pr,int(seed)); combo=pd.concat([tr,f],ignore_index=True,sort=False);qmask=np.r_[np.zeros(len(tr),bool),m];print('seed',seed,'start',flush=True);a=focused(combo,qmask);a['seed']=int(seed);parts.append(a);print('seed',seed,'coverage',a.peer_pred.notna().mean(),flush=True)
    out=pd.concat(parts,ignore_index=True);out.to_csv(R/'paired_trainaug_focus_v1_predictions.csv',index=False,float_format='%.9f');print(out.groupby('seed').peer_pred.apply(lambda x:x.notna().mean()).to_string())
if __name__=='__main__':run()

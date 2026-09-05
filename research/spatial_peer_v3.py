"""Quick ablation of stronger same-date peer-AOI transfers."""
from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
import argparse
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'research')); sys.path.insert(0,str(ROOT/'src'))
from paired_aoi_v2 import _pair_model, _random_mask
from validate import make_fold
warnings.filterwarnings('ignore')
ID='anon_polygon_id'; DATE='date'; Y='primary_ndvi'; GAP='is_synthetic_gap'

def peer_custom(d, mask, mode='base', top=4, min_n=12, min_corr=.4, max_rmse=.125):
    d=d.copy().reset_index(drop=True); d[DATE]=pd.to_datetime(d[DATE]); mask=np.asarray(mask,bool)
    known=d[Y].notna().to_numpy() & ~mask
    if GAP in d: known &= ~d[GAP].fillna(False).astype(bool).to_numpy()
    d['_year']=d[DATE].dt.year; d['_crop']=d.get('crop_type',pd.Series('',index=d.index)).fillna('').astype(str)
    d['_src']=np.select([d.get('s2_ndvi',pd.Series(np.nan,index=d.index)).notna(),d.get('landsat_ndvi',pd.Series(np.nan,index=d.index)).notna(),d.get('modis_ndvi',pd.Series(np.nan,index=d.index)).notna()],['S2','L8','MOD'],default='NONE')
    q=d.loc[mask,[ID,DATE]].copy(); out=np.full(len(q),np.nan); q['_pos']=np.arange(len(q))
    for yr,qy in q.assign(_year=q[DATE].dt.year).groupby('_year',sort=False):
        yy=d['_year'].eq(yr).to_numpy(); obs=d.loc[known&yy,[DATE,ID,Y]]
        if obs.empty: continue
        piv=obs.pivot_table(index=DATE,columns=ID,values=Y,aggfunc='first')
        meta=d[[ID,'_crop']].drop_duplicates(ID).set_index(ID)['_crop'].to_dict()
        smeta=d[[ID,'_src']].drop_duplicates(ID).set_index(ID)['_src'].to_dict()
        for tid, qt in qy.groupby(ID,sort=False):
            if tid not in piv: continue
            ty=piv[tid]; models=[]
            for pid in piv.columns:
                if pid==tid: continue
                if mode=='crop' and meta.get(pid,'')!=meta.get(tid,''): continue
                if mode=='crop_source' and (meta.get(pid,'')!=meta.get(tid,'') or smeta.get(pid,'')!=smeta.get(tid,'')): continue
                xy=pd.concat([piv[pid],ty],axis=1,keys=['x','y']).dropna()
                m=_pair_model(xy.x.to_numpy(float),xy.y.to_numpy(float),min_fit=min_n)
                if m is not None and m.n>=min_n and m.corr>=min_corr and m.cv_rmse<=max_rmse: models.append((pid,m))
            if not models: continue
            if mode=='corr_rank': models.sort(key=lambda z:(-z[1].corr,z[1].cv_rmse,-z[1].n))
            else: models.sort(key=lambda z:(z[1].cv_rmse,-z[1].n,-z[1].corr))
            models=models[:max(top,1)]
            for _, row in qt.iterrows():
                pos=int(row['_pos']); dt=row[DATE]
                if dt not in piv.index: continue
                avail=[]
                for pid,m in models:
                    x=piv.at[dt,pid] if pid in piv.columns else np.nan
                    if np.isfinite(x): avail.append((float(x),m))
                if not avail: continue
                vals=np.array([m.intercept+m.slope*x for x,m in avail])
                if mode=='median': pred=np.median(vals)
                elif mode=='mean': pred=np.mean(vals)
                else:
                    w=np.array([min(1.,m.n/24.)/(m.cv_rmse*m.cv_rmse+.0009) for x,m in avail]); pred=np.average(vals,weights=w)
                out[pos]=np.clip(pred,-.2,1.1)
    return out
def score(p,y): return float(np.sqrt(np.nanmean((p-y)**2)))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--random-only',action='store_true'); ap.add_argument('--exact-only',action='store_true'); args=ap.parse_args()
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
    specs=[('base',4),('crop',4),('crop_source',4),('corr_rank',4),('median',4),('mean',4)]; rows=[]
    for yr in ([] if args.random_only else [2019,2020,2021,2022,2023,2024]):
        f,t=make_fold(tr,pr,yr); mask=f[GAP].fillna(False).to_numpy(); y=t.to_numpy(float)
        for mode,top in specs:
            p=peer_custom(f,mask,mode,top); ok=np.isfinite(p); rows.append(dict(protocol='exact',year=yr,mode=mode,top=top,n=int(ok.sum()),coverage=float(ok.mean()),rmse=score(p[ok],y[ok])))
        print('exact',yr,flush=True)
    if args.exact_only: args.random_only=True
    for seed in ([] if args.exact_only else [0,1,2]):
        f,mask=_random_mask(pr,seed); truth=f['_truth'].to_numpy(float); y=truth[mask]
        for mode,top in specs:
            p=peer_custom(f,mask,mode,top); ok=np.isfinite(p); rows.append(dict(protocol='random',year=0,seed=seed,mode=mode,top=top,n=int(ok.sum()),coverage=float(ok.mean()),rmse=score(p[ok],y[ok])))
        qmask=mask & f[DATE].dt.year.eq(2025).to_numpy(); y=truth[qmask]
        for mode,top in specs:
            p=peer_custom(f,qmask,mode,top); ok=np.isfinite(p); rows.append(dict(protocol='random2025',year=2025,seed=seed,mode=mode,top=top,n=int(ok.sum()),coverage=float(ok.mean()),rmse=score(p[ok],y[ok])))
        print('random',seed,flush=True)
    o=pd.DataFrame(rows); o.to_csv(ROOT/'research/spatial_peer_v3_results.csv',index=False)
    agg=o.groupby(['protocol','mode','top']).apply(lambda g: pd.Series({'n':g.n.sum(),'cov':np.average(g.coverage,weights=g.n),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n))})).sort_values('rmse'); print(agg.to_string())
if __name__=='__main__': main()

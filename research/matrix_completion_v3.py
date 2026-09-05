import numpy as np, pandas as pd
from pathlib import Path

ROOT=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
TR=ROOT/'train_dataset.csv'; PR=ROOT/'private_features.csv'

def complete_group(g, hold_idx=None, rank=8, iters=12):
    # AOI matrix years x DOY; only target values are used
    years=np.sort(g.year.unique()); yi={y:i for i,y in enumerate(years)}
    a=np.full((len(years),366),np.nan,float)
    for r in g.itertuples():
        if pd.notna(r.primary_ndvi) and (hold_idx is None or r.Index not in hold_idx): a[yi[r.year],int(r.doy)]=r.primary_ndvi
    obs=np.isfinite(a); fill=np.where(obs,a,np.nanmean(a,axis=0,keepdims=True))
    fill=np.where(np.isfinite(fill),fill,np.nanmean(a))
    for _ in range(iters):
        u,s,v=np.linalg.svd(fill,full_matrices=False); rr=min(rank,len(s)); z=(u[:,:rr]*s[:rr])@v[:rr]
        fill=np.where(obs,a,z)
    return {(r.year,int(r.doy)):fill[yi[r.year],int(r.doy)] for r in g.itertuples()}

def run(seed, frac=.15):
    d=pd.read_csv(TR); d['date']=pd.to_datetime(d.date); d=d[d.primary_ndvi.notna()].copy()
    rng=np.random.default_rng(seed); hold=set(rng.choice(d.index,int(len(d)*frac),replace=False)); pred={}
    for _,g in d.groupby('anon_polygon_id',sort=False): pred.update(complete_group(g,hold,rank=8))
    y=[]; yh=[]
    for r in d.loc[list(hold)].itertuples():
        p=pred.get((r.year,int(r.doy)),np.nan); 
        if np.isfinite(p): y.append(r.primary_ndvi); yh.append(p)
    return len(y),float(np.sqrt(np.mean((np.array(y)-np.array(yh))**2)))

def private_pred(rank=8):
    tr=pd.read_csv(TR); pr=pd.read_csv(PR); tr['date']=pd.to_datetime(tr.date); pr['date']=pd.to_datetime(pr.date)
    # combine known targets across train and private; infer each private missing row
    known=tr[tr.primary_ndvi.notna()][['anon_polygon_id','date','year','doy','primary_ndvi']].copy()
    # for shared AOIs private known values supersede/augment train
    pk=pr[pr.primary_ndvi.notna()][['anon_polygon_id','date','year','doy','primary_ndvi']]
    allk=pd.concat([known,pk],ignore_index=True).drop_duplicates(['anon_polygon_id','date'],keep='last')
    out=pr[pr.is_synthetic_gap.astype(bool)][['anon_polygon_id','date','year','doy']].copy(); vals=[]
    for a,gq in out.groupby('anon_polygon_id'):
        g=allk[allk.anon_polygon_id==a]
        if len(g):
            pp=complete_group(g,rank=rank)
            vals.extend([pp.get((r.year,int(r.doy)) if pd.notna(r.doy) else (r.year,-1),np.nan) for r in gq.itertuples()])
        else: vals.extend([np.nan]*len(gq))
    out['primary_ndvi_pred']=vals
    return out

if __name__=='__main__':
    rows=[]
    for s in range(3):
        n,e=run(s); print('seed',s,'n',n,'rmse',e); rows.append((s,n,e))
    pd.DataFrame(rows,columns=['seed','n','rmse']).to_csv('research/matrix_completion_v3_results.csv',index=False)
    pp=private_pred(8); pp.to_csv('research/matrix_completion_v3_private.csv',index=False)
    print('private',len(pp),'nan',pp.primary_ndvi_pred.isna().sum(),pp.primary_ndvi_pred.describe().to_dict())

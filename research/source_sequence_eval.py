"""Fast leakage-safe acquisition-source sequence predictor.

The predictor combines same-AOI/DOY source counts across years, same-year
nearest acquisition source, and global date/DOY schedule.  It is only an
observable route diagnostic; it never reads target values of query rows.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src'))
from validate import make_fold
from overnight_source_eval import _mask_private_like, _source_labels

SRC=np.array(['s2','landsat','modis'])
SENS=['s2_ndvi','landsat_ndvi','modis_ndvi']

def _mode(tab, keys, q):
    """Return smoothed count posterior reindexed to q."""
    z=tab.groupby(keys+['_src'], observed=True).size().unstack('_src',fill_value=0)
    z=z.reindex(columns=SRC,fill_value=0).astype(float)
    qi=pd.MultiIndex.from_frame(q[keys]) if len(keys)>1 else pd.Index(q[keys[0]])
    a=z.reindex(qi).fillna(0).to_numpy(float).copy()
    a += 0.20
    return a/a.sum(1,keepdims=True)

def _nearest_same_year(d, obs):
    """Source of nearest visible same-AOI/year date, vectorized via merge_asof."""
    base=d[['anon_polygon_id','date','_year','_src']].copy()
    o=base.loc[obs & base._src.isin(SRC)].copy().sort_values('date')
    q=base.loc[~obs].copy(); q['_qi']=q.index
    if q.empty or o.empty:return np.full((len(d),3),1/3.),np.full(len(d),np.nan)
    # nearest before/after, then choose smaller absolute distance
    left=q[['anon_polygon_id','date','_year','_qi']].sort_values('date')
    right=o[['anon_polygon_id','date','_year','_src']].sort_values('date')
    l=pd.merge_asof(left,right,on='date',by=['anon_polygon_id','_year'],direction='backward',tolerance=pd.Timedelta(days=90),suffixes=('','_b'))
    r=pd.merge_asof(left,right,on='date',by=['anon_polygon_id','_year'],direction='forward',tolerance=pd.Timedelta(days=90),suffixes=('','_f'))
    # merge_asof retains query rows in original order of sorted left
    qix=l['_qi'].to_numpy(); db=(l['date']-l['date']).to_numpy() if False else None
    # We need right dates; merge with renamed source date.
    rb=right.rename(columns={'date':'date_b','_src':'src_b'}).sort_values('date_b')
    rf=right.rename(columns={'date':'date_f','_src':'src_f'}).sort_values('date_f')
    lb=pd.merge_asof(left.rename(columns={'date':'date_q'}),rb,left_on='date_q',right_on='date_b',by=['anon_polygon_id','_year'],direction='backward',tolerance=pd.Timedelta(days=90))
    lf=pd.merge_asof(left.rename(columns={'date':'date_q'}),rf,left_on='date_q',right_on='date_f',by=['anon_polygon_id','_year'],direction='forward',tolerance=pd.Timedelta(days=90))
    bdist=(lb.date_q-lb.date_b).dt.days.abs().to_numpy(float); fdist=(lf.date_f-lf.date_q).dt.days.abs().to_numpy(float)
    usef=(~np.isfinite(bdist)) | (np.isfinite(fdist)&(fdist<bdist))
    ss=np.where(usef,lf.src_f.to_numpy(object),lb.src_b.to_numpy(object))
    dist=np.where(usef,fdist,bdist)
    out=np.full((len(d),3),1/3.); route=np.full(len(d),np.nan)
    for qi,s,dd in zip(left['_qi'].to_numpy(),ss,dist):
        if s in SRC and np.isfinite(dd):
            j=np.flatnonzero(SRC==s)[0]; out[qi]=0.08;out[qi,j]=0.84;route[qi]=j
    return out,route

def posterior(frame, obs, query=None, true_src=None):
    d=frame.copy().reset_index(drop=True); d['date']=pd.to_datetime(d.date); d['_year']=d.date.dt.year.astype(int);d['_doy']=d.date.dt.dayofyear.astype(int)
    labels=_source_labels(d)
    d['_src']=np.where(obs,labels,'none')
    z=d.loc[obs & (labels!='none'),['anon_polygon_id','_year','_doy','date','_src']].copy()
    if query is None: query=~obs
    q=d.loc[np.asarray(query,bool)].copy()
    if q.empty:return pd.DataFrame()
    # Modes at four granularities, all based on visible rows only.
    p_aoi=_mode(z,['anon_polygon_id','_doy'],q)
    p_doy=_mode(z,['_doy'],q)
    p_date=_mode(z,['_year','_doy'],q)
    p_global=_mode(z,['_year'],q)
    # Neighboring global calendar dates reveal cadence even when an entire
    # hidden date has no visible peers.  Use a compact kernel over +/-32d.
    dt=z.groupby(['_year','_doy','_src'],observed=True).size().unstack('_src',fill_value=0).reindex(columns=SRC,fill_value=0).astype(float)
    sched=np.zeros((len(q),3),float)
    for off in range(-32,33):
        if off==0: continue
        keys=pd.MultiIndex.from_arrays([q['_year'].to_numpy(),q['_doy'].to_numpy()+off],names=['_year','_doy'])
        a=dt.reindex(keys).fillna(0).to_numpy(float)
        sched += a/(1.0+abs(off)/5.0)
    sched += .2; sched/=sched.sum(1,keepdims=True)
    p_near,_=_nearest_same_year(d,obs)
    pn=p_near[q.index.to_numpy()]
    # AOI/DOY is extremely stable for complete private AOIs.  For unseen or
    # sparse AOIs nearest same-year schedule gets more weight.
    na=z.groupby(['anon_polygon_id','_doy']).size()
    keys=pd.MultiIndex.from_frame(q[['anon_polygon_id','_doy']]); n=na.reindex(keys).fillna(0).to_numpy()
    w=np.clip(n/3,0,1)[:,None]
    p=(0.42*w*p_aoi + (0.16+0.10*(1-w))*p_date + 0.10*p_doy + 0.10*pn + 0.22*sched)
    p/=p.sum(1,keepdims=True)
    return pd.DataFrame({'row_index':q.index,'p_s2':p[:,0],'p_landsat':p[:,1],'p_modis':p[:,2],'route':SRC[p.argmax(1)],'true':labels[q.index]})

def run():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rows=[]
    for year in [2019,2020,2021,2022,2023,2024]:
        f,m=make_fold(tr.copy(),pr.copy(),year); orig=_source_labels(tr); midx=f.index[f.is_synthetic_gap.to_numpy(bool)]; f['_true_src']='none'; f.loc[midx,'_true_src']=orig[midx]
        # make_fold has target NaN for query and visible target elsewhere
        obs=f.primary_ndvi.notna().to_numpy(bool); pp=posterior(f,obs,f.is_synthetic_gap.to_numpy(bool))
        y=orig[midx]; pred=pp.route.to_numpy(); p=pp[['p_s2','p_landsat','p_modis']].to_numpy()
        rows += [{'protocol':'exact','part':year,'method':'sequence','n':len(y),'acc':accuracy_score(y,pred),'ll':log_loss(y,p,labels=list(SRC))}]
        print('exact',year,rows[-1]['acc'],flush=True)
    for seed in [0,1,2]:
        f,m=_mask_private_like(pr,seed); orig=_source_labels(pr); obs=f.primary_ndvi.notna().to_numpy(bool); pp=posterior(f,obs,f.is_synthetic_gap.to_numpy(bool)); y=orig[m]; p=pp[['p_s2','p_landsat','p_modis']].to_numpy()
        rows += [{'protocol':'random','part':seed,'method':'sequence','n':len(y),'acc':accuracy_score(y,pp.route),'ll':log_loss(y,p,labels=list(SRC))}]
        print('random',seed,rows[-1]['acc'],flush=True)
    o=pd.DataFrame(rows); o.to_csv(ROOT/'research/source_sequence_results.csv',index=False); print(o.to_string(index=False)); print(o.groupby('protocol').apply(lambda g:pd.Series({'n':g.n.sum(),'acc':np.average(g.acc,weights=g.n),'ll':np.average(g.ll,weights=g.n)}),include_groups=False))

if __name__=='__main__':run()

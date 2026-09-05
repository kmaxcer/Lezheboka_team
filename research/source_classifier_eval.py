"""Classify hidden acquisition source from the observable acquisition schedule."""
from __future__ import annotations
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src'));from validate import make_fold
SENS=['s2_ndvi','landsat_ndvi','modis_ndvi']

def source(d):return np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],[0,1,2],-1)

def _count_table(z, keys):
    tab=z.groupby(keys+['src']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    if len(keys)==1: idx=pd.Index(z[keys[0]],name=keys[0])
    else: idx=pd.MultiIndex.from_frame(z[keys])
    return tab,idx

def features(frame, known, labels):
    d=frame.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date);d['yr']=d.date.dt.year.astype(int);d['doy2']=d.date.dt.dayofyear.astype(int);d['src']=labels
    n=len(d); out=pd.DataFrame(index=np.arange(n));
    # Stable numeric date/AOI descriptors.
    out['doy']=d.doy2;out['yr']=d.yr;out['sin']=np.sin(2*np.pi*d.doy2/366);out['cos']=np.cos(2*np.pi*d.doy2/366);out['aoi_code']=pd.Categorical(d.anon_polygon_id).codes;out['crop_code']=pd.Categorical(d.crop_type.astype(str)).codes
    z=d.loc[known & (labels>=0),['anon_polygon_id','yr','doy2','crop_type','src']].copy()
    # source counts at exact global dates, same DOY all years, AOI+DOY, AOI+year
    specs=[('date',['yr','doy2']),('doy',['doy2']),('aoi_doy',['anon_polygon_id','doy2']),('aoi_year',['anon_polygon_id','yr'])]
    for name,keys in specs:
        tab,idx=_count_table(z,keys)
        if len(keys)==1: rowidx=pd.Index(d[keys[0]],name=keys[0])
        else: rowidx=pd.MultiIndex.from_frame(d[keys])
        arr=tab.reindex(rowidx).fillna(0).to_numpy(float).copy()
        # Leave current source out for labeled rows (query rows have -1).
        ii=np.arange(n); good=labels>=0; arr[ii[good],labels[good]]-=1
        for j,s in enumerate(['s2','ls','md']):out[f'{name}_{s}']=arr[:,j]
        out[f'{name}_n']=arr.sum(1)
    # Global date neighborhood counts/proportions; dates are daily but source
    # acquisitions recur at characteristic 5/8/16/30-day cadences.
    date_tab=z.groupby(['yr','doy2','src']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    for off in [-32,-30,-16,-14,-8,-7,-5,-3,-2,-1,0,1,2,3,5,7,8,14,16,30,32]:
        keys=pd.MultiIndex.from_arrays([d.yr,d.doy2+off],names=['yr','doy2']);arr=date_tab.reindex(keys).fillna(0).to_numpy(float).copy()
        for j,s in enumerate(['s2','ls','md']):out[f'off{off}_{s}']=arr[:,j]
        out[f'off{off}_n']=arr.sum(1)
    # Rolling source frequencies by AOI/year in a +/- window, excluding row.
    # Use day-indexed lookup and direct loops over groups (small: <=213 rows).
    for w in [3,7,15,31]:
        vals=np.zeros((n,3),float)
        for (aid,yr),ix0 in d.groupby(['anon_polygon_id','yr'],sort=False).groups.items():
            ix=np.asarray(ix0,dtype=int); xx=d.doy2.to_numpy()[ix]; ss=labels[ix]
            for j,i in enumerate(ix):
                sel=(np.abs(xx-xx[j])<=w)&(np.arange(len(ix))!=j)&(ss>=0)
                for k in range(3):vals[i,k]=np.sum(ss[sel]==k)
        for k,s in enumerate(['s2','ls','md']):out[f'aoiroll{w}_{s}']=vals[:,k]
        out[f'aoiroll{w}_n']=vals.sum(1)
    return out.astype(float)

def one(frame,q,true_src):
    lab=source(frame);known=frame.primary_ndvi.notna().to_numpy(bool)&~q
    X=features(frame,known,lab)
    train=known&(lab>=0); qi=q
    # A few models; train on all visible source labels.  Evaluate only query.
    out={}
    for name,m in [
        ('hgb',HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=31,learning_rate=.06,l2_regularization=4.,random_state=42)),
        ('rf',RandomForestClassifier(n_estimators=250,min_samples_leaf=8,max_features=.7,n_jobs=4,random_state=42,class_weight='balanced_subsample')),
    ]:
        m.fit(X.loc[train],lab[train]);p=m.predict_proba(X.loc[qi]); pred=m.classes_[p.argmax(1)];out[name]=(pred,p)
    # schedule count baselines
    p=X.loc[qi,['date_s2','date_ls','date_md']].to_numpy(float);out['date']=(p.argmax(1),p/(p.sum(1,keepdims=True)+1e-9))
    p=X.loc[qi,['doy_s2','doy_ls','doy_md']].to_numpy(float);out['doy']=(p.argmax(1),p/(p.sum(1,keepdims=True)+1e-9))
    return out,np.asarray(true_src)

def main():
    warnings.filterwarnings('ignore');tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False);rows=[]
    for yr in [2019,2020,2021,2022,2023,2024]:
        src_orig=source(tr)
        f,t=make_fold(tr.copy(),pr.copy(),yr);q=f.is_synthetic_gap.fillna(False).to_numpy(bool);outs,y=one(f,q,src_orig[q])
        for name,(pred,p) in outs.items():
            rows.append({'year':yr,'method':name,'n':len(y),'acc':accuracy_score(y,pred),'logloss':log_loss(y,p,labels=[0,1,2])})
        print('done',yr,flush=True)
    o=pd.DataFrame(rows);o.to_csv(ROOT/'research/source_classifier_results.csv',index=False);print(o.to_string(index=False))
    def _agg(g): return pd.Series({'n':g.n.sum(),'acc':np.average(g.acc,weights=g.n),'logloss':np.average(g.logloss,weights=g.n)})
    print(o.groupby('method').apply(_agg,include_groups=False).to_string())
if __name__=='__main__':main()

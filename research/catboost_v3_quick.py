from __future__ import annotations
import sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(ROOT/'src'))
from validate import make_fold
sys.path.insert(0,str(ROOT/'research'))
from feature_hgb_v3 import extra_features_v3
sys.path.insert(0,str(ROOT/'_archive_inspect'/'agropulse_max_score'/'src'))
from agropulse.pipeline import build_features

TARGET='primary_ndvi'
DYN=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','modis_ndwi','era5_temp_c','era5_precip_mm','year',TARGET,'doy','ndvi_climatology_mean','ndvi_climatology_std','ndvi_zscore','n_reference_years','status']

def clear(d, mask):
    z=d.copy().reset_index(drop=True); mask=np.asarray(mask,bool)
    for c in DYN:
        if c in z: z.loc[mask,c]=np.nan
    z['is_synthetic_gap']=mask
    z['date']=pd.to_datetime(z.date)
    z['year']=z.year.fillna(z.date.dt.year).astype(int); z['doy']=z.doy.fillna(z.date.dt.dayofyear).astype(int)
    return z

def matrix(d, obs, mask):
    fr=clear(d,mask)
    bx=build_features(fr,obs,pd.Series(np.asarray(mask,bool)))
    ex=extra_features_v3(fr,obs,np.asarray(mask,bool))
    return pd.concat([bx.reset_index(drop=True),ex.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)

def pseudo(d, excluded, seed=17, frac=.18):
    known=d[TARGET].notna().to_numpy(bool)&~np.asarray(excluded,bool); out=np.zeros(len(d),bool); rng=np.random.default_rng(seed)
    for _,ix0 in d.loc[known].groupby(['anon_polygon_id',d.date.dt.year],sort=False).groups.items():
        ix=np.asarray(ix0,int); out[rng.choice(ix,size=min(len(ix),max(1,int(round(frac*len(ix))))),replace=False)]=True
    return out

def fit_cb(x,y,q,depth=7):
    m=CatBoostRegressor(iterations=260,depth=depth,learning_rate=.045,l2_leaf_reg=16,random_strength=.8,loss_function='RMSE',verbose=False,allow_writing_files=False,thread_count=4,random_seed=42)
    m.fit(x,y); return np.clip(m.predict(q),-.2,1.1)

def main():
    t=time.time(); tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rows=[]
    # One exact fold, one random fold; enough to reject a model class quickly.
    for name,d,qm in [('exact2024',*make_fold(tr.copy(),pr.copy(),2024))]:
        pass
    fold,truth=make_fold(tr.copy(),pr.copy(),2024); qm=fold.is_synthetic_gap.fillna(False).to_numpy(bool); d=fold.copy(); d['_truth']=pd.to_numeric(d[TARGET],errors='coerce'); d.loc[qm,'_truth']=truth.to_numpy(float)
    for mode,base in ([] if os.getenv('ONLY_RANDOM') else [('exact',qm)]):
        blocks=[]; ys=[]
        for k in range(2):
            pm=pseudo(d,base,100+k); comb=base|pm; obs=clear(d,comb)[TARGET].where(~comb); x=matrix(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True))
        xall=pd.concat(blocks,ignore_index=True); yall=pd.concat(ys,ignore_index=True); qx=matrix(d,clear(d,base)[TARGET].where(~base),base).loc[base].reset_index(drop=True); y=d.loc[base,'_truth'].to_numpy(float)
        for dep in (6,7,8):
            p=fit_cb(xall,yall,qx,dep); rows.append({'protocol':mode,'depth':dep,'n':len(y),'features':xall.shape[1],'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(np.abs(p-y)))})
    # random train mask
    d=tr.copy().reset_index(drop=True); d['_truth']=d[TARGET].astype(float); d['is_synthetic_gap']=False; hold=np.zeros(len(d),bool); rng=np.random.default_rng(991)
    for _,ix0 in d.groupby(['anon_polygon_id',d.date.dt.year],sort=False).groups.items():
        ix=np.asarray(ix0,int); ix=ix[d.loc[ix,TARGET].notna().to_numpy()]
        if len(ix): hold[rng.choice(ix,size=min(len(ix),max(1,int(round(.15*len(ix))))),replace=False)]=True
    blocks=[]; ys=[]
    for k in range(2):
        pm=pseudo(d,hold,200+k); comb=hold|pm; obs=clear(d,comb)[TARGET].where(~comb); x=matrix(d,obs,comb); blocks.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True))
    xall=pd.concat(blocks,ignore_index=True); yall=pd.concat(ys,ignore_index=True); qx=matrix(d,clear(d,hold)[TARGET].where(~hold),hold).loc[hold].reset_index(drop=True); y=d.loc[hold,'_truth'].to_numpy(float)
    for dep in (7,):
        p=fit_cb(xall,yall,qx,dep); ok=np.isfinite(p)&np.isfinite(y); rows.append({'protocol':'random','depth':dep,'n':len(y),'features':xall.shape[1],'finite_pred':int(np.isfinite(p).sum()),'finite_y':int(np.isfinite(y).sum()),'rmse':float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan,'mae':float(np.mean(np.abs(p[ok]-y[ok]))) if ok.any() else np.nan})
    out=pd.DataFrame(rows); out.to_csv(ROOT/'research'/'catboost_v3_quick_results.csv',index=False); print(out.to_string(index=False)); print('elapsed',round(time.time()-t,1),flush=True)

if __name__=='__main__': main()

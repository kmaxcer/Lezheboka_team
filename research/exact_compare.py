"""Evaluate candidate imputers on train years using the real private mask DOYs.

This is the closest local proxy for the organizer gap: for each year we hide
the AOI/DOY pairs appearing in the actual private synthetic rows and clear all
dynamic columns.  It also writes row-level predictions for later blending.
"""
from __future__ import annotations
from pathlib import Path
import sys, tempfile, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src'))
from validate import make_fold
from infer import predict_private
from infer_lag import predict_private_lag
ARCH=ROOT/'_archive_inspect'/'agropulse_max_score'/'src'
sys.path.insert(0,str(ARCH))
from agropulse.pipeline import fit_final_model, build_features, FULL_FEATURES

def hgb_predict(fold:pd.DataFrame):
    # fit OOF model on this masked train frame, then score its hidden rows
    ref=fold.copy()
    ref['_origin']='train'; ref['_test_order']=np.nan
    ref['year']=ref['year'].fillna(ref.date.dt.year).astype(int)
    ref['doy']=ref['doy'].fillna(ref.date.dt.dayofyear).astype(int)
    model,_=fit_final_model(ref,seed=42)
    hidden=ref.is_synthetic_gap.astype(bool)
    observed=ref.primary_ndvi.where(~hidden)
    x=build_features(ref,observed,hidden)
    pred=np.full(len(ref),np.nan)
    pred[hidden.to_numpy()]=np.clip(model.predict(x.loc[hidden,FULL_FEATURES]),-0.2,1.1)
    return pred

def local_arrays(fold):
    hidden=fold.is_synthetic_gap.astype(bool).to_numpy()
    # predict_private returns only hidden rows in frame order
    out={}
    for name,p in {
        'base_k6':predict_private(fold,k=6,bin_days=30,date_weight=1.0),
        'base_k8':predict_private(fold,k=8,bin_days=30,date_weight=1.0),
        'base_k12':predict_private(fold,k=12,bin_days=30,date_weight=1.0),
        'lag_k12_d2':predict_private_lag(fold,k=12,degree=2,bin_days=30,date_weight=1.0),
        'lag_k16_d3':predict_private_lag(fold,k=16,degree=3,bin_days=30,date_weight=1.0),
        'lag_k24_d2':predict_private_lag(fold,k=24,degree=2,bin_days=30,date_weight=1.0),
    }.items():
        # keyed merge avoids assumptions about internal order
        q=fold.loc[hidden,['anon_polygon_id','date','_truth']].copy()
        pp=p.copy(); pp['date']=pd.to_datetime(pp.date)
        q=q.merge(pp,on=['anon_polygon_id','date'],how='left',validate='one_to_one')
        out[name]=q.primary_ndvi_pred.to_numpy(float)
    return out

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False)
    pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    rows=[]; allpred=[]
    years=[int(x) for x in (sys.argv[1].split(',') if len(sys.argv)>1 else ['2019','2020','2021','2022','2023','2024'])]
    for year in years:
        fold,truth=make_fold(tr,pr,year)
        if len(truth)==0: continue
        hidden=fold.is_synthetic_gap.astype(bool).to_numpy(); q=fold.loc[hidden].copy()
        q['_true_src']=np.select([q.s2_ndvi.notna(),q.landsat_ndvi.notna(),q.modis_ndvi.notna()],['s2','landsat','modis'],'none')
        # source is lost in fold after masking; recover from original train row
        orig=tr.set_index(['anon_polygon_id','date'])
        keys=pd.MultiIndex.from_frame(q[['anon_polygon_id','date']])
        oo=orig.reindex(keys)
        q['_true_src']=np.select([oo.s2_ndvi.notna(),oo.landsat_ndvi.notna(),oo.modis_ndvi.notna()],['s2','landsat','modis'],'none')
        preds=local_arrays(fold)
        print('year',year,'n',len(q),'local done',flush=True)
        # HGB is expensive but run once
        hp=hgb_predict(fold); preds['hgb']=hp[hidden]
        print('year',year,'hgb done',flush=True)
        y=q['_truth'].to_numpy(float)
        for name,a in preds.items():
            e=a-y; row={'year':year,'method':name,'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e)))}
            for s in ['s2','landsat','modis']:
                z=q['_true_src'].to_numpy()==s
                row['n_'+s]=int(z.sum()); row['rmse_'+s]=float(np.sqrt(np.mean(e[z]**2))) if z.any() else np.nan
            rows.append(row)
        # save all prediction rows for optimization
        qq=q[['anon_polygon_id','date','_truth','_true_src']].copy(); qq['year']=year
        for name,a in preds.items(): qq[name]=a
        allpred.append(qq)
        pd.DataFrame(rows).to_csv(ROOT/'research'/'exact_compare_results.csv',index=False)
        pd.concat(allpred,ignore_index=True).to_csv(ROOT/'research'/'exact_compare_preds.csv',index=False)
        print(pd.DataFrame(rows).tail(len(preds)).to_string(index=False),flush=True)
    out=pd.DataFrame(rows); out.to_csv(ROOT/'research'/'exact_compare_results.csv',index=False)
    if allpred:
        pp=pd.concat(allpred,ignore_index=True)
        pp.to_csv(ROOT/'research'/'exact_compare_preds.csv',index=False)
        # pooled blend grid among local/hgb, plus per-source scalar weights
        names=[c for c in ['base_k6','base_k8','base_k12','lag_k12_d2','lag_k16_d3','lag_k24_d2','hgb'] if c in pp]
        scores=[]
        y=pp._truth.to_numpy(float)
        for a in names:
            for b in names:
                if a>=b:continue
                for w in np.linspace(0,1,21):
                    z=(1-w)*pp[a].to_numpy(float)+w*pp[b].to_numpy(float); e=z-y
                    scores.append({'a':a,'b':b,'w_b':w,'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e)))})
        ss=pd.DataFrame(scores).sort_values('rmse'); ss.to_csv(ROOT/'research'/'exact_blends.csv',index=False)
        print('BEST BLENDS\n',ss.head(20).to_string(index=False))
        print('POOLED SINGLE\n',out.groupby('method').rmse.mean().sort_values())

if __name__=='__main__':main()

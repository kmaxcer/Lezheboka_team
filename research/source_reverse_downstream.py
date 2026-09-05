"""Evaluate source route from date/crop spatial neighbours with source experts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'research'))
from validate import make_fold
from overnight_source_eval import _predict_matrix,_source_labels,_mask_private_like

def route_nc(d,q,w=6):
    d=d.copy().reset_index(drop=True); q=np.asarray(q); s=np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],[0,1,2],-1)
    d['_idn']=d.anon_polygon_id.str.extract(r'(\d+)',expand=False).astype(int); dates=d.date.to_numpy(); ids=d._idn.to_numpy(); crops=d.crop_type.astype(str).to_numpy(); vis=np.flatnonzero((~q)&(s>=0)); pred=np.full(q.sum(),-1,int)
    for n,i in enumerate(np.flatnonzero(q)):
        sel=vis[(dates[vis]==dates[i])&(np.abs(ids[vis]-ids[i])<=w)]; ss=sel[crops[sel]==crops[i]]
        if len(ss)==0: ss=sel
        if len(ss): pred[n]=np.bincount(s[ss],minlength=3).argmax()
    return pred

def run():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False); rows=[]
    for yr in [2019,2020,2021,2022,2023,2024]:
        f,truth=make_fold(tr.copy(),pr.copy(),yr); q=f.is_synthetic_gap.fillna(False).to_numpy(bool); y=truth.to_numpy(float)
        pm,_=_predict_matrix(f,None,family='base',k=8,degree=1,bin_days=30,date_weight=1.0)
        route=route_nc(f,q,6); mat=pm[[f'pred_{x}' for x in ['s2','landsat','modis']]].to_numpy(float);
        # fallback to production hard/soft where neighbour route absent
        soft=pm.soft.to_numpy(float); hard=pm.hard.to_numpy(float); ext=np.array([mat[i,r] if r>=0 and np.isfinite(mat[i,r]) else hard[i] for i,r in enumerate(route)])
        for name,p in [('soft',soft),('hard',hard),('extnc6',ext)]: rows.append({'year':yr,'method':name,'n':len(y),'rmse':float(np.sqrt(np.mean((p-y)**2))), 'route_acc':float(np.mean(route==np.array([{'s2':0,'landsat':1,'modis':2}[x] for x in _source_labels(tr)[q]]))) if name=='extnc6' else np.nan})
        print(yr,rows[-1],flush=True)
    o=pd.DataFrame(rows);o.to_csv(ROOT/'research/source_reverse_downstream_results.csv',index=False);print(o.groupby('method').apply(lambda g: np.sqrt(np.average(g.rmse**2,weights=g.n)),include_groups=False))
if __name__=='__main__':run()

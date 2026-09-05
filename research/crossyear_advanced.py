"""Leakage-safe cross-year seasonal-shape matching evaluator."""
from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
from scipy.interpolate import interp1d
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src'))
from validate import make_fold
warnings.filterwarnings('ignore')

def affine(x,y):
    z=np.isfinite(x)&np.isfinite(y);x=x[z];y=y[z]
    if len(x)<4 or np.ptp(x)<1e-8:return 0.,1.
    qx=np.quantile(x,[.05,.95]);qy=np.quantile(y,[.05,.95]);z=(x>=qx[0])&(x<=qx[1])&(y>=qy[0])&(y<=qy[1])
    if z.sum()>=4:x=x[z];y=y[z]
    try:b,a=np.polyfit(x,y,1)
    except:return float(np.median(y)-np.median(x)),1.
    return (float(a),float(b)) if np.isfinite(a+b) and abs(b)<5 else (0.,1.)

def curve(g):
    g=g[g.primary_ndvi.notna()].sort_values('date')
    if len(g)==0:return None
    x=g.date.dt.dayofyear.to_numpy(float); y=g.primary_ndvi.to_numpy(float)
    u,inv=np.unique(x,return_inverse=True); yy=np.array([np.median(y[inv==j]) for j in range(len(u))])
    if len(u)==1:return lambda q:np.full(len(np.asarray(q)),yy[0])
    f=interp1d(u,yy,bounds_error=False,fill_value=(yy[0],yy[-1]))
    return lambda q:np.asarray(f(q),float)

def predict(frame,mode='weighted',top=3,min_overlap=5):
    d=frame.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date)
    qm=d.is_synthetic_gap.fillna(False).astype(bool).to_numpy();y=d.primary_ndvi.to_numpy(float);known=np.isfinite(y)&~qm
    d['_yr']=d.date.dt.year.to_numpy(int);d['_doy']=d.date.dt.dayofyear.to_numpy(int);out=np.full(len(d),np.nan)
    for (pid,yr),ix0 in d.groupby(['anon_polygon_id','_yr'],sort=False).groups.items():
        ix=np.asarray(list(ix0));cur=ix[known[ix]];qry=ix[qm[ix]]
        if not len(qry) or not len(cur):continue
        cd=d.loc[cur,'_doy'].to_numpy(float);cy=y[cur]; cs=[]
        for oy,g in d[(d.anon_polygon_id==pid)&(d._yr!=yr)].groupby('_yr',sort=False):
            fn=curve(g)
            if fn is None:continue
            hv=fn(cd);ok=np.isfinite(hv)&np.isfinite(cy)
            if ok.sum()<min_overlap:continue
            a,b=affine(hv[ok],cy[ok]);rm=float(np.sqrt(np.mean((a+b*hv[ok]-cy[ok])**2)));co=float(np.corrcoef(hv[ok],cy[ok])[0,1]) if ok.sum()>3 else 0.
            cs.append((rm,-co,fn,a,b))
        qd=d.loc[qry,'_doy'].to_numpy(float)
        if not cs:
            fn=curve(d.loc[cur]);out[qry]=fn(qd) if fn else np.nanmedian(cy);continue
        cs.sort(key=lambda z:(z[0],z[1]));use=cs[:max(1,int(top))];vals=[];ws=[]
        for rm,nco,fn,a,b in use:
            vals.append(a+b*fn(qd));ws.append(1. if mode=='mean' else 1/(.015+rm)**2 if mode=='weighted' else max(.03,-nco))
        out[qry]=np.average(np.vstack(vals),axis=0,weights=ws)
    for q in np.flatnonzero(qm&~np.isfinite(out)):
        s=np.flatnonzero(known&(d.anon_polygon_id.to_numpy()==d.anon_polygon_id.iat[q]))
        if len(s):out[q]=y[s[np.argmin(abs(d.loc[s,'_doy'].to_numpy()-d.at[q,'_doy']))]]
    return out[qm]

def mask_private(pr,seed,frac=.15,year=None):
    d=pr.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date)
    if year is not None:d=d[d.date.dt.year.eq(year)].copy().reset_index(drop=True)
    truth=d.primary_ndvi.to_numpy(float);d.is_synthetic_gap=False;rng=np.random.default_rng(seed);m=np.zeros(len(d),bool)
    for _,g in d[d.primary_ndvi.notna()].groupby(['anon_polygon_id',d.date.dt.year]):
        ii=g.index.to_numpy();m[rng.choice(ii,size=min(len(ii),max(1,int(round(frac*len(ii))))),replace=False)]=True
    dyn=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','era5_temp_c','era5_precip_mm','year','primary_ndvi','doy','ndvi_climatology_mean','ndvi_climatology_std','n_reference_years']
    for c in dyn:
        if c in d:d.loc[m,c]=np.nan
    d.loc[m,'is_synthetic_gap']=True;return d,m,truth

def stat(g):return pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n)),'mae':np.average(g.mae,weights=g.n)})
def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    cfg=[('best',1),('mean',3),('weighted',1),('weighted',3),('weighted',5),('corr',3)];rows=[]
    for yr in [2019,2020,2021,2022,2023,2024]:
        f,t=make_fold(tr,pr,yr);truth=t.to_numpy(float)
        for mode,top in cfg:
            p=predict(f,mode,top);e=p-truth;rows.append(dict(protocol='exact',year=yr,mode=mode,top=top,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
        print('exact',yr,flush=True)
    for seed in [0,1,2]:
      for yr in [None,2025]:
        f,m,ta=mask_private(pr,seed,year=yr);truth=ta[m]
        for mode,top in cfg:
            p=predict(f,mode,top);e=p-truth;rows.append(dict(protocol='random2025' if yr else 'random',year=yr or 0,seed=seed,mode=mode,top=top,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),n=len(e)))
        print('random',seed,yr,flush=True)
    o=pd.DataFrame(rows);o.to_csv(ROOT/'research/crossyear_results.csv',index=False);print(o.groupby(['protocol','mode','top']).apply(stat).sort_values('rmse').to_string())
if __name__=='__main__':main()

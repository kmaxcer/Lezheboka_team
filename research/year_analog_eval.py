"""Evaluate analogue-year transfer for masked NDVI points.

For a query in AOI/year, historical years of the same AOI are ranked by how
well their visible seasonal trajectory matches the visible current-year
trajectory.  The query value is then transferred from the nearest analogue.
This is a diagnostic and never changes production files.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R=ROOT/'research'
sys.path.insert(0, str(ROOT/'src'))

def mask_private(d,seed=70404):
    z=d.copy().reset_index(drop=True); z['date']=pd.to_datetime(z.date);z['_truth']=z.primary_ndvi.astype(float)
    gap=z.is_synthetic_gap.fillna(False).astype(bool).to_numpy();known=z.primary_ndvi.notna().to_numpy()&~gap
    rng=np.random.default_rng(seed);hold=np.zeros(len(z),bool);yr=z.date.dt.year
    for _,ix0 in z.loc[known].groupby(['anon_polygon_id',yr],sort=False).groups.items():
        ix=np.asarray(ix0,int);n=max(1,int(round(.15*len(ix))));hold[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
    # actual gaps stay excluded from visible history
    z.loc[hold,'primary_ndvi']=np.nan
    return z,hold

def interp_curve(g, qd, ycol='primary_ndvi'):
    x=g.date.dt.dayofyear.to_numpy(float);y=g[ycol].to_numpy(float);ok=np.isfinite(y)
    if ok.sum()==0:return np.full(len(qd),np.nan)
    x=x[ok];y=y[ok];o=np.argsort(x);x=x[o];y=y[o]
    # np.interp edge-clips; we separately mark distant extrapolation as nan
    p=np.interp(qd,x,y)
    return p

def analog_predictions(d,hold,alpha=.0,k=3,metric='rmse',min_overlap=3,decay=30):
    d=d.copy();d['yr']=d.date.dt.year;d['doy']=d.date.dt.dayofyear
    qidx=np.flatnonzero(hold);pred=np.full(len(d),np.nan,float)
    # cache group curves
    groups={(aid,int(yr)):g for (aid,yr),g in d.groupby(['anon_polygon_id','yr'],sort=False)}
    for (aid,yr),qg in d.loc[hold].groupby(['anon_polygon_id','yr'],sort=False):
        cur=groups[(aid,int(yr))]; curk=cur[cur.primary_ndvi.notna()]
        if len(curk)<3:continue
        hist=[(yy,g) for (a,yy),g in groups.items() if a==aid and int(yy)!=int(yr) and g.primary_ndvi.notna().sum()>=3]
        if not hist:continue
        # compare on current visible dates, with historical interpolation.
        cx=curk.doy.to_numpy(float);cy=curk.primary_ndvi.to_numpy(float)
        scores=[]
        for yy,hg in hist:
            hk=hg[hg.primary_ndvi.notna()];hx=hk.doy.to_numpy(float);hy=hk.primary_ndvi.to_numpy(float);o=np.argsort(hx);hx=hx[o];hy=hy[o]
            # only compare current points inside historical support
            ok=(cx>=hx.min())&(cx<=hx.max())
            if ok.sum()<min_overlap:continue
            hp=np.interp(cx[ok],hx,hy);err=cy[ok]-hp
            # robust score trims giant target outliers
            if metric=='mae':sc=np.mean(np.abs(err))
            else:sc=np.sqrt(np.mean(np.clip(err,-.5,.5)**2))
            scores.append((sc,yy,hx,hy))
        if not scores:continue
        scores.sort(key=lambda x:x[0]);sel=scores[:max(1,k)]
        qd=qg.doy.to_numpy(float)
        vals=[];ws=[]
        for sc,yy,hx,hy in sel:
            # Don't extrapolate beyond support; fallback to nearest current.
            v=np.interp(qd,hx,hy);v[(qd<hx.min())|(qd>hx.max())]=np.nan
            w=np.exp(-sc/max(1e-3,decay/100.))
            vals.append(v);ws.append(w)
        a=np.asarray(vals);w=np.asarray(ws)[:,None];ap=np.nansum(a*w,axis=0)/np.nansum(np.where(np.isfinite(a),w,0),axis=0)
        # Current-year linear interpolation as a stabilizing component.
        cp=np.interp(qd,cx,cy);cp[(qd<cx.min())|(qd>cx.max())]=np.nan
        if alpha:
            ap=np.where(np.isfinite(ap)&np.isfinite(cp),(1-alpha)*ap+alpha*cp,np.where(np.isfinite(ap),ap,cp))
        pred[qg.index.to_numpy()]=ap
    return pred

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    cases=[]
    # private-only holdout (the actual new/shared structure)
    z,h=mask_private(pr,70404); cases.append(('private',z,h))
    # train folds using private hidden DOYs, each year
    from validate import make_fold
    for yr in [2019,2020,2021,2022,2023,2024]:
        f,t=make_fold(tr.copy(),pr.copy(),yr); f.date=pd.to_datetime(f.date);f['_truth']=f.primary_ndvi; h=f.is_synthetic_gap.fillna(False).to_numpy(bool); f.loc[h,'primary_ndvi']=np.nan;cases.append((str(yr),f,h))
    rows=[]
    for name,d,h in cases:
      y=d.loc[h,'_truth'].to_numpy(float)
      for k in [1,2,3,5,8]:
       for a in [0,.25,.5,.75]:
        p=analog_predictions(d,h,alpha=a,k=k,decay=30);ok=np.isfinite(p[h]);e=p[h][ok]-y[ok];rows.append({'case':name,'k':k,'alpha':a,'n':len(e),'coverage':ok.mean(),'rmse':np.sqrt(np.mean(e*e)) if len(e) else np.nan,'mae':np.mean(np.abs(e)) if len(e) else np.nan})
      print(name,flush=True)
    o=pd.DataFrame(rows);o.to_csv(R/'year_analog_results.csv',index=False);print(o.sort_values(['case','rmse']).groupby('case').head(12).to_string(index=False))

if __name__=='__main__':main()

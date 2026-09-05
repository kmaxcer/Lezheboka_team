"""Screen raw per-sensor curve interpolation and source uncertainty.

The important difference from earlier sensor_curve_eval is that every raw
sensor observation is used, including secondary sensors on a row whose target
came from a higher-priority sensor.  Oracle-source scores are diagnostic only;
date-count mixtures are deployable.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, UnivariateSpline

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'src'))
from validate import make_fold  # noqa
sys.path.insert(0,str(ROOT/'research'))
from teammate_sweep_postcorr import _mask_private  # noqa

SENS=['s2_ndvi','landsat_ndvi','modis_ndvi']; ID='anon_polygon_id'; TARGET='primary_ndvi'; GAP='is_synthetic_gap'

def source(d):return np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],[0,1,2],-1)
def rmse(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);z=np.isfinite(y)&np.isfinite(p)
    return float(np.sqrt(np.mean((y[z]-p[z])**2))) if z.any() else np.nan

def curve(x,y,q,kind):
    z=np.isfinite(x)&np.isfinite(y);x=np.asarray(x[z],float);y=np.asarray(y[z],float);q=np.asarray(q,float)
    if len(x)==0:return np.full(len(q),np.nan)
    o=np.argsort(x);x=x[o];y=y[o];ux,inv=np.unique(x,return_inverse=True);y=np.array([np.median(y[inv==i]) for i in range(len(ux))]);x=ux
    if len(x)==1:return np.full(len(q),y[0])
    if kind=='linear':return np.interp(q,x,y)
    if kind=='nearest':
        pos=np.searchsorted(x,q).clip(0,len(x)-1);le=np.maximum(0,pos-1);sel=np.where(abs(q-x[le])<=abs(x[pos]-q),le,pos);return y[sel]
    if kind.startswith('smooth'):
        fac=float(kind[6:]);k=min(3,len(x)-1)
        try:return UnivariateSpline(x,y,k=k,s=fac*len(x),ext=3)(q)
        except Exception:return np.interp(q,x,y)
    if kind=='pchip':
        try:return PchipInterpolator(x,y,extrapolate=True)(q)
        except Exception:return np.interp(q,x,y)
    if kind.startswith('local'):
        bw=float(kind[5:]);out=[]
        for qi in q:
            dd=abs(x-qi);take=np.argsort(dd)[:min(len(x),12)];xx=x[take]-qi;yy=y[take];w=np.exp(-.5*(dd[take]/bw)**2)
            if len(take)<3:out.append(np.average(yy,weights=w));continue
            X=np.c_[np.ones(len(xx)),xx,xx*xx];A=X*np.sqrt(w)[:,None];b=yy*np.sqrt(w)
            try:c=np.linalg.solve(A.T@A+np.diag([1e-8,1e-4,1e-4]),A.T@b);v=c[0]
            except Exception:v=np.average(yy,weights=w)
            lo,hi=np.quantile(yy,[.05,.95]);out.append(np.clip(v,lo-.05,hi+.05))
        return np.asarray(out)
    raise ValueError(kind)

def source_probs(d, qmask):
    """Observable schedule posterior using visible rows only."""
    x=d.copy().reset_index(drop=True);x['yr']=pd.to_datetime(x.date).dt.year;x['dy']=pd.to_datetime(x.date).dt.dayofyear;lab=source(x);known=(lab>=0)&~qmask
    z=pd.DataFrame({'id':x[ID].astype(str),'yr':x.yr,'dy':x.dy,'src':lab})[known]
    tabs={}
    for name,cols in [('date',['yr','dy']),('id_dy',['id','dy']),('dy',['dy']),('id_bin',['id'])]:
        if name=='id_bin':
            zz=z.assign(bin=(z.dy//8).astype(int));cols=['id','bin']
        else:zz=z
        tabs[name]=zz.groupby(cols+['src']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    out=np.zeros((len(x),3))
    for i,r in x.iterrows():
        vals=[];weights=[]
        keys=[('date',(r.yr,r.dy),8.),('id_dy',(str(r[ID]),r.dy),3.),('dy',(r.dy,),1.),('id_bin',(str(r[ID]),int(r.dy//8)),1.)]
        for name,key,w in keys:
            tab=tabs[name];k=key if len(key)>1 else key[0]
            if k in tab.index:
                c=tab.loc[k].to_numpy(float);s=c.sum()
                if s:vals.append(c/s);weights.append(w*np.sqrt(s))
        out[i]=np.average(vals,axis=0,weights=weights) if vals else np.array([.4,.4,.2])
        out[i]=(out[i]+.01)/(out[i].sum()+.03)
    return out

def predict(d,qmask,kind,oracle_src=None):
    x=d.copy().reset_index(drop=True);x.date=pd.to_datetime(x.date);qmask=np.asarray(qmask,bool);doy=x.date.dt.dayofyear.to_numpy(float);yr=x.date.dt.year.to_numpy(int);ids=x[ID].astype(str).to_numpy()
    probs=source_probs(x,qmask);allp=np.full((len(x),3),np.nan)
    for (aid,year),ix0 in x.groupby([ids,yr],sort=False).groups.items():
        ix=np.asarray(ix0,int);q=ix[qmask[ix]]
        if not len(q):continue
        for s,col in enumerate(SENS):
            vals=pd.to_numeric(x.loc[ix,col],errors='coerce').to_numpy(float);allp[q,s]=curve(doy[ix],vals,doy[q],kind)
    pp=allp[qmask];pr=probs[qmask];good=np.isfinite(pp);pr=np.where(good,pr,0);pr=pr/(pr.sum(1,keepdims=True)+1e-12)
    mix=np.nansum(pp*pr,axis=1);mode=pp[np.arange(len(pp)),np.argmax(pr,axis=1)]
    out={'mix':mix,'mode':mode}
    if oracle_src is not None:
        out['oracle']=pp[np.arange(len(pp)),np.asarray(oracle_src,int)]
        out['p_true']=np.sum(pr*np.eye(3)[np.asarray(oracle_src,int)],axis=1)
    return out

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    scenarios=[]
    orig=source(tr)
    for y in range(2019,2025):
        f,_=make_fold(tr.copy(),pr.copy(),y);f=f.reset_index(drop=True);q=f[GAP].fillna(False).to_numpy(bool);scenarios.append((f'e{y}',f,q,f.loc[q,'_truth'].to_numpy(float),orig[q]))
    for seed in (0,1,2):
        # save original source before masking
        origp=source(pr);f,q=_mask_private(pr.copy(),seed);f=f.reset_index(drop=True);q=np.asarray(q,bool);scenarios.append((f'r{seed}',f,q,f.loc[q,'_truth'].to_numpy(float),origp[q]))
    rows=[]
    for kind in ['linear','nearest','pchip','smooth.0003','smooth.001','smooth.003','smooth.01','smooth.03','local4','local8','local12','local20','local30']:
        for sn,f,q,y,s in scenarios:
            out=predict(f,q,kind,s)
            for mode,p in out.items():
                if mode=='p_true':continue
                rows.append({'kind':kind,'scenario':sn,'family':'exact' if sn[0]=='e' else 'random','mode':mode,'n':len(y),'rmse':rmse(y,p)})
        print(kind,flush=True)
    z=pd.DataFrame(rows);z.to_csv(ROOT/'research/raw_sensor_oracle_results.csv',index=False)
    a=z.groupby(['family','kind','mode'],as_index=False).apply(lambda g:pd.Series({'n':g.n.sum(),'rmse':np.sqrt(np.average(g.rmse**2,weights=g.n))}),include_groups=False).reset_index(drop=True).sort_values(['family','rmse']);a.to_csv(ROOT/'research/raw_sensor_oracle_aggregate.csv',index=False);print(a.to_string(index=False))

if __name__=='__main__':main()

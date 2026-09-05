"""Source-specific cross-AOI latent factor audit.

The hidden target is the first available sensor in the S2/Landsat/MODIS
priority chain.  This experiment estimates each sensor's value from visible
same-date peers (pairwise affine maps + a common factor), then mixes the
three experts using only the observable acquisition schedule.  It is
deliberately independent of production files.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); R=ROOT/'research'
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'; SC=['s2_ndvi','landsat_ndvi','modis_ndvi']; SN=['s2','ls','md']
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold

def src_label(d):
    return np.select([d.s2_ndvi.notna(),d.landsat_ndvi.notna(),d.modis_ndvi.notna()],[0,1,2],-1)

def _maps(M, ids, alpha=10., min_common=20):
    """Pairwise affine maps and correlations for a date x AOI matrix."""
    n=M.shape[1]; maps={}; cor=np.full((n,n),np.nan)
    for i in range(n):
      for j in range(n):
        if i==j:continue
        ok=np.isfinite(M[:,i])&np.isfinite(M[:,j]); nn=int(ok.sum())
        if nn<min_common:continue
        x=M[ok,j];z=M[ok,i]; xm=np.median(x);zm=np.median(z);den=np.sum((x-xm)**2)+alpha;b=np.clip(np.sum((x-xm)*(z-zm))/den,-3,3);a=zm-b*xm
        maps[(i,j)]=(float(a),float(b));
        cor[i,j]=np.corrcoef(x,z)[0,1] if np.std(x)>1e-8 and np.std(z)>1e-8 else 0.
    return maps,cor

def _source_expert(d, known, qidx, sensor, k=8, alpha=10., use_factor=True):
    """Predict one raw sensor for q rows from same-date peers."""
    ids=d['id'].to_numpy(str); dates=d['datekey'].to_numpy(str); vals=pd.to_numeric(d[sensor],errors='coerce').to_numpy(float)
    idlist=np.array(sorted(np.unique(ids)),object); ip={x:i for i,x in enumerate(idlist)}; dtlist=np.array(sorted(np.unique(dates)),object); dp={x:i for i,x in enumerate(dtlist)}
    M=np.full((len(dtlist),len(idlist)),np.nan)
    # Sensor itself is observable only on non-gap rows; additionally require
    # target-known so a malformed secondary field cannot leak a query label.
    good=known & np.isfinite(vals)
    for i in np.flatnonzero(good):M[dp[dates[i]],ip[ids[i]]]=vals[i]
    maps,cor=_maps(M,idlist,alpha=alpha,min_common=15)
    # common date factor (standardized by AOI medians) can rescue peers with
    # few pairwise overlaps; maps are preferred when available.
    colmed=np.nanmedian(M,axis=0);colmed[~np.isfinite(colmed)]=np.nanmedian(vals[good]) if good.any() else .35
    centered=M-colmed[None,:]
    f=np.nanmedian(centered,axis=1); f[~np.isfinite(f)]=0.
    # AOI loading/offset against factor, fit on observed sensor values.
    load=np.zeros(len(idlist));
    for ci in range(len(idlist)):
      ok=np.isfinite(M[:,ci])&np.isfinite(f); den=np.sum(f[ok]**2)+alpha
      if ok.sum()>=8:load[ci]=np.clip(np.sum(f[ok]*(M[ok,ci]-colmed[ci]))/den,-3,3)
    out=np.full(len(qidx),np.nan); cover=np.zeros(len(qidx),int)
    for jj,i in enumerate(qidx):
      ci=ip[ids[i]];dt=dp[dates[i]]; cand=np.flatnonzero(np.isfinite(M[dt]))
      cand=cand[cand!=ci]
      # maps sorted by positive correlation; fallback to factor if no map.
      cand=cand[np.argsort(np.nan_to_num(cor[ci,cand],nan=-1))[::-1]] if len(cand) else cand
      vals2=[];ws=[]
      for cj in cand[:int(k)]:
        mp=maps.get((ci,cj));
        if mp is None:continue
        a,b=mp; vals2.append(a+b*M[dt,cj]); ws.append(max(float(cor[ci,cj]),.05))
      if vals2:
        out[jj]=float(np.average(vals2,weights=ws));cover[jj]=len(vals2)
      elif use_factor and np.isfinite(f[dt]):
        out[jj]=float(colmed[ci]+load[ci]*f[dt]);cover[jj]=0
    return out,cover,M

def _schedule_probs(d,known,qidx,mode='date',smooth=.5):
    """Observable source posterior from peer acquisition counts."""
    labels=src_label(d);ids=d['id'].to_numpy(str);dates=d['datekey'].to_numpy(str);doy=d['doyx'].to_numpy(int);yr=d['yearx'].to_numpy(int)
    # exact date counts and same AOI/DOY counts; current row omitted by design
    p=np.full((len(qidx),3),1/3.)
    for j,i in enumerate(qidx):
      if mode=='date': ii=np.flatnonzero((dates==dates[i])&known&(ids!=ids[i]))
      elif mode=='doy': ii=np.flatnonzero((doy==doy[i])&known&(ids!=ids[i]))
      else: ii=np.flatnonzero(known&(ids==ids[i])&(doy==doy[i]))
      c=np.array([(labels[ii]==s).sum() for s in range(3)],float)+smooth
      p[j]=c/c.sum()
    return p

def prepare_fold(frame, qmask):
    d=frame.copy().reset_index(drop=True);d[DATE]=pd.to_datetime(d[DATE]);d['id']=d[ID].astype(str);d['yearx']=d[DATE].dt.year.astype(int);d['doyx']=d[DATE].dt.dayofyear.astype(int);d['datekey']=d[DATE].dt.strftime('%Y-%m-%d');q=np.asarray(qmask,bool);known=d[TARGET].notna().to_numpy(bool)&~q;return d,known

def run_frame(frame,qmask,true=None,oracle=False):
    d,known=prepare_fold(frame,qmask);qidx=np.flatnonzero(qmask); y=d[TARGET].to_numpy(float)[qidx] if true is None else np.asarray(true,float)
    ex=[];cov=[]
    for s in SC:
      p,c,_=_source_expert(d,known,qidx,s,k=12,alpha=10.);ex.append(p);cov.append(c)
    ex=np.array(ex).T;cov=np.array(cov).T
    # source posterior from schedule; query source oracle is diagnostic only.
    probs=_schedule_probs(d,known,qidx,'date',.5)
    if oracle:
      lab=src_label(frame.reset_index(drop=True));probs=np.eye(3)[lab[qidx]]
    p=np.nansum(ex*probs,axis=1);bad=~np.isfinite(p)
    # fallback weighted raw peer median / global target median
    if bad.any():p[bad]=np.nanmedian(d.loc[known,TARGET].to_numpy(float))
    return p,y,probs,ex,cov

def one_exact(tr,pr,year):
    f,t=make_fold(tr.copy(),pr.copy(),year);q=f.is_synthetic_gap.fillna(False).to_numpy(bool);return run_frame(f,q,t.to_numpy(float),False)

def private_random(pr,seed=70404,frac=.15,year=None):
    d=pr.copy().reset_index(drop=True);d[DATE]=pd.to_datetime(d[DATE]);
    if year is not None:d=d[d[DATE].dt.year.eq(year)].copy().reset_index(drop=True)
    known=d[TARGET].notna().to_numpy(bool)&~d[GAP].fillna(False).to_numpy(bool);q=np.zeros(len(d),bool);rng=np.random.default_rng(seed);yrs=d[DATE].dt.year
    for _,ix0 in d.loc[known].groupby([ID,yrs],sort=False).groups.items():
      ix=np.asarray(ix0);q[rng.choice(ix,size=min(len(ix),max(1,int(round(frac*len(ix))))),replace=False)]=True
    dyn=[c for c in d.columns if c not in [ID,DATE,'crop_type',GAP,TARGET]]
    d.loc[q,[TARGET]+dyn]=np.nan;d.loc[q,GAP]=True;truth=pr.loc[d.index,TARGET].to_numpy(float) if year is None else pr.loc[d.index,TARGET].to_numpy(float)
    return run_frame(d,q,truth[q],False)

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False);tr[GAP]=False;pr[GAP]=pr[GAP].fillna(False).astype(bool);rows=[]
 for yr in [2019,2020,2021,2022,2023,2024]:
  p,y,pro,ex,c=one_exact(tr,pr,yr);rows.append({'protocol':'exact','year':yr,'method':'sensor_mix','n':len(y),'rmse':np.sqrt(np.mean((p-y)**2)),'mae':np.mean(abs(p-y))});
  po,_,_,_,_=one_exact(tr,pr,yr); # placeholder for oracle below
  f,t=make_fold(tr.copy(),pr.copy(),yr);q=f.is_synthetic_gap.fillna(False).to_numpy(bool);oo,yy,_,_,_=run_frame(f,q,t.to_numpy(float),True);rows.append({'protocol':'exact_oracle','year':yr,'method':'sensor_mix','n':len(yy),'rmse':np.sqrt(np.mean((oo-yy)**2)),'mae':np.mean(abs(oo-yy))});print('exact',yr,rows[-2]['rmse'],rows[-1]['rmse'],flush=True)
 for seed in [0,1,2]:
  for yr in [None,2025]:
   p,y,_,_,_=private_random(pr,seed,year=yr);rows.append({'protocol':'random2025' if yr else 'random','year':yr or 0,'seed':seed,'method':'sensor_mix','n':len(y),'rmse':np.sqrt(np.mean((p-y)**2)),'mae':np.mean(abs(p-y))});print('random',seed,yr,rows[-1]['rmse'],flush=True)
 out=pd.DataFrame(rows);out.to_csv(R/'sensor_factor_eval_results.csv',index=False);print(out.to_string(index=False))
if __name__=='__main__':main()

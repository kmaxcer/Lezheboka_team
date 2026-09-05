"""Test disjoint OOF pseudo-gap training for extended HGB (research only)."""
from __future__ import annotations
import sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); R=ROOT/'research'
sys.path.insert(0,str(ROOT/'src')); from validate import make_fold
sys.path.insert(0,str(R)); from build_extended_hgb_private import _clear,_fit,_matrix

TARGET='primary_ndvi'
def part_masks(d,exclude,nfold=5,seed=1):
    known=d[TARGET].notna().to_numpy(bool)&~np.asarray(exclude,bool); out=[np.zeros(len(d),bool) for _ in range(nfold)]
    rng=np.random.default_rng(seed); tab=pd.DataFrame({'id':d.anon_polygon_id.astype(str),'yr':pd.to_datetime(d.date).dt.year})
    for _,ix0 in tab.loc[known].groupby(['id','yr'],sort=False).groups.items():
        ix=np.asarray(ix0,int); rng.shuffle(ix)
        for j,ii in enumerate(ix): out[j%nfold][ii]=True
    return out
def run(d,query,exclude,label):
    d=d.copy().reset_index(drop=True); d.date=pd.to_datetime(d.date); d.year=d.year.fillna(d.date.dt.year).astype(int); d.doy=d.doy.fillna(d.date.dt.dayofyear).astype(int)
    if '_truth' not in d: d['_truth']=pd.to_numeric(d[TARGET],errors='coerce')
    bs=[];ys=[];t=time.time()
    for j,pm in enumerate(part_masks(d,exclude,5,1200),1):
        comb=np.asarray(exclude,bool)|pm; fr=_clear(d,comb); obs=fr[TARGET].where(~comb); x=_matrix(d,obs,comb); bs.append(x.loc[pm].reset_index(drop=True)); ys.append(d.loc[pm,'_truth'].reset_index(drop=True)); print(label,'block',j,int(pm.sum()),'sec',round(time.time()-t,1),flush=True)
    vf=_clear(d,exclude); obs=vf[TARGET].where(~exclude); qx=_matrix(d,obs,exclude).loc[query].reset_index(drop=True); xa=pd.concat(bs,ignore_index=True); ya=pd.concat(ys,ignore_index=True).astype(float); y=d.loc[query,'_truth'].to_numpy(float)
    rows=[]
    for kind in ('regular','wide'):
        m=_fit(kind,xa,ya,42); p=np.clip(m.predict(qx),-.2,1.1); rows.append({'protocol':label,'kind':kind,'n':len(y),'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(abs(p-y))),'train_n':len(xa),'features':xa.shape[1]}); print(label,kind,rows[-1]['rmse'],flush=True)
    return pd.DataFrame(rows)
def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False)
    out=[]
    # exact 2024
    f,t=make_fold(tr.copy(),pr.copy(),2024); q=f.is_synthetic_gap.fillna(False).to_numpy(bool); out.append(run(f,q,q,'exact2024_oof'))
    # 2025 proxy, 30% known holdout and real hidden excluded
    tr.is_synthetic_gap=False; pr.is_synthetic_gap=pr.is_synthetic_gap.fillna(False).astype(bool); d=pd.concat([tr,pr],ignore_index=True,sort=False); d.date=pd.to_datetime(d.date); d.year=d.year.fillna(d.date.dt.year).astype(int); d.doy=d.doy.fillna(d.date.dt.dayofyear).astype(int); d['_truth']=d[TARGET].astype(float); hid=d.is_synthetic_gap.to_numpy(bool); targ=(d.date.dt.year.eq(2025)&d[TARGET].notna()&~hid).to_numpy(); hold=np.zeros(len(d),bool); rng=np.random.default_rng(202403)
    for _,ix0 in d.loc[targ].groupby('anon_polygon_id',sort=False).groups.items():
        ix=np.asarray(ix0,int); n=max(1,int(round(.30*len(ix)))); hold[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
    out.append(run(d,hold,hid|hold,'proxy2025_oof'))
    z=pd.concat(out,ignore_index=True); z.to_csv(R/'extended_oof_results.csv',index=False); print(z.to_string(index=False))
if __name__=='__main__': main()

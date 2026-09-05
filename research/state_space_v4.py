"""Fast leakage-safe robust state-space-ish seasonal smoother screen."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R=ROOT/'research'; ID,DATE,Y,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
def smooth_predict(d, observed, qmask, bandwidth=25., seasonal_w=.25):
    x=d.copy(); x[DATE]=pd.to_datetime(x[DATE]); x['_doy']=x[DATE].dt.dayofyear.to_numpy(); y=pd.to_numeric(x[Y],errors='coerce').to_numpy(float)
    obs=np.asarray(observed,bool)&np.isfinite(y); out=np.full(len(x),np.nan)
    for _,gi0 in x.groupby(ID,sort=False).groups.items():
        gi=np.asarray(gi0,dtype=int); oi=gi[obs[gi]]; qi=gi[qmask[gi]]
        if len(oi)==0 or len(qi)==0: continue
        od=x['_doy'].to_numpy()[oi].astype(float); oy=np.clip(y[oi],-.2,1.1); med=np.nanmedian(oy); mad=np.nanmedian(np.abs(oy-med))+1e-3; oy=np.clip(oy,med-5*mad,med+5*mad)
        qd=x['_doy'].to_numpy()[qi].astype(float); dd=np.abs(od[None,:]-qd[:,None]); dd=np.minimum(dd,366-dd); ww=np.exp(-.5*(dd/bandwidth)**2)*(dd<=120); den=ww.sum(axis=1); local=(ww@oy)/np.maximum(den,1e-9)
        sd=np.empty(len(qi))
        for j,doy in enumerate(qd):
            z=np.minimum(np.abs(od-doy),366-np.abs(od-doy)); vals=oy[z<=10]; sd[j]=np.nanmedian(vals) if len(vals) else local[j]
        alpha=seasonal_w*np.exp(-den/3.); out[qi]=np.clip(np.where(den>0,(1-alpha)*local+alpha*sd,med),-.2,1.1)
    return out
def make_private_holdout(pr,seed=70404):
    known=pr[Y].notna().to_numpy(bool)&~pr[GAP].fillna(False).to_numpy(bool); out=np.zeros(len(pr),bool); rng=np.random.default_rng(seed); yy=pd.to_datetime(pr[DATE]).dt.year
    for _,ix0 in pr.loc[known].groupby([ID,yy],sort=False).groups.items():
        ix=np.asarray(ix0,dtype=int); n=max(1,int(round(.15*len(ix)))); out[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
    return out
def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False); pr[GAP]=pr[GAP].fillna(False).astype(bool); hold=make_private_holdout(pr); hidden=pr[GAP].to_numpy(bool); gaps=hold|hidden; tr[GAP]=False
    ref=pd.concat([tr,pr],ignore_index=True,sort=False); ref[DATE]=pd.to_datetime(ref[DATE]); hk=set(map(tuple,pr.loc[gaps,[ID,DATE]].to_numpy())); qref=np.array([tuple(v) in hk for v in ref[[ID,DATE]].to_numpy()],bool); observed=ref[Y].notna().to_numpy(bool)&~qref; p=smooth_predict(ref,observed,qref)
    q=pd.DataFrame({ID:ref.loc[qref,ID].to_numpy(),DATE:ref.loc[qref,DATE].to_numpy(),'state':p[qref]}); keys=pr.loc[hold,[ID,DATE,Y]].rename(columns={Y:'truth'}).copy(); keys[DATE]=pd.to_datetime(keys[DATE]); q[DATE]=pd.to_datetime(q[DATE]); keys=keys.merge(q,on=[ID,DATE],how='left',validate='one_to_one')
    old=pd.read_csv(R/'private_cohort_blend_holdout_predictions.csv',parse_dates=[DATE],low_memory=False); v3=pd.read_csv(R/'v3_private_holdout_predictions.csv',parse_dates=[DATE],low_memory=False); keys=keys.merge(old[[ID,DATE,'ext40']],on=[ID,DATE],how='left',validate='one_to_one').merge(v3[[ID,DATE,'v3']],on=[ID,DATE],how='left',validate='one_to_one'); keys['base']=.7*keys.ext40+.3*keys.v3; keys['cohort']=np.where(keys[ID].isin(set(tr[ID])),'shared','new'); keys['year']=keys[DATE].dt.year
    rows=[]
    for w in [0,.03,.05,.08,.1,.15,.2,.25,.3,.4]:
        keys[f'b{w}']=(1-w)*keys.base+w*keys.state
        for gn,g in [('all',keys),('history',keys[keys.year<2025]),('2025',keys[keys.year==2025]),('new2025',keys[(keys.cohort=='new')&(keys.year==2025)]),('shared2025',keys[(keys.cohort=='shared')&(keys.year==2025)])]: rows.append({'group':gn,'w':w,'n':len(g),'rmse':float(np.sqrt(np.mean((g[f'b{w}']-g.truth)**2)))})
    res=pd.DataFrame(rows); res.to_csv(R/'state_space_v4_private_holdout_results.csv',index=False); keys.to_csv(R/'state_space_v4_private_holdout_predictions.csv',index=False); print(res.to_string(index=False)); print('query',int(gaps.sum()),'hold',int(hold.sum()))
if __name__=='__main__': main()

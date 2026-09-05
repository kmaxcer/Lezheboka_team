"""Fast leakage-safe audit of direct same-date spatial sensor values.

This is deliberately independent of source-expert model fitting.  For every
outer mask, train rows and visible private rows are indexed by date; hidden
queries receive medians/means of nearby AOI sensor values.  The resulting
source-specific values are affine-calibrated on train-known rows and mixed by
the observable acquisition posterior.  Only relative blend gains are used to
decide whether this weak signal is worth adding to the route candidate.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R=ROOT/"research"; REPORT=ROOT/"reports"
ID,DATE,TARGET,GAP="anon_polygon_id","date","primary_ndvi","is_synthetic_gap"
SRC=("s2","landsat","modis")
SCOLS=("s2_ndvi","landsat_ndvi","modis_ndvi")
SEEDS=(0,1,2,70404)
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0,str(R))
from evaluate_private_cohort_blend import make_holdout  # noqa: E402
from source_expert_route_v2 import _baseline_for_seed, _masked_private  # noqa: E402
from overnight_source_eval import _source_labels, _predict_matrix  # noqa: E402

def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan

def id_num(x):
    # IDs are AOI-0001 etc.  Keep the parser explicit and robust to malformed IDs.
    return pd.to_numeric(pd.Series(x,dtype='string').str.extract(r"(\d+)",expand=False),errors='coerce').fillna(-1).to_numpy(np.int32)

def fit_affine(tr):
    out=[]
    for c in SCOLS:
        z=tr[[c,TARGET]].dropna(); x=z[c].to_numpy(float); y=z[TARGET].to_numpy(float)
        if len(z)<10: out.append((1.,0.)); continue
        a,b=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]
        out.append((float(np.clip(a,.5,1.5)),float(np.clip(b,-.25,.25))))
    return np.asarray(out,float)

def posterior(pm,tr):
    """Observable schedule posterior keyed by private positional row."""
    out=np.full((len(pm),3),1/3.,float)
    mat,_=_predict_matrix(pm,train=tr,family='base',k=8,degree=1,bin_days=30,date_weight=1.)
    if len(mat):
        mm=mat.set_index('row_index')
        for j,c in enumerate(('p_s2','p_landsat','p_modis')):
            if c in mm:
                ix=mm.index.to_numpy(np.int64); out[ix,j]=mm[c].to_numpy(float)
    out=np.where(np.isfinite(out),out,1/3.); out/=out.sum(1,keepdims=True); return out

def spatial_values(pm,tr,gaps,qidx,radius,crop,method):
    """Vectorized by-date same-date sensor summaries for query rows."""
    n=len(qidx); out=np.full((n,3),np.nan); cnt=np.zeros((n,3),np.int16); near=np.full(n,np.inf)
    tr2=tr[[ID,DATE,'crop_type']+list(SCOLS)].copy(); tr2[DATE]=pd.to_datetime(tr2[DATE])
    p2=pm[[ID,DATE,'crop_type']+list(SCOLS)].copy(); p2[DATE]=pd.to_datetime(p2[DATE])
    allf=pd.concat([tr2,p2],ignore_index=True,sort=False)
    aids=id_num(allf[ID]); adates=allf[DATE].to_numpy(); acrop=allf.crop_type.fillna('unknown').astype(str).to_numpy()
    av=[allf[c].to_numpy(float) for c in SCOLS]
    vis=np.r_[np.ones(len(tr2),bool),(~np.asarray(gaps,bool))]
    qdates=p2[DATE].to_numpy()[qidx]; qaids=aids[len(tr2)+qidx]; qcrop=p2.crop_type.fillna('unknown').astype(str).to_numpy()[qidx]
    # Group candidate and query positions by exact date.  Date cardinality is
    # small, so each broadcast matrix is at most O(78*78).
    cand_by={dt:ix for dt,ix in pd.Series(np.flatnonzero(vis),index=np.flatnonzero(vis)).groupby(adates[vis])}
    q_by={dt:ix for dt,ix in pd.Series(np.arange(n),index=np.arange(n)).groupby(qdates)}
    for dt,qi in q_by.items():
        qi=np.asarray(qi,dtype=int); z0=np.asarray(cand_by.get(dt,np.empty(0,int)),dtype=int)
        if len(z0)==0: continue
        # Exclude the same AOI (train/private duplicate key) and apply radius/crop.
        dist=np.abs(qaids[qi,None]-aids[z0][None,:]); ok=(dist<=radius)&(qaids[qi,None]!=aids[z0][None,:])
        if crop: ok &= (qcrop[qi,None]==acrop[z0][None,:])
        if not ok.any(): continue
        near[qi]=np.where(ok,dist,np.inf).min(axis=1)
        for j in range(3):
            vv=np.broadcast_to(av[j][z0][None,:],(len(qi),len(z0))).copy(); vv[~ok]=np.nan
            cnt[qi,j]=np.isfinite(vv).sum(axis=1).astype(np.int16)
            if method=='mean': out[qi,j]=np.nanmean(vv,axis=1)
            elif method=='trim':
                # A trimmed mean is useful only with at least 5 peers; median fallback otherwise.
                for k,row in enumerate(vv):
                    x=row[np.isfinite(row)]
                    if len(x)==0: continue
                    if len(x)>=5:
                        lo,hi=np.quantile(x,[.1,.9]); x=x[(x>=lo)&(x<=hi)]
                    out[qi[k],j]=float(np.mean(x)) if len(x) else np.nan
            else: out[qi,j]=np.nanmedian(vv,axis=1)
    return out,cnt,near

def base_for(seed,pr,hold):
    # Seed 2 was generated by the standalone route runner; use its saved
    # baseline sidecar because the generic baseline cache predates that mask.
    if int(seed)==2:
        z=pd.read_csv(R/'source_expert_route_v2_seed2_rows.csv',parse_dates=[DATE],low_memory=False)
        b=z[[ID,DATE,'baseline']]
    else:
        b=_baseline_for_seed(seed,pr,hold)
    keys=pd.MultiIndex.from_frame(pr.loc[hold,[ID,DATE]])
    return b.set_index([ID,DATE]).loc[keys,'baseline'].to_numpy(float)

def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False)
    pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
    cal=fit_affine(tr); print('cal',dict(zip(SRC,cal.tolist())),flush=True)
    rows=[]; qrows=[]
    for seed in SEEDS:
        hold=make_holdout(pr,seed=int(seed)); pm,gaps=_masked_private(pr,hold); qi=np.flatnonzero(hold)
        y=pr.loc[hold,TARGET].to_numpy(float); B=base_for(seed,pr,hold); P=posterior(pm,tr)[qi]; ts=_source_labels(pr)[hold]
        cohort=np.where(pr.loc[hold,ID].astype(str).isin(set(tr[ID].astype(str))),'shared','new'); year=pr.loc[hold,DATE].dt.year.to_numpy(int)
        # Fast first-pass shortlist.  Median/mean no-crop variants are only
        # added for radii where a crop-aware peer exists; trim was previously
        # dominated and is omitted to keep the 4-mask audit bounded.
        for radius in (1,2,4,8,16,32):
          for crop in (True,False):
            for method in ('median','mean'):
              V,C,N=spatial_values(pm,tr,gaps,qi,radius,crop,method)
              V=V*cal[:,0][None,:]+cal[:,1][None,:]; ok=np.isfinite(V); W=P*ok; den=W.sum(1)
              mix=np.divide(np.nan_to_num(V)*W,den[:,None],out=np.full_like(V,np.nan),where=den[:,None]>0).sum(1)
              # For missing direct values, retain baseline. Evaluate both source posterior mix and hard source mode.
              hard=V[np.arange(len(V)),P.argmax(1)]; hard=np.where(np.isfinite(hard),hard,mix)
              for name,Z in [('mix',mix),('hard',hard)]:
                rows.append(dict(seed=seed,radius=radius,crop=int(crop),method=method,pred=name,n=len(y),rmse=rmse(y,Z),coverage=float(np.isfinite(Z).mean())))
                for alpha in (.01,.02,.03,.05,.08,.10,.15,.20,.30,.40):
                    Zb=np.where(np.isfinite(Z),(1-alpha)*B+alpha*Z,B)
                    rows.append(dict(seed=seed,radius=radius,crop=int(crop),method=method,pred=f'blend{alpha:g}',n=len(y),rmse=rmse(y,Zb),coverage=float(np.isfinite(Z).mean())))
              # Keep rows for a compact, likely useful shortlist and direct source diagnostics.
              if crop and radius in (1,2,4,8,16):
                for k,ix in enumerate(qi):
                  qrows.append(dict(seed=seed,private_index=int(ix),anon_polygon_id=pr.iloc[ix][ID],date=pr.iloc[ix][DATE],truth=float(y[k]),baseline=float(B[k]),mix=float(mix[k]) if np.isfinite(mix[k]) else np.nan,hard=float(hard[k]) if np.isfinite(hard[k]) else np.nan,near_dist=float(N[k]),n_s2=int(C[k,0]),n_landsat=int(C[k,1]),n_modis=int(C[k,2]),true_src=ts[k],year=int(year[k]),cohort=cohort[k],radius=radius,method=method))
        print('seed',seed,'done',flush=True)
    m=pd.DataFrame(rows); m.to_csv(R/'direct_spatial_sensor_fast_metrics_20260905.csv',index=False)
    q=pd.DataFrame(qrows); q.to_csv(R/'direct_spatial_sensor_fast_rows_20260905.csv',index=False)
    # Pooled RMSE from equal n per seed, and robust all-seed criterion.
    pp=[]
    for key,g in m.groupby(['radius','crop','method','pred'],sort=False):
        pp.append(dict(radius=key[0],crop=key[1],method=key[2],pred=key[3],pooled_rmse=float(np.sqrt(np.average(g.rmse.to_numpy()**2,weights=g.n.to_numpy()))),min_seed_rmse=float(g.rmse.min()),max_seed_rmse=float(g.rmse.max()),mean_delta_vs_global40=np.nan))
    p=pd.DataFrame(pp).sort_values('pooled_rmse'); p.to_csv(R/'direct_spatial_sensor_fast_pooled_20260905.csv',index=False)
    print(p.head(40).to_string(index=False),flush=True)
    best=p.iloc[0].to_dict() if len(p) else {}
    REPORT.mkdir(exist_ok=True); (REPORT/'direct_spatial_sensor_fast_report_20260905.md').write_text('# Direct same-date spatial sensor audit (fast)\n\nLeakage-safe: only train + visible private rows enter same-date neighbours; all organiser gaps and holdout rows have dynamic fields masked. Sensor values are affine-calibrated on train-known rows and mixed by observable schedule posterior.\n\nBest pooled row: '+json.dumps(best,ensure_ascii=False)+'\n\nArtifacts: `research/direct_spatial_sensor_fast_metrics_20260905.csv`, `research/direct_spatial_sensor_fast_rows_20260905.csv`, `research/direct_spatial_sensor_fast_pooled_20260905.csv`.\n',encoding='utf-8')

if __name__=='__main__': main()

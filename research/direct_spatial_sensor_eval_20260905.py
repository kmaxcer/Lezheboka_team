"""Leakage-safe direct same-date spatial sensor-value interpolation audit.

For each pseudo-private mask, only visible rows (train + unmasked private) are
used.  Query rows receive per-sensor medians/weighted medians from nearby AOIs
on the same date, optionally crop restricted.  The source schedule posterior
is estimated from visible same-date source counts.  This is intentionally
separate from the source-expert route and can be blended conservatively with
the saved route baseline.
"""
from __future__ import annotations
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
SENSORS = {"s2":"s2_ndvi", "landsat":"landsat_ndvi", "modis":"modis_ndvi"}
SRC = list(SENSORS)
SEEDS = [0, 1, 2, 70404]

sys.path.insert(0, str(R))
from evaluate_private_cohort_blend import make_holdout  # noqa: E402
from source_expert_route_v2 import _baseline_for_seed, _masked_private, _neighbor_counts, ROUTE_RADII  # noqa: E402
from overnight_source_eval import _source_labels, _predict_matrix  # noqa: E402

def rmse(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok]-y[ok])**2))) if ok.any() else np.nan

def id_num(x):
    z = pd.Series(x, dtype="string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(z, errors="coerce").fillna(-1).to_numpy(int)

def robust_mean(v, method="median"):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)==0: return np.nan
    if method=="mean": return float(np.mean(v))
    if method=="trim":
        if len(v)>=5:
            lo,hi=np.quantile(v,[.1,.9]); v=v[(v>=lo)&(v<=hi)]
        return float(np.mean(v)) if len(v) else np.nan
    return float(np.median(v))

def sensor_affine_fit(tr, source):
    """Robust source->primary affine calibration on train known rows."""
    c=SENSORS[source]; z=tr[[c,TARGET]].dropna();
    if len(z)<20: return 1.,0.
    x=z[c].to_numpy(float); y=z[TARGET].to_numpy(float)
    # OLS is exact for S2 and stable for other sensors; clip implausible slope.
    A=np.column_stack([x,np.ones(len(x))]); coef=np.linalg.lstsq(A,y,rcond=None)[0]
    a,b=map(float,coef); a=float(np.clip(a,.5,1.5)); b=float(np.clip(b,-.25,.25)); return a,b

def source_posteriors(pm, tr):
    """Return per-private-row source schedule posterior from visible frame."""
    mat,_ = _predict_matrix(pm, train=tr, family="base", k=8, degree=1, bin_days=30, date_weight=1.0)
    out=np.full((len(pm),3),1/3,float)
    if len(mat):
        mm=mat.set_index("row_index")
        cols=["p_s2","p_landsat","p_modis"]
        for j,c in enumerate(cols):
            if c in mm:
                ix=mm.index.to_numpy(int); out[ix,j]=mm[c].to_numpy(float)
    out=np.where(np.isfinite(out),out,1/3); out/=out.sum(1,keepdims=True); return out

def build_spatial(pm, tr, gaps, qidx, radius=8, crop=True, method="median", weighted=False):
    """Per-query direct sensor predictions from same-date neighbouring AOIs."""
    n=len(qidx); out=np.full((n,3),np.nan); cnt=np.zeros((n,3),int); near=np.full(n,np.inf)
    d=pm.reset_index(drop=True); dates=pd.to_datetime(d[DATE]).to_numpy(); crops=d.crop_type.fillna("unknown").astype(str).to_numpy(); ids=id_num(d[ID]);
    # Visible sensors only. train rows are prepended and retain source values.
    tr2=tr.copy(); tr2[DATE]=pd.to_datetime(tr2[DATE]); tr2[GAP]=False
    # Use a single frame with explicit duplicate rows; train values are always visible.
    # pm already includes private rows; append train so same-date train-only neighbors enter.
    # Duplicate keys are harmless; query rows are excluded by visible mask below.
    allf=pd.concat([tr2[[ID,DATE,"crop_type"]+list(SENSORS.values())], d[[ID,DATE,"crop_type"]+list(SENSORS.values())]],ignore_index=True,sort=False)
    adates=pd.to_datetime(allf[DATE]).to_numpy(); acrops=allf.crop_type.fillna("unknown").astype(str).to_numpy(); aids=id_num(allf[ID])
    sensor_arr=[allf[SENSORS[s]].to_numpy(float) for s in SRC]
    visp=~np.asarray(gaps,bool)
    vis=np.r_[np.ones(len(tr2),bool),visp]
    bydate={dt:np.asarray(ix,dtype=int) for dt,ix in pd.Series(np.flatnonzero(vis),index=np.flatnonzero(vis)).groupby(adates[vis])}
    # Query metadata from pm rows; direct lookup by date/id.
    for n0,q in enumerate(np.asarray(qidx,int)):
        dt=dates[q]; aid=ids[q]; crop=crops[q]; z0=bydate.get(dt,np.empty(0,int))
        if len(z0)==0: continue
        z=z0[np.abs(aids[z0]-aid)<=radius]
        # remove same AOI; a duplicate train row would otherwise leak same key if present
        z=z[aids[z]!=aid]
        if crop:
            zc=z[acrops[z]==crop]
            if len(zc): z=zc
        if len(z)==0: continue
        dd=np.abs(aids[z]-aid); near[n0]=float(np.min(dd))
        for j,s in enumerate(SRC):
            v=sensor_arr[j][z]
            ok=np.isfinite(v); v=v[ok]
            if len(v):
                if weighted:
                    # nearest inverse-distance weighted mean; exact peers impossible after exclusion
                    w=1/np.maximum(1.,dd[ok]); out[n0,j]=float(np.sum(v*w)/np.sum(w))
                else: out[n0,j]=robust_mean(v,method)
                cnt[n0,j]=len(v)
    return out,cnt,near

def main():
    tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False)
    pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False)
    allrows=[]; metrics=[]
    # Source calibration from static train only; sensor values are target-domain candidates.
    cal=np.array([sensor_affine_fit(tr,s) for s in SRC]); print('calibration',dict(zip(SRC,cal.tolist())),flush=True)
    for seed in SEEDS:
        hold=make_holdout(pr,seed=int(seed)); pm,gaps=_masked_private(pr,hold)
        qi=np.flatnonzero(hold); y=pr.loc[hold,TARGET].to_numpy(float)
        B=_baseline_for_seed(seed,pr,hold).set_index([ID,DATE]).loc[pd.MultiIndex.from_frame(pr.loc[hold,[ID,DATE]])].baseline.to_numpy(float)
        post=source_posteriors(pm,tr)[qi]
        true_src=_source_labels(pr)[hold]
        for radius in [1,2,4,8,16,32]:
          for crop in [True,False]:
           for method in ["median","mean","trim"]:
            vals,cnt,near=build_spatial(pm,tr,gaps,qi,radius=radius,crop=crop,method=method)
            # Calibrate each source sensor to primary target, then posterior mix.
            vc=vals*cal[:,0][None,:]+cal[:,1][None,:]
            valid=np.isfinite(vc); w=post*valid; den=w.sum(1); pred=np.divide((np.nan_to_num(vc)*w).sum(1),den,out=np.full(len(qi),np.nan),where=den>0)
            # hard source post mode and each source are diagnostic
            for nm,p in [("raw_mix",pred),("baseline",B)]:
              metrics.append(dict(seed=seed,radius=radius,crop=int(crop),method=method,pred=nm,n=len(y),rmse=rmse(y,p),coverage=float(np.isfinite(p).mean())))
            # conservative baseline blend grid; only where spatial pred available
            for alpha in [0.01,.02,.03,.05,.08,.10,.15,.20,.30,.40]:
              p=np.where(np.isfinite(pred),(1-alpha)*B+alpha*pred,B)
              metrics.append(dict(seed=seed,radius=radius,crop=int(crop),method=method,pred=f"blend_{alpha:g}",n=len(y),rmse=rmse(y,p),coverage=float(np.isfinite(pred).mean())))
            # Save only compact best-configuration rows later; retain per query for diagnostics.
            if radius in (2,4,8) and crop and method in ("median","trim"):
                for i,(ix,yy,bb,pp) in enumerate(zip(qi,y,B,pred)):
                    allrows.append(dict(seed=seed,private_index=int(ix),anon_polygon_id=pr.iloc[ix][ID],date=pr.iloc[ix][DATE],truth=yy,baseline=bb,spatial=pp,near_dist=near[i],n_s2=int(cnt[i,0]),n_landsat=int(cnt[i,1]),n_modis=int(cnt[i,2]),true_src=true_src[i],year=int(pr.iloc[ix][DATE].year),cohort="shared" if str(pr.iloc[ix][ID]) in set(tr[ID].astype(str)) else "new",radius=radius,method=method))
    m=pd.DataFrame(metrics); m.to_csv(R/"direct_spatial_sensor_metrics_20260905.csv",index=False)
    a=pd.DataFrame(allrows); a.to_csv(R/"direct_spatial_sensor_rows_20260905.csv",index=False)
    # pooled score over equal-size masks (all rows) and per-seed best blend summaries
    out=[]
    for keys,g in m.groupby(["radius","crop","method","pred"],sort=False):
        # Recompute pooled RMSE from per-seed MSE weighted n.
        if g.pred.iloc[0] == "baseline": continue
        out.append(dict(radius=keys[0],crop=keys[1],method=keys[2],pred=keys[3],pooled_rmse=float(np.sqrt(np.average(g.rmse.to_numpy()**2,weights=g.n.to_numpy()))),min_seed_rmse=float(g.rmse.min()),max_seed_rmse=float(g.rmse.max())))
    oo=pd.DataFrame(out).sort_values("pooled_rmse"); oo.to_csv(R/"direct_spatial_sensor_pooled_20260905.csv",index=False)
    print(oo.head(30).to_string(index=False),flush=True)
    # report
    best=oo.iloc[0].to_dict() if len(oo) else {}
    report=R.parent/"reports"/"direct_spatial_sensor_report_20260905.md"; report.parent.mkdir(exist_ok=True)
    report.write_text("# Direct same-date spatial sensor audit (2026-09-05)\n\n"+"Only visible train + unmasked private rows enter each same-date neighbour set; target/sensors are masked on organiser gaps + holdout. Per-source sensor values are train-fitted affine-calibrated then mixed by observable schedule posterior.\n\n"+f"Best pooled configuration: `{best}`. Full metrics: `research/direct_spatial_sensor_metrics_20260905.csv`; query rows: `research/direct_spatial_sensor_rows_20260905.csv`; pooled table: `research/direct_spatial_sensor_pooled_20260905.csv`.\n",encoding="utf-8")

if __name__=='__main__': main()

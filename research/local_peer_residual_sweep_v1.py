"""Fast sweep of local peer residual profile/weighting choices.

This is an audit-only companion to ``local_peer_residual_v1``.  It reuses the
trainaug-r2 route rows and tests seasonal-bin widths, source-specific profile
fallbacks, radii, and robust peer aggregators on four independent masks.
"""
from __future__ import annotations
from pathlib import Path
import sys, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(R))
from teammate_sweep_postcorr import _mask_private  # noqa: E402
from local_peer_residual_v1 import _profile  # noqa: E402

ID, DATE = "anon_polygon_id", "date"; SEEDS = (0, 1, 2, 70404)


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); return float(np.sqrt(np.mean((p-y)**2)))


def feature(d, known, qidx, width, radius, source_level, agg):
    d = d.reset_index(drop=True).copy(); d[DATE] = pd.to_datetime(d[DATE])
    ids = pd.to_numeric(d[ID].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(-1).to_numpy(int)
    crops = d.crop_type.fillna("unknown").astype(str).to_numpy(); y = pd.to_numeric(d._truth, errors="coerce").to_numpy(float)
    p = _profile(d, known, width=width, source_level=source_level); r = np.clip(y-p, -.5, .5)
    ok = np.asarray(known, bool) & np.isfinite(r); vi = np.flatnonzero(ok); bydate = {}
    for dt, gi in pd.Series(vi, index=vi).groupby(d.loc[vi, DATE]): bydate[dt] = np.asarray(gi, int)
    out = np.full(len(qidx), np.nan)
    for j, qi in enumerate(qidx):
        z = bydate.get(d.at[qi, DATE], np.empty(0, int)); z = z[ids[z] != ids[qi]]
        z = z[crops[z] == crops[qi]] if len(z) else z
        if len(z): z = z[np.abs(ids[z]-ids[qi]) <= radius]
        if len(z) == 0: continue
        rr = r[z]; dd = np.abs(ids[z]-ids[qi]).astype(float); good = np.isfinite(rr); rr=rr[good];dd=dd[good]
        if len(rr)==0: continue
        if agg == "median": out[j] = np.median(rr)
        elif agg == "mean": out[j] = np.mean(rr)
        elif agg == "inv1": out[j] = np.sum(rr/np.maximum(1.,dd))/np.sum(1./np.maximum(1.,dd))
        elif agg == "inv2": out[j] = np.sum(rr/np.maximum(1.,dd)**2)/np.sum(1./np.maximum(1.,dd)**2)
        elif agg == "exp":
            w=np.exp(-dd/4.);out[j]=np.sum(w*rr)/np.sum(w)
        elif agg == "trim":
            if len(rr)>=5:
                so=np.sort(rr); k=max(1,int(.15*len(so))); out[j]=np.mean(so[k:-k])
            else: out[j]=np.mean(rr)
    return out


def feature_multi(d, known, qidx, width, source_level, radii=(2,4,8,16,32)):
    """Compute all radius/aggregator variants in one pass over each query."""
    d = d.reset_index(drop=True).copy(); d[DATE] = pd.to_datetime(d[DATE])
    ids = pd.to_numeric(d[ID].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(-1).to_numpy(int)
    crops = d.crop_type.fillna("unknown").astype(str).to_numpy(); y = pd.to_numeric(d._truth, errors="coerce").to_numpy(float)
    p = _profile(d, known, width=width, source_level=source_level); r = np.clip(y-p, -.5, .5)
    ok = np.asarray(known, bool) & np.isfinite(r); vi = np.flatnonzero(ok); bydate = {}
    for dt, gi in pd.Series(vi, index=vi).groupby(d.loc[vi, DATE]): bydate[dt] = np.asarray(gi, int)
    aggs = ("inv1", "inv2", "median", "mean", "exp")
    out = {(rad, a): np.full(len(qidx), np.nan) for rad in radii for a in aggs}
    for j, qi in enumerate(qidx):
        z = bydate.get(d.at[qi, DATE], np.empty(0, int)); z = z[ids[z] != ids[qi]]
        z = z[crops[z] == crops[qi]] if len(z) else z
        for rad in radii:
            zr = z[np.abs(ids[z]-ids[qi]) <= rad] if len(z) else z
            if len(zr) == 0: continue
            rr = r[zr]; dd = np.abs(ids[zr]-ids[qi]).astype(float); good = np.isfinite(rr); rr=rr[good];dd=dd[good]
            if len(rr)==0: continue
            out[(rad,"median")][j] = np.median(rr); out[(rad,"mean")][j] = np.mean(rr)
            w=1./np.maximum(1.,dd); out[(rad,"inv1")][j] = np.sum(w*rr)/np.sum(w)
            w=1./np.maximum(1.,dd)**2; out[(rad,"inv2")][j] = np.sum(w*rr)/np.sum(w)
            w=np.exp(-dd/4.); out[(rad,"exp")][j] = np.sum(w*rr)/np.sum(w)
    return out


def base_rows():
    rows=pd.read_csv(R/"source_expert_route_v2_fixed_radius_trainaug_rows.csv",parse_dates=[DATE],low_memory=False); probe=pd.read_csv(R/"source_schedule_route_probe_rows.csv",parse_dates=[DATE],low_memory=False); q=rows.merge(probe[[ID,DATE,"seed","sp_crop_2_n","sp_crop_8_n"]],on=[ID,DATE,"seed"],validate="one_to_one"); near=q.sp_crop_2_n.fillna(0).to_numpy()>0;mid=(~near)&(q.sp_crop_8_n.fillna(0).to_numpy()>0);yr=q.year.to_numpy(int);co=q.cohort.astype(str).to_numpy();a=np.where(near,.5,np.where(mid,.4,.3));a=np.where((co=='new')&(yr==2025),.6,a);a=np.where((co=='shared')&(yr==2025),.35,a);q['route_base']=(1-a)*q.baseline+a*q.expert_trainaug_r2;return q


def run():
    tr=pd.read_csv(DATA/"train_dataset.csv",parse_dates=[DATE],low_memory=False);pr=pd.read_csv(DATA/"private_features.csv",parse_dates=[DATE],low_memory=False);q=base_rows(); parts=[]
    # Build one feature per configuration and seed.  The resulting table is
    # intentionally compact to keep this sweep manageable.
    cfg=[]
    for w in (8,12,16,20,24,32,45,60):
        for rad in (2,4,8,16,32):
            for src in (False,True):
                for agg in ("inv1","inv2","median","mean","exp"):
                    cfg.append((w,rad,src,agg))
    rec=[]
    # Private mask/key frames are invariant across configurations; cache them
    # to avoid repeatedly copying the 57k-row frame during evaluation.
    mask_cache = {}
    for seed in SEEDS:
        f, m = _mask_private(pr, int(seed)); kk = f.loc[m, [ID, DATE]].copy().reset_index(drop=True); kk[DATE] = pd.to_datetime(kk[DATE]); mask_cache[int(seed)] = (f, m, kk)
    # Cache all aggregators/radii for each width/profile and seed.  This keeps
    # the sweep bounded while retaining the same leakage-safe construction.
    cache={}
    for wi,w in enumerate((8,12,16,20,24,32,45,60)):
        for src in (False,True):
            for seed in SEEDS:
                f,m,_ = mask_cache[int(seed)]; tr0=tr.copy(); tr0['_truth']=pd.to_numeric(tr0.primary_ndvi,errors='coerce');tr0['_hidden']=False;f=f.copy();f['_truth']=pd.to_numeric(f.primary_ndvi,errors='coerce');f['_hidden']=m;cof=pd.concat([tr0,f],ignore_index=True,sort=False); known=cof.primary_ndvi.notna().to_numpy(bool)&~cof._hidden.to_numpy(bool);qidx=np.flatnonzero(np.r_[np.zeros(len(tr),bool),m]); cache[(w,src,seed)] = feature_multi(cof,known,qidx,w,src)
            print('prepared width/profile',w,src,flush=True)
    # Evaluate each cached variant against the route-r2 base.
    # Build one merged table per width/profile with all radius/aggregator
    # columns; each subsequent configuration is then a cheap column lookup.
    merged_cache = {}
    for w in (8,12,16,20,24,32,45,60):
        for src in (False,True):
            vals=[]
            for seed in SEEDS:
                _,_,keys = mask_cache[int(seed)]; ff=keys.copy(); ff['seed']=int(seed)
                for (rad, an), arr in cache[(w,src,seed)].items(): ff[f'f_{rad}_{an}']=arr
                vals.append(ff)
            merged_cache[(w,src)] = q.merge(pd.concat(vals,ignore_index=True),on=[ID,DATE,'seed'],how='left',validate='one_to_one')
    for ci,(w,rad,src,agg_name) in enumerate(cfg):
        zz=merged_cache[(w,src)]; zz=zz.copy(); zz['feat']=zz[f'f_{rad}_{agg_name}']
        for held in SEEDS:
            te=zz[zz.seed==held]; trn=zz[zz.seed!=held]; x=trn.feat.to_numpy(float); rr=trn.truth.to_numpy(float)-trn.route_base.to_numpy(float);ok=np.isfinite(x); aa=float(np.clip(np.dot(x[ok],rr[ok])/max(np.dot(x[ok],x[ok]),1e-9),-.8,.8));
            for mode,a in [('loo',aa),('fixed020',.20)]:
                p=np.clip(te.route_base.to_numpy(float)+a*np.nan_to_num(te.feat.to_numpy(float)),-.5,1.2); rec.append({'width':w,'radius':rad,'source_profile':src,'agg':agg_name,'held_seed':int(held),'mode':mode,'alpha':a,'n':len(te),'coverage':float(np.isfinite(te.feat).mean()),'rmse':rmse(te.truth,p),'base_rmse':rmse(te.truth,te.route_base)})
        if (ci+1) % 50 == 0: print('evaluated cfg',ci+1,'/',len(cfg),flush=True)
    out=pd.DataFrame(rec);out['delta']=out.rmse-out.base_rmse;out.to_csv(R/'local_peer_residual_sweep_v1_results.csv',index=False,float_format='%.10f');agg=out.groupby(['width','radius','source_profile','agg','mode'],as_index=False).apply(lambda g:pd.Series({'n':int(g.n.sum()),'rmse_pooled':float(np.sqrt(np.average(g.rmse**2,weights=g.n))),'base_rmse_pooled':float(np.sqrt(np.average(g.base_rmse**2,weights=g.n))),'delta_pooled':float(np.sqrt(np.average(g.rmse**2,weights=g.n))-np.sqrt(np.average(g.base_rmse**2,weights=g.n))),'wins':int((g.rmse<g.base_rmse).sum()),'coverage':float(np.average(g.coverage,weights=g.n))}),include_groups=False).reset_index(drop=True).sort_values('rmse_pooled');agg.to_csv(R/'local_peer_residual_sweep_v1_aggregate.csv',index=False,float_format='%.10f');(R/'local_peer_residual_sweep_v1_report.md').write_text('# Local peer residual sweep v1\n\n'+agg.head(80).to_string(index=False)+'\n',encoding='utf8');print(agg.head(60).to_string(index=False),flush=True)


if __name__=='__main__':run()

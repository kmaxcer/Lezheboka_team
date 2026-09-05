"""Small fast sweep around the observable peer/shock rule.

This intentionally tests only configurations that survived the full AOI-peer
screen and reports every leave-year/leave-seed fold, making coefficient
stability visible without the multi-million-row retrospective grid.
"""
from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
DOY = {97,113,129,145,161,177,193,209,225,241,257,273,289}


def norm(x):
    x = x.astype(str)
    return x.str.replace(r"^(exact)(\d+)$", r"exact_\2", regex=True).str.replace(r"^(random)(\d+)$", r"random_\2", regex=True)


def load():
    p = pd.read_csv(R / "paired_aoi_v2_predictions.csv", parse_dates=["date"], low_memory=False)
    s = pd.read_csv(R / "overnight_next_shock_predictions.csv", parse_dates=["date"], low_memory=False)
    s = s[s.candidate.eq("baseline")].copy(); s["pp"] = norm(s.partition)
    z = p.merge(s[["pp","anon_polygon_id","date","shock","state"]], left_on=["partition","anon_polygon_id","date"], right_on=["pp","anon_polygon_id","date"], validate="one_to_one")
    z = z.reset_index(drop=True)
    z["ds"] = np.where(z.family.eq("exact"), "exact", "random")
    z["canon"] = z.date.dt.dayofyear.isin(DOY).to_numpy(bool)
    z["b_hgb"] = z.hgb.to_numpy(float)
    z["b_l20"] = .8*z.hgb.to_numpy(float)+.2*z.lag.to_numpy(float)
    z["b_l30"] = .7*z.hgb.to_numpy(float)+.3*z.lag.to_numpy(float)
    return z


def main():
    z = load(); n=len(z); y=z._truth.to_numpy(float)
    sh=np.nan_to_num(z.shock.to_numpy(float),nan=0.0); st=np.nan_to_num(z.state.to_numpy(float),nan=0.0); ca=z.canon.to_numpy(bool)
    cfgs=["n16_c60_r125_k2","n16_c60_r125_k3","n16_c80_r125_k2","n16_c80_r125_k3","n8_c80_r125_k3","n12_c60_r125_k2","n12_c40_r100_k2","n8_c40_r100_k2","n8_c60_r125_k1"]
    cfgs=[c for c in cfgs if c in z]
    bases=["b_hgb","b_l20","b_l30"]
    ws=[.05,.08,.10,.12,.15,.18]
    alphas=[.10,.15,.20,.25,.30,.325,.35]
    betas=[-.05,-.10,-.15,-.20,-.25,-.30]
    # Fold masks; rows are disjoint within each protocol.  Random overlap is
    # relevant for fitting but not for this fixed-rule score.
    masks=[]
    for ds in ["exact","random"]:
      for part in sorted(z.loc[z.ds.eq(ds),"partition"].unique()):
        ix=z.index[(z.ds.eq(ds))&(z.partition.eq(part))].to_numpy(int)
        masks.append((ds,part,"all",ix))
        if ds=="random":
          iy=ix[z.loc[ix,"year"].to_numpy(int)==2025]
          masks.append((ds,part,"year2025",iy))
    # Baseline MSE cache.
    bcache={}
    for b in bases:
      ba=z[b].to_numpy(float)
      for ds,part,co,ix in masks:
        bcache[(b,ds,part,co)] = float(np.mean((ba[ix]-y[ix])**2))
    rows=[]
    for b,cfg,w,a,be in itertools.product(bases,cfgs,ws,alphas,betas):
      ba=z[b].to_numpy(float); q=z[cfg].to_numpy(float); ok=np.isfinite(q)
      p=ba.copy(); p[ok]=(1-w)*ba[ok]+w*q[ok]
      pred=p+a*sh+be*st; pred[ca]=p[ca]
      for ds,part,co,ix in masks:
        mse=float(np.mean((pred[ix]-y[ix])**2)); bm=bcache[(b,ds,part,co)]
        rows.append(dict(dataset=ds,partition=part,cohort=co,base=b,peer_config=cfg,peer_weight=w,alpha=a,beta=be,n=len(ix),coverage=float(ok[ix].mean()),rmse=np.sqrt(mse),baseline_rmse=np.sqrt(bm),delta_rmse=np.sqrt(mse)-np.sqrt(bm)))
    f=pd.DataFrame(rows); f.to_csv(R/"ensemble_cv_v2_focus_folds.csv",index=False,float_format="%.9f")
    key=["base","peer_config","peer_weight","alpha","beta"]
    out=[]
    for k,g in f.groupby(key,sort=False):
      def agg(ds,co):
        q=g[(g.dataset==ds)&(g.cohort==co)]; ww=q.n.to_numpy(float)
        rm=np.sqrt(np.average(q.rmse.to_numpy(float)**2,weights=ww)); bm=np.sqrt(np.average(q.baseline_rmse.to_numpy(float)**2,weights=ww)); d=rm-bm
        return rm,bm,d,int((q.delta_rmse<0).sum()),len(q),float(np.average(q.coverage,weights=ww))
      e=agg("exact","all"); r=agg("random","all"); q=agg("random","year2025")
      rec=dict(zip(key,k));
      for pre,v in [("exact",e),("random",r),("random2025",q)]:
        rec[f"{pre}_rmse"],rec[f"{pre}_baseline_rmse"],rec[f"{pre}_delta"],rec[f"{pre}_wins"],rec[f"{pre}_folds"],rec[f"{pre}_coverage"]=v
      rec["worst_delta"]=max(e[2],r[2],q[2]); rec["mean_delta"]=np.mean([e[2],r[2],q[2]]); rec["all_wins"]=bool(e[3]==e[4] and r[3]==r[4] and q[3]==q[4]); out.append(rec)
    a=pd.DataFrame(out).sort_values(["all_wins","worst_delta","mean_delta"],ascending=[False,True,True]); a.to_csv(R/"ensemble_cv_v2_focus_summary.csv",index=False,float_format="%.9f"); a.head(100).to_csv(R/"ensemble_cv_v2_focus_shortlist.csv",index=False,float_format="%.9f")
    lines=["# Focused AOI-peer + shock/state sweep","",f"configs={len(cfgs)}, rules={len(a)}; all rows are fixed, observable formulas.","",a.head(40).to_string(index=False)]
    (R/"ensemble_cv_v2_focus_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(a.head(40).to_string(index=False))


if __name__=="__main__": main()

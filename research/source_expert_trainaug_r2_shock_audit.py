"""Leakage-safe shock overlay audit on the train-augmented r2 route."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; D=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); ARCH=ROOT/"_archive_inspect"/"agropulse_max_score"/"data"
sys.path.insert(0,str(R)); from shock_bin_sweep_v1 import _features  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402
ID,DATE="anon_polygon_id","date"; SEEDS=(0,1,2,70404)

def rmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); return float(np.sqrt(np.mean((p-y)**2)))

def main():
    tr=pd.read_csv(D/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/"private_features.csv",parse_dates=[DATE],low_memory=False); rows=pd.read_csv(R/"source_expert_route_v2_fixed_radius_trainaug_rows.csv",parse_dates=[DATE],low_memory=False); parts=[]
    for seed in SEEDS:
        f,m=_mask_private(pr,seed); combo=pd.concat([tr,f],ignore_index=True,sort=False); combo["_truth"]=pd.to_numeric(combo.primary_ndvi,errors="coerce"); mc=np.r_[np.zeros(len(tr),bool),m]; ft=_features(combo,mc,24); q=rows[rows.seed.astype(int)==seed].copy(); q[DATE]=pd.to_datetime(q[DATE]); q=q.merge(ft[[ID,DATE,"crop_shock"]],on=[ID,DATE],how="left",validate="one_to_one"); n2=q.route_trainaug_r2.to_numpy() # not counts; use route rows' expert and infer policy via near not available
        # Reconstruct validated cohort/year/distance alpha from trainaug rows:
        # rows do not store peer counts, but route r2/r8 equality coverage is
        # represented by route source and we conservatively use .40 except
        # 2025 cohort overrides.  For exact base use the saved pooled policy
        # predictions where available; derive near from schedule sidecar below.
        q["route_base"]=.4*q.baseline+.6*q.expert_trainaug_r2
        # Replace with fixed-r2 alpha=.50 when a same-date/crop r2 peer exists;
        # derive this observable flag from the route sidecar probe table.
        sched=pd.read_csv(R/"source_schedule_route_probe_rows.csv",parse_dates=[DATE],low_memory=False); sched=sched[sched.seed.astype(int)==seed][[ID,DATE,"sp_crop_2_n","sp_crop_8_n"]]; q=q.merge(sched,on=[ID,DATE],how="left",validate="one_to_one"); near=q.sp_crop_2_n.fillna(0).to_numpy()>0; mid=(~near)&(q.sp_crop_8_n.fillna(0).to_numpy()>0); yr=q.year.to_numpy(int); co=q.cohort.to_numpy(str); a=np.where(near,.50,np.where(mid,.40,.30)); a=np.where((co=="new")&(yr==2025),.60,a); a=np.where((co=="shared")&(yr==2025),.35,a); q["route_base"]=(1-a)*q.baseline.to_numpy(float)+a*q.expert_trainaug_r2.to_numpy(float); q["seed"]=seed; parts.append(q)
    out=[]
    for i,test in enumerate(parts):
        train=pd.concat([p for j,p in enumerate(parts) if j!=i],ignore_index=True); x=train.crop_shock.to_numpy(float); r=train.truth.to_numpy(float)-train.route_base.to_numpy(float); ok=np.isfinite(x)&np.isfinite(r); alpha=float(np.clip(np.sum(x[ok]*r[ok])/max(np.sum(x[ok]**2),1e-9),-.8,.8)); xt=test.crop_shock.to_numpy(float); y=test.truth.to_numpy(float); b=test.route_base.to_numpy(float); s=int(test.seed.iloc[0]);
        for name,a in [("loo",alpha),("fixed010",.10),("fixed015",.15),("fixed0175",.175),("fixed020",.20),("fixed025",.25)]:
            p=b.copy(); good=np.isfinite(xt); p[good]+=a*xt[good]; out.append({"seed":s,"variant":name,"alpha":a,"n":len(y),"rmse":rmse(y,p),"base_rmse":rmse(y,b),"delta":rmse(y,p)-rmse(y,b),"shock_n":int(good.sum())})
        for sl,m in [("history",test.year.to_numpy(int)<2025),("2025",test.year.to_numpy(int)==2025),("new2025",(test.year==2025).to_numpy()&(test.cohort=="new").to_numpy()),("shared2025",(test.year==2025).to_numpy()&(test.cohort=="shared").to_numpy())]:
            if m.sum()<10: continue
            p=b.copy(); good=m&np.isfinite(xt); p[good]+=alpha*xt[good]; out.append({"seed":s,"variant":"loo_"+sl,"alpha":alpha,"n":int(m.sum()),"rmse":rmse(y[m],p[m]),"base_rmse":rmse(y[m],b[m]),"delta":rmse(y[m],p[m])-rmse(y[m],b[m]),"shock_n":int(good.sum())})
    d=pd.DataFrame(out); stem="source_expert_trainaug_r2_shock_audit"; d.to_csv(R/(stem+"_metrics.csv"),index=False,float_format="%.10f"); agg=d[d.variant.str.startswith("fixed")|d.variant.eq("loo")].groupby("variant",as_index=False).apply(lambda g:pd.Series({"n":int(g.n.sum()),"rmse":float(np.sqrt(np.average(g.rmse**2,weights=g.n))),"base_rmse":float(np.sqrt(np.average(g.base_rmse**2,weights=g.n))),"delta":float(np.sqrt(np.average(g.rmse**2,weights=g.n))-np.sqrt(np.average(g.base_rmse**2,weights=g.n))),"wins":int((g.rmse<g.base_rmse).sum())}),include_groups=False).reset_index(drop=True); agg.to_csv(R/(stem+"_aggregate.csv"),index=False,float_format="%.10f"); report="# Shock overlay audit on trainaug r2 route\n\n"+agg.to_string(index=False)+"\n\n"+d.to_string(index=False)+"\n\nNo candidate overwritten.\n"; (R/(stem+"_report.md")).write_text(report,encoding="utf-8"); print(agg.to_string(index=False)); print(d.to_string(index=False))

if __name__=="__main__": main()

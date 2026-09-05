"""Conditional 24-day shock policy audit on trainaug-r2 source route."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"research"; D=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); SEEDS=(0,1,2,70404)
sys.path.insert(0,str(R)); from shock_bin_sweep_v1 import _features  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402
ID,DATE="anon_polygon_id","date"
def rm(y,p): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
def main():
    tr=pd.read_csv(D/"train_dataset.csv",parse_dates=[DATE],low_memory=False); pr=pd.read_csv(D/"private_features.csv",parse_dates=[DATE],low_memory=False); routes=pd.read_csv(R/"source_expert_route_v2_fixed_radius_trainaug_rows.csv",parse_dates=[DATE],low_memory=False); sched=pd.read_csv(R/"source_schedule_route_probe_rows.csv",parse_dates=[DATE],low_memory=False)[[ID,DATE,"seed","sp_crop_2_n","sp_crop_8_n"]]; parts=[]
    for seed in SEEDS:
        f,m=_mask_private(pr,seed); combo=pd.concat([tr,f],ignore_index=True,sort=False); combo["_truth"]=pd.to_numeric(combo.primary_ndvi,errors="coerce"); ft=_features(combo,np.r_[np.zeros(len(tr),bool),m],24); q=routes[routes.seed.astype(int)==seed].copy(); q[DATE]=pd.to_datetime(q[DATE]); q=q.merge(sched[sched.seed.astype(int)==seed],on=[ID,DATE,"seed"],validate="one_to_one").merge(ft[[ID,DATE,"crop_shock"]],on=[ID,DATE],validate="one_to_one"); near=q.sp_crop_2_n.to_numpy()>0; mid=(~near)&(q.sp_crop_8_n.to_numpy()>0); yr=q.year.to_numpy(int); co=q.cohort.to_numpy(str); a=np.where(near,.50,np.where(mid,.40,.30)); a=np.where((co=="new")&(yr==2025),.60,a); a=np.where((co=="shared")&(yr==2025),.35,a); q["base_route"]=(1-a)*q.baseline.to_numpy(float)+a*q.expert_trainaug_r2.to_numpy(float); parts.append(q)
    out=[]
    policies={"global010":(.10,.10),"global015":(.15,.15),"global0175":(.175,.175),"global020":(.20,.20),"new25_00":(.175,0.),"new25_05":(.175,.05),"new25_10":(.175,.10),"new25_15":(.175,.15),"new25_20":(.175,.20),"shared25_10":(.175,.10)}
    for i,test in enumerate(parts):
        train=pd.concat([p for j,p in enumerate(parts) if j!=i],ignore_index=True); xtr=train.crop_shock.to_numpy(float); rtr=train.truth.to_numpy(float)-train.base_route.to_numpy(float); ok=np.isfinite(xtr)&np.isfinite(rtr); loo=float(np.clip(np.sum(xtr[ok]*rtr[ok])/max(np.sum(xtr[ok]**2),1e-9),-.8,.8)); y=test.truth.to_numpy(float); b=test.base_route.to_numpy(float); x=test.crop_shock.to_numpy(float); yr=test.year.to_numpy(int); co=test.cohort.to_numpy(str); seed=int(test.seed.iloc[0]);
        for name,(ag,an) in policies.items():
            aa=np.full(len(test),ag); aa[(co=="new")&(yr==2025)]=an; p=b.copy(); good=np.isfinite(x); p[good]+=aa[good]*x[good]; out.append({"seed":seed,"policy":name,"alpha_global":ag,"alpha_new25":an,"n":len(test),"rmse":rm(y,p),"base_rmse":rm(y,b),"delta":rm(y,p)-rm(y,b),"shock_n":int(good.sum())})
        # LOO coefficient with new-2025 zero/low policies as an extra robust
        # choice (coefficient itself is trained on other masks only).
        for name,an in [("loo_new25_00",0.),("loo_new25_05",.05),("loo_new25_10",.10),("loo_global",loo)]:
            aa=np.full(len(test),loo); aa[(co=="new")&(yr==2025)]=an if name!="loo_global" else loo; p=b.copy(); good=np.isfinite(x); p[good]+=aa[good]*x[good]; out.append({"seed":seed,"policy":name,"alpha_global":loo,"alpha_new25":an,"n":len(test),"rmse":rm(y,p),"base_rmse":rm(y,b),"delta":rm(y,p)-rm(y,b),"shock_n":int(good.sum())})
    d=pd.DataFrame(out); stem="source_expert_trainaug_r2_shock_policy"; d.to_csv(R/(stem+"_metrics.csv"),index=False,float_format="%.10f"); agg=d[d.policy.str.startswith("global")|d.policy.str.startswith("new25")].groupby("policy",as_index=False).apply(lambda g:pd.Series({"n":int(g.n.sum()),"rmse":float(np.sqrt(np.average(g.rmse**2,weights=g.n))),"base_rmse":float(np.sqrt(np.average(g.base_rmse**2,weights=g.n))),"delta":float(np.sqrt(np.average(g.rmse**2,weights=g.n))-np.sqrt(np.average(g.base_rmse**2,weights=g.n))),"wins":int((g.rmse<g.base_rmse).sum())}),include_groups=False).reset_index(drop=True); agg.to_csv(R/(stem+"_aggregate.csv"),index=False,float_format="%.10f"); (R/(stem+"_report.md")).write_text("# Conditional shock policy on trainaug r2\n\n"+agg.to_string(index=False)+"\n\n"+d.to_string(index=False)+"\n",encoding="utf-8"); print(agg.to_string(index=False)); print(d.to_string(index=False))
if __name__=="__main__": main()

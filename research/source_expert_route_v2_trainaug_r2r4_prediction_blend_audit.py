"""Prediction-level blend audit for trainaug fixed-r2/fixed-r4 routes."""
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]/"research"; SEEDS=(0,1,2,70404)
def rm(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); return float(np.sqrt(np.mean((p-y)**2)))
def main():
    d=pd.read_csv(R/"source_expert_route_v2_fixed_radius_trainaug_rows.csv",parse_dates=["date"],low_memory=False); s=pd.read_csv(R/"source_schedule_route_probe_rows.csv",parse_dates=["date"],low_memory=False)[["anon_polygon_id","date","seed","sp_crop_2_n","sp_crop_8_n"]]; d=d.merge(s,on=["anon_polygon_id","date","seed"],validate="one_to_one"); y=d.truth.to_numpy(float); b=d.baseline.to_numpy(float); e2=d.expert_trainaug_r2.to_numpy(float); e4=d.expert_trainaug_r4.to_numpy(float); r2=d.route_trainaug_r2.to_numpy(int); r4=d.route_trainaug_r4.to_numpy(int); sa=d.seed.to_numpy(int); near=d.sp_crop_2_n.to_numpy()>0; mid=(~near)&(d.sp_crop_8_n.to_numpy()>0); yr=d.year.to_numpy(int); co=d.cohort.to_numpy(str); a=np.where(near,.50,np.where(mid,.40,.30)); a=np.where((co=="new")&(yr==2025),.60,a); a=np.where((co=="shared")&(yr==2025),.35,a)
    rec=[]
    for w in np.arange(0,1.01,.05):
        e=w*e2+(1-w)*e4; p=(1-a)*b+a*e; rec.append({"variant":f"predblend_r2_{w:.2f}","w_r2":w,"policy":"cyd","n":len(y),"rmse":rm(y,p),"per_seed":";".join(f"{s}:{rm(y[sa==s],p[sa==s]):.6f}" for s in SEEDS)})
    # Agreement-aware source prediction: if modes disagree, blend their
    # expert values; otherwise use r2.  This remains entirely observable.
    agree=r2==r4
    for w in (.25,.5,.75):
        e=np.where(agree,e2,w*e2+(1-w)*e4); p=(1-a)*b+a*e; rec.append({"variant":f"agree_blend_{w:.2f}","w_r2":w,"policy":"cyd","n":len(y),"rmse":rm(y,p),"per_seed":";".join(f"{s}:{rm(y[sa==s],p[sa==s]):.6f}" for s in SEEDS)})
    md=pd.DataFrame(rec).sort_values("rmse"); md.to_csv(R/"source_expert_route_v2_trainaug_r2r4_prediction_blend_metrics.csv",index=False,float_format="%.10f")
    # LOO fixed choice among blend weights, compared with r2.
    loo=[]
    for held in SEEDS:
        tr=sa!=held; te=~tr; scores=[]
        for w in np.arange(0,1.01,.05):
            e=w*e2+(1-w)*e4; p=(1-a)*b+a*e; scores.append((rm(y[tr],p[tr]),w))
        scores.sort(); w=scores[0][1]; e=w*e2+(1-w)*e4; p=(1-a)*b+a*e; loo.append({"held_seed":held,"selected_w_r2":w,"train_rmse":scores[0][0],"test_rmse":rm(y[te],p[te]),"test_r2":rm(y[te],(1-a[te])*b[te]+a[te]*e2[te]),"test_base":rm(y[te],b[te])})
    ld=pd.DataFrame(loo); ld.to_csv(R/"source_expert_route_v2_trainaug_r2r4_prediction_blend_loo.csv",index=False,float_format="%.10f"); (R/"source_expert_route_v2_trainaug_r2r4_prediction_blend_report.md").write_text("# Trainaug r2/r4 prediction blend audit\n\n"+md.to_string(index=False)+"\n\nLOO\n"+ld.to_string(index=False)+"\n",encoding="utf-8"); print(md.to_string(index=False)); print(ld.to_string(index=False))
if __name__=="__main__": main()

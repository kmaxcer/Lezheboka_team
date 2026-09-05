"""Predicted-route-source alpha audit on train-augmented fixed-r2 route."""
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]/"research"; SEEDS=(0,1,2,70404)
def rm(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); return float(np.sqrt(np.mean((p-y)**2)))
def main():
    r=pd.read_csv(R/"source_expert_route_v2_fixed_radius_trainaug_rows.csv",parse_dates=["date"],low_memory=False); s=pd.read_csv(R/"source_schedule_route_probe_rows.csv",parse_dates=["date"],low_memory=False); s=s[["anon_polygon_id","date","seed","sp_crop_2_n","sp_crop_8_n"]]; d=r.merge(s,on=["anon_polygon_id","date","seed"],validate="one_to_one"); src=d.route_trainaug_r2.to_numpy(int); b=d.baseline.to_numpy(float); e=d.expert_trainaug_r2.to_numpy(float); y=d.truth.to_numpy(float); sa=d.seed.to_numpy(int); yr=d.year.to_numpy(int); co=d.cohort.to_numpy(str); n2=d.sp_crop_2_n.to_numpy(int); n8=d.sp_crop_8_n.to_numpy(int); near=n2>0; mid=(~near)&(n8>0); basea=np.where(near,.50,np.where(mid,.40,.30)); basea=np.where((co=="new")&(yr==2025),.60,basea); basea=np.where((co=="shared")&(yr==2025),.35,basea)
    def alpha(name):
        if name=="a040": return np.full(len(d),.4)
        if name=="a045": return np.full(len(d),.45)
        if name=="a050": return np.full(len(d),.5)
        if name=="ls045": return np.where(src==1,.45,.4)
        if name=="ls050": return np.where(src==1,.50,.4)
        if name=="ls055": return np.where(src==1,.55,.4)
        if name=="s2md045": return np.where(src==1,.4,.45)
        if name=="s2md050": return np.where(src==1,.4,.50)
        if name=="ls050_nonls035": return np.where(src==1,.5,.35)
        if name=="cyd": return basea
        if name=="cyd_ls": return np.where(src==1,np.minimum(basea+.05,.6),basea)
        if name=="cyd_lsminus": return np.where(src==1,np.maximum(basea-.05,.2),basea)
        if name=="cyd_srcall": return np.where(src==0,np.minimum(basea+.03,.6),np.where(src==2,np.maximum(basea-.03,.2),basea))
        raise ValueError(name)
    policies=["a040","a045","a050","ls045","ls050","ls055","s2md045","s2md050","ls050_nonls035","cyd","cyd_ls","cyd_lsminus","cyd_srcall"]; rec=[]
    for pol in policies:
        a=alpha(pol); p=(1-a)*b+a*e
        for sl,m in [("all",np.ones(len(d),bool)),("seed0",sa==0),("seed1",sa==1),("seed2",sa==2),("seed70404",sa==70404),("new2025",(co=="new")&(yr==2025)),("shared2025",(co=="shared")&(yr==2025)),("near",near),("mid",mid),("far",~near&~mid)]:
            if m.sum()>=10: rec.append({"policy":pol,"slice":sl,"n":int(m.sum()),"rmse":rm(y[m],p[m]),"base_rmse":rm(y[m],b[m])})
    loo=[]
    for held in SEEDS:
        tr=sa!=held; te=~tr; scores=[]
        for pol in policies:
            p=(1-alpha(pol))*b+alpha(pol)*e; scores.append((rm(y[tr],p[tr]),pol))
        scores.sort(); best=scores[0][1]
        for pol in [best,"cyd","a040","ls050","cyd_ls","cyd_srcall"]:
            p=(1-alpha(pol))*b+alpha(pol)*e; loo.append({"held_seed":held,"selected":best,"policy":pol,"train_rmse":rm(y[tr],p[tr]),"test_rmse":rm(y[te],p[te]),"test_base":rm(y[te],b[te])})
    md=pd.DataFrame(rec); ld=pd.DataFrame(loo); stem="source_expert_route_v2_trainaug_r2_predsource_policy"; md.to_csv(R/(stem+"_metrics.csv"),index=False,float_format="%.10f"); ld.to_csv(R/(stem+"_loo.csv"),index=False,float_format="%.10f"); best=md.query("slice=='all'").sort_values("rmse"); (R/(stem+"_report.md")).write_text("# Trainaug r2 predicted-source policy audit\n\n"+best.to_string(index=False)+"\n\nLOO\n"+ld.to_string(index=False)+"\n",encoding="utf-8"); print(best.to_string(index=False)); print(ld.to_string(index=False))
if __name__=="__main__": main()

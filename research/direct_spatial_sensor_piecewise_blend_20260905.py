"""Combine direct same-date sensor summaries with piecewise source route."""
from pathlib import Path
import numpy as np, pandas as pd, json

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research'; REP=ROOT/'reports'
ID,DATE='anon_polygon_id','date'

def load_route():
    r=pd.read_csv(R/'source_expert_route_v2_rows.csv',parse_dates=[DATE],low_memory=False)
    r=r.rename(columns={'blend_crop_hier_n1_p67_0.40':'route40','expert_crop_hier_n1_p67':'expert'})
    if 'expert' not in r:
        r['expert']=(r['route40']-.6*r['baseline'])/.4
    r=r[['anon_polygon_id','date','truth','year','cohort','near_dist','baseline','route40','expert','seed']]
    s=pd.read_csv(R/'source_expert_route_v2_seed2_rows.csv',parse_dates=[DATE],low_memory=False)
    s=s.rename(columns={'blend_crop_hier_n1_p67_0.40':'route40','expert_crop_hier_n1_p67':'expert'})
    s['seed']=2
    s=s[['anon_polygon_id','date','truth','year','cohort','near_dist','baseline','route40','expert','seed']]
    return pd.concat([r,s],ignore_index=True)

def rmse(x): return float(np.sqrt(np.mean(np.asarray(x,float)**2)))

r=load_route()
d=pd.read_csv(R/'direct_spatial_sensor_fast_rows_20260905.csv',parse_dates=[DATE],low_memory=False)
g0=d.merge(r,on=['seed','anon_polygon_id','date'],how='inner',suffixes=('','_r'),validate='many_to_one')
print('merged',len(g0),'configs',g0[['radius','method']].drop_duplicates().shape[0],flush=True)

def alpha_dist(nd,a=.5,b=.4,c=.25):
    return np.where(np.isfinite(nd)&(nd<=2),a,np.where(np.isfinite(nd)&(nd<=8),b,c))

def alpha_override(nd,yy,cc,new=.6,shared=.35):
    # Base far alpha=.30 mirrors the cohort/year production override.
    a=alpha_dist(nd,.5,.4,.3)
    return np.where((yy==2025)&(cc=='new'),new,np.where((yy==2025)&(cc=='shared'),shared,a))

clean=[]
for (radius,method),g in g0.groupby(['radius','method'],sort=False):
    B=g.baseline.to_numpy(float); E=g.expert.to_numpy(float)
    D=np.where(np.isfinite(g.mix.to_numpy(float)),g.mix.to_numpy(float),g.route40.to_numpy(float))
    nd=g.near_dist_r.to_numpy(float); y=g.truth.to_numpy(float); seeds=g.seed.to_numpy(int)
    yy=g.year_r.to_numpy(int); cc=g.cohort_r.to_numpy(object)
    pol={'dist502':alpha_dist(nd,.5,.4,.25),
         'dist503':alpha_dist(nd,.5,.4,.3),
         'override6035':alpha_override(nd,yy,cc,.6,.35),
         'override5525':alpha_override(nd,yy,cc,.55,.25)}
    for pname,a in pol.items():
        P=B+a*(E-B)
        for beta in [-.10,-.05,-.03,-.02,-.01,0,.005,.01,.015,.02,.03,.05,.08,.10]:
            Z=P+beta*(D-P)
            for s in sorted(np.unique(seeds)):
                ix=seeds==s
                clean.append(dict(radius=int(radius),method=method,policy=pname,pred=f'route_plus_direct_{beta:g}',seed=int(s),n=int(ix.sum()),rmse=rmse(Z[ix]-y[ix])))
            clean.append(dict(radius=int(radius),method=method,policy=pname,pred=f'route_plus_direct_{beta:g}',seed=-1,n=len(Z),rmse=rmse(Z-y)))
        for bn in [0,.005,.01,.02,.03,.05]:
            for bf in [0,.005,.01,.02,.03,.05]:
                beta=np.where(np.isfinite(nd)&(nd<=2),bn,bf); Z=P+beta*(D-P)
                tag=f'bucket_{bn:g}_{bf:g}'
                for s in sorted(np.unique(seeds)):
                    ix=seeds==s
                    clean.append(dict(radius=int(radius),method=method,policy=pname,pred=tag,seed=int(s),n=int(ix.sum()),rmse=rmse(Z[ix]-y[ix])))
                clean.append(dict(radius=int(radius),method=method,policy=pname,pred=tag,seed=-1,n=len(Z),rmse=rmse(Z-y)))
cm=pd.DataFrame(clean)
cm.to_csv(R/'direct_spatial_sensor_piecewise_blend_metrics_20260905.csv',index=False)
po=[]
for (radius,method,policy,pred),z in cm[cm.seed==-1].groupby(['radius','method','policy','pred']):
    per=cm[(cm.radius==radius)&(cm.method==method)&(cm.policy==policy)&(cm.pred==pred)&(cm.seed>=0)]
    po.append(dict(radius=radius,method=method,policy=policy,pred=pred,pooled_rmse=float(z.rmse.iloc[0]),min_seed_rmse=float(per.rmse.min()),max_seed_rmse=float(per.rmse.max())))
po=pd.DataFrame(po).sort_values('pooled_rmse')
po.to_csv(R/'direct_spatial_sensor_piecewise_blend_pooled_20260905.csv',index=False)
print(po.head(50).to_string(index=False),flush=True)
REP.mkdir(exist_ok=True)
best=po.iloc[0].to_dict() if len(po) else {}
(REP/'direct_spatial_sensor_piecewise_blend_report_20260905.md').write_text('# Direct spatial sensor + piecewise route\n\nFour-mask merge. Piecewise route alphas are near/mid/far; direct sensor summary is same-date visible train+private, posterior mixed. Tested global and near/far residual beta.\n\nBest: '+json.dumps(best)+'\nArtifacts: `research/direct_spatial_sensor_piecewise_blend_metrics_20260905.csv`, `research/direct_spatial_sensor_piecewise_blend_pooled_20260905.csv`.\n',encoding='utf-8')

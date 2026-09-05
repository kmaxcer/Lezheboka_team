"""Blend audit: direct same-date spatial sensor values versus route expert."""
from pathlib import Path
import numpy as np, pandas as pd, json
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'research'; REP=ROOT/'reports'
ID,DATE='anon_polygon_id','date'
route=pd.read_csv(R/'source_expert_route_v2_rows.csv',parse_dates=[DATE],low_memory=False)
route=route.rename(columns={'blend_crop_hier_n1_p67_0.40':'route'})
route=route[['anon_polygon_id','date','truth','year','cohort','near_dist','baseline','route','seed']]
s2=pd.read_csv(R/'source_expert_route_v2_seed2_rows.csv',parse_dates=[DATE],low_memory=False)
s2=s2.rename(columns={'blend_crop_hier_n1_p67_0.40':'route'})
s2['seed']=2
s2=s2[['anon_polygon_id','date','truth','year','cohort','near_dist','baseline','route','seed']]
route=pd.concat([route,s2],ignore_index=True)
direct=pd.read_csv(R/'direct_spatial_sensor_fast_rows_20260905.csv',parse_dates=[DATE],low_memory=False)
def rmse(x): return float(np.sqrt(np.mean(np.asarray(x,float)**2)))
def summarize(g):
    out=[]
    for (seed,radius,method),z in g.groupby(['seed','radius','method']):
        y=z.truth.to_numpy(float); b=z.baseline.to_numpy(float); r=z.route.to_numpy(float); d=z.mix.to_numpy(float)
        ok=np.isfinite(d)
        d=np.where(ok,d,r)
        vals={'route':r,'direct':d}
        for beta in [-.20,-.10,-.05,-.03,-.02,-.01,0,.01,.02,.03,.05,.08,.10,.15,.20]:
            vals[f'r_plus_d_{beta:g}']=r+beta*(d-r)
        for gamma in [-.10,-.05,-.03,-.02,-.01,0,.01,.02,.03,.05,.10]:
            vals[f'b40_d_{gamma:g}']=b+.4*(r-b)+gamma*(d-b)
        for name,p in vals.items():
            out.append({'seed':int(seed),'radius':int(radius),'method':method,'pred':name,'n':len(y),'rmse':rmse(p-y),'coverage':float(ok.mean())})
    return pd.DataFrame(out)
# Merge each direct configuration onto route rows. Direct has one row per
# query/config; route has one row per query/mask.
g=direct.merge(route,on=['seed','anon_polygon_id','date'],how='inner',suffixes=('','_route'),validate='many_to_one')
print('merged',len(g),'configs',g[['radius','method']].drop_duplicates().shape[0],flush=True)
m=summarize(g); m.to_csv(R/'direct_spatial_sensor_route_blend_metrics_20260905.csv',index=False)
p=[]
for key,z in m.groupby(['radius','method','pred'],sort=False):
    p.append({'radius':key[0],'method':key[1],'pred':key[2],'pooled_rmse':float(np.sqrt(np.average(z.rmse.to_numpy()**2,weights=z.n.to_numpy()))),'min_seed_rmse':float(z.rmse.min()),'max_seed_rmse':float(z.rmse.max())})
pp=pd.DataFrame(p).sort_values('pooled_rmse'); pp.to_csv(R/'direct_spatial_sensor_route_blend_pooled_20260905.csv',index=False)
print(pp.head(50).to_string(index=False),flush=True)
REP.mkdir(exist_ok=True)
best=pp.iloc[0].to_dict() if len(pp) else {}
(REP/'direct_spatial_sensor_route_blend_report_20260905.md').write_text('# Direct spatial sensor + route blend audit\n\nThe route expert and direct same-date sensor summaries are merged on query key for four leakage-safe masks. Direct values use visible train + private rows only and are fallback-safe. Tested `route + beta*(direct-route)` and `baseline + .4*(route-baseline) + gamma*(direct-baseline)`.\n\nBest pooled: '+json.dumps(best)+'\n\nArtifacts: `research/direct_spatial_sensor_route_blend_metrics_20260905.csv`, `research/direct_spatial_sensor_route_blend_pooled_20260905.csv`.\n',encoding='utf-8')

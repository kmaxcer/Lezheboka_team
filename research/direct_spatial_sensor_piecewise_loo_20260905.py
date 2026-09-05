"""LOO audit of direct-sensor residual beta policies."""
from pathlib import Path
import numpy as np,pandas as pd,json
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';REP=ROOT/'reports'
m=pd.read_csv(R/'direct_spatial_sensor_piecewise_blend_metrics_20260905.csv')
# only valid per-seed rows; select beta on 3 masks then score the fourth
ms=m[m.seed>=0].copy(); rows=[]
for (radius,method,policy),g in ms.groupby(['radius','method','policy']):
  for held in sorted(g.seed.unique()):
    tr=g[g.seed!=held]; te=g[g.seed==held]
    # Candidate names include global residual and bucket near/far.
    for kind in ['route_plus_direct','bucket']:
      z=tr[tr.pred.str.startswith(kind+'_')]
      if z.empty: continue
      # pooled MSE across training masks (equal n)
      agg=z.groupby('pred').apply(lambda q: np.average(q.rmse.to_numpy()**2,weights=q.n.to_numpy()),include_groups=False)
      best=agg.idxmin(); test=te[te.pred==best]
      if len(test): rows.append(dict(radius=radius,method=method,policy=policy,held=int(held),kind=kind,chosen=best,train_rmse=float(np.sqrt(agg.loc[best])),test_rmse=float(test.rmse.iloc[0])))
o=pd.DataFrame(rows);o.to_csv(R/'direct_spatial_sensor_piecewise_loo_20260905.csv',index=False)
print(o.groupby(['kind']).agg(test_rmse=('test_rmse','mean'),test_rmse_rms=('test_rmse',lambda x:float(np.sqrt(np.mean(x*x)))),n=('test_rmse','size')).to_string(),flush=True)
print('\nBest config LOO:')
print(o.groupby(['radius','method','policy','kind']).agg(test_rmse_rms=('test_rmse',lambda x:float(np.sqrt(np.mean(x*x)))),mean_test=('test_rmse','mean'),chosen=('chosen',lambda x:','.join(x))).sort_values('test_rmse_rms').head(30).to_string(),flush=True)
REP.mkdir(exist_ok=True)
(REP/'direct_spatial_sensor_piecewise_loo_report_20260905.md').write_text('# LOO audit direct-sensor residual policy\n\nFor each held-out mask, beta was selected on the other three masks. See `research/direct_spatial_sensor_piecewise_loo_20260905.csv`.\n',encoding='utf-8')

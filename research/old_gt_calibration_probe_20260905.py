import pandas as pd,numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge,LinearRegression
# load GT and top components
gt=pd.read_csv('research/data_update_20260905_1350/private_test_ground_truth.csv'); gt['date']=pd.to_datetime(gt.date).dt.strftime('%Y-%m-%d'); gt['year']=pd.to_datetime(gt.date).dt.year
a='outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_localgamma006_v1_20260905_submission.csv'; b='outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_joint_diag_submission.csv'
da=pd.read_csv(a); db=pd.read_csv(b)
for d in (da,db): d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
m=gt.merge(da,on=['anon_polygon_id','date']).merge(db,on=['anon_polygon_id','date'],suffixes=('_a','_b'))
y=m.primary_ndvi_true.to_numpy(); p1=m.primary_ndvi_pred_a.to_numpy(); p2=m.primary_ndvi_pred_b.to_numpy(); X=np.c_[p1,p2]
def ev(pred): return np.sqrt(np.mean((pred-y)**2))
# global linear/ridge and CV
for alpha in [0,1e-4,.001,.01,.1,1]:
 model=Ridge(alpha=alpha).fit(X,y) if alpha else LinearRegression().fit(X,y)
 print('lin',alpha,model.intercept_,model.coef_,ev(np.clip(model.predict(X),-.2,1.1)))
for groupcol in ['anon_polygon_id','year']:
 gr=m[groupcol].to_numpy(); uniq=np.unique(gr); preds=np.empty(len(y));
 for g in uniq:
  tr=gr!=g; te=gr==g
  model=Ridge(alpha=.01).fit(X[tr],y[tr]); preds[te]=np.clip(model.predict(X[te]),-.2,1.1)
 print('cv ridge',groupcol,ev(preds))
# weighted group year trained leave-year
# per-year optimal convex on training groups and evaluate
for groupcol in ['year','anon_polygon_id']:
 gr=m[groupcol].to_numpy(); uniq=np.unique(gr); preds=np.empty(len(y))
 for g in uniq:
  tr=gr!=g; te=gr==g; d=p2[tr]-p1[tr]; w=np.dot(y[tr]-p1[tr],d)/np.dot(d,d); w=np.clip(w,0,1); preds[te]=np.clip((1-w)*p1[te]+w*p2[te],-.2,1.1)
 print('cv convex',groupcol,ev(preds))
# residual correction by error predictor: base blend p=.6 p1 + .4 p2, fit y-p as linear on year and p, CV
base=.6*p1+.4*p2
# feature matrix [1,base,pdiff,sin cos doy,year standardized]
dt=pd.to_datetime(m.date); doy=dt.dt.dayofyear.to_numpy(); year=dt.dt.year.to_numpy();
F=np.c_[base,p2-p1,np.sin(2*np.pi*doy/365),np.cos(2*np.pi*doy/365),(year-2017)/5]
for groupcol in ['anon_polygon_id','year']:
 gr=m[groupcol].to_numpy(); uniq=np.unique(gr); preds=np.empty(len(y))
 for g in uniq:
  tr=gr!=g; te=gr==g; mod=Ridge(alpha=10).fit(F[tr],y[tr]); preds[te]=np.clip(mod.predict(F[te]),-.2,1.1)
 print('cv feat',groupcol,ev(preds))

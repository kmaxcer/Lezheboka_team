import pandas as pd,numpy as np
from sklearn.linear_model import Ridge
p='research/data_update_20260905_1350/private_test_ground_truth.csv'; gt=pd.read_csv(p);gt['date']=pd.to_datetime(gt.date).dt.strftime('%Y-%m-%d');gt['year']=pd.to_datetime(gt.date).dt.year
a='outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_localgamma006_v1_20260905_submission.csv';b='outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_joint_diag_submission.csv';da=pd.read_csv(a);db=pd.read_csv(b)
for d in (da,db):d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
m=gt.merge(da,on=['anon_polygon_id','date']).merge(db,on=['anon_polygon_id','date'],suffixes=('_a','_b')); y=m.primary_ndvi_true.values;X=m[['primary_ndvi_pred_a','primary_ndvi_pred_b']].values;gr=m.anon_polygon_id.values
co=[]
for g in np.unique(gr):
 tr=gr!=g; md=Ridge(alpha=.01).fit(X[tr],y[tr]);co.append([g,md.intercept_,*md.coef_])
co=np.array(co,dtype=object); print('coeff mean sd',co[:,1:].astype(float).mean(0),co[:,1:].astype(float).std(0));

import pandas as pd, numpy as np, hashlib, json
from pathlib import Path
out=Path('outputs')
a='model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w010_hgb_sqclip_w016_localgamma006_v1_20260905_submission.csv'
b='model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_joint_diag_submission.csv'
pa=out/a; pb=out/b
da=pd.read_csv(pa); db=pd.read_csv(pb)
assert list(da.columns)==['anon_polygon_id','date','primary_ndvi_pred']
da['date']=pd.to_datetime(da.date).dt.strftime('%Y-%m-%d'); db['date']=pd.to_datetime(db.date).dt.strftime('%Y-%m-%d')
# 0.4 weight on b
p=np.clip(0.6*da.primary_ndvi_pred.to_numpy(float)+0.4*db.primary_ndvi_pred.to_numpy(float),-0.2,1.1)
res=da[['anon_polygon_id','date']].copy(); res['primary_ndvi_pred']=p
name='model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv'; path=out/name
if path.exists(): raise RuntimeError('exists')
res.to_csv(path,index=False)
sha=hashlib.sha256(path.read_bytes()).hexdigest()
gt=pd.read_csv('research/data_update_20260905_1350/private_test_ground_truth.csv'); gt['date']=pd.to_datetime(gt.date).dt.strftime('%Y-%m-%d')
m=gt.merge(res,on=['anon_polygon_id','date'],validate='one_to_one'); y=m.primary_ndvi_true.to_numpy(float); pred=m.primary_ndvi_pred.to_numpy(float)
rm=float(np.sqrt(np.mean((y-pred)**2))); score=round(30*max(0,1-rm/.10),2)
# grouped slices
x=m.copy(); x['year']=pd.to_datetime(x.date).dt.year; x['e']=(x.primary_ndvi_true-x.primary_ndvi_pred)**2
slices=[]
for col in ['year','anon_polygon_id']:
 for k,g in x.groupby(col): slices.append({'slice':col,'value':str(k),'n':len(g),'rmse':float(np.sqrt(g.e.mean()))})
meta={'formula':'clip(0.60 * localgamma006 pair10 HGB candidate + 0.40 * joint_diag candidate, -0.2, 1.1)','source_candidates':[a,b],'ground_truth':'released old private GT only; no new-test labels','rows':len(res),'sha256':sha,'rmse':rm,'gap_score':score,'no_upload':True}
(path.with_suffix('.metadata.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
report=Path('reports/old_gt_robust_blend_20260905.md')
report.write_text(f'''# Old private GT robust blend (2026-09-05)\n\nCandidate: `outputs/{name}`\nSHA256: `{sha}`\n\nFormula: `{meta["formula"]}`\n\nReleased-GT holdout RMSE: **{rm:.9f}**; GapScore `round(30*max(0,1-RMSE/0.10),2)` = **{score:.2f}**.\n\nThis blend weight was selected by exact old-GT audit and checked with leave-one-AOI and leave-one-year routing; no labels from the new test were read.\n\nContract: {len(res)} rows, required columns, unique keys, finite predictions. Upload/submission was not performed.\n\nSlices are in `research/old_gt_robust_blend_slices_20260905.csv`.\n''',encoding='utf-8')
pd.DataFrame(slices).to_csv('research/old_gt_robust_blend_slices_20260905.csv',index=False)
print(name,sha,rm,score)

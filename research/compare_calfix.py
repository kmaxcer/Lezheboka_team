import pandas as pd, numpy as np
from pathlib import Path
base=Path('outputs/test_20260905_1350'); new=Path('outputs/test_20260905_1350_calfix')
for k in ['regular','wide']:
 a=pd.read_csv(base/f'model_newtest_extended_hgb_{k}_20260905.csv'); b=pd.read_csv(new/f'model_newtest_calfix_hgb_{k}_20260905.csv'); d=b.primary_ndvi_pred-a.primary_ndvi_pred; print(k,'max',d.abs().max(),'mean',d.mean(),'rmse_diff',np.sqrt(np.mean(d*d)), 'corr',a.primary_ndvi_pred.corr(b.primary_ndvi_pred)); z=b.copy(); z.primary_ndvi_pred=.5*(a.primary_ndvi_pred+b.primary_ndvi_pred); z.to_csv(new/f'model_newtest_calfix_hgb_{k}_avg_20260905.csv',index=False,float_format='%.9f')

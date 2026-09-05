"""Проверка date-level weather контекста как безопасной коррекции HGB."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'research'; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
ID,DATE='anon_polygon_id','date'
def rm(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float); return float(np.sqrt(np.mean((p-y)**2)))
def weather_features(rows, private, train):
 pr=private.copy(); pr[DATE]=pd.to_datetime(pr[DATE]); tr=train.copy(); tr[DATE]=pd.to_datetime(tr[DATE])
 # date medians from rows that remain visible; weather is independent of target.
 cols=['era5_temp_c','era5_precip_mm']
 g=pr.groupby(DATE)[cols].agg(['median','mean','std','quantile'])
 # explicit quantiles separately to avoid pandas naming ambiguity
 out=pd.DataFrame({DATE:rows[DATE].values})
 for c in cols:
  s=pr.groupby(DATE)[c]; out[c+'_date_med']=rows[DATE].map(s.median()).to_numpy(float); out[c+'_date_std']=rows[DATE].map(s.std()).to_numpy(float)
  doy=tr[DATE].dt.dayofyear; vals=pd.DataFrame({'doy':doy,'v':pd.to_numeric(tr[c],errors='coerce')}).dropna(); clim=vals.groupby(vals.doy//15).v.median(); out[c+'_doy_med']=(rows[DATE].dt.dayofyear//15).map(clim).to_numpy(float); out[c+'_anom']=out[c+'_date_med']-out[c+'_doy_med']
 return out.drop(columns=[DATE]).replace([np.inf,-np.inf],np.nan).fillna(0.0)
def main():
 rows=pd.read_csv(R/'hgb_exact_mask_validation_20260905_rows.csv',parse_dates=[DATE]); tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE]); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE]); wf=weather_features(rows,pr,tr); rows=rows.reset_index(drop=True); X=wf.to_numpy(float); resid=rows.hgb_sq_clip.to_numpy(float)-rows.base25.to_numpy(float); y=rows.truth.to_numpy(float); out=[]
 for seed in [0,1,2,70404]:
  te=rows.seed.eq(seed).to_numpy(); fit=~te
  for alpha in [1,10,100,1000]:
   m=Ridge(alpha=alpha).fit(X[fit],resid[fit]); corr=m.predict(X[te]); p=np.clip(rows.base25.to_numpy(float)[te]+rows.hgb_sq_clip.to_numpy(float)[te]-rows.base25.to_numpy(float)[te]-corr*0.0,-.2,1.1)
   # correction is applied to base25 residual learned directly
   p=np.clip(rows.base25.to_numpy(float)[te]+m.predict(X[te]),-.2,1.1)
   out.append({'eval_seed':seed,'alpha':alpha,'base_rmse':rm(y[te],rows.base25.to_numpy(float)[te]),'corrected_rmse':rm(y[te],p),'delta':rm(y[te],p)-rm(y[te],rows.base25.to_numpy(float)[te])})
 met=pd.DataFrame(out); met.to_csv(R/'weather_date_correction_probe_20260905.csv',index=False); best=met.sort_values('corrected_rmse').iloc[0]; report=['# Date-level weather correction probe (2026-09-05)','', 'Date median/dispersion ERA5 features are derived from private rows on the same date and train DOY climatology; no target labels are used. Ridge learns residual `(hgb_sq_clip - base25)` on the other three pseudo-mask seeds and is scored on the held-out seed.', '', met.to_string(index=False), '', f'Лучший OOF результат: {best.to_dict()}. Коррекция не материализована в submission; released GT не использовался для fit.']; (R/'weather_date_correction_probe_20260905_report.md').write_text('\n'.join(report)+'\n',encoding='utf8'); print(met.to_string(index=False))
if __name__=='__main__': main()

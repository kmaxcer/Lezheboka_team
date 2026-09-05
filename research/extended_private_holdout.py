"""Leakage-safe extended-HGB audit on a private-like all-year holdout."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904"); R=ROOT/'research'; sys.path.insert(0,str(R)); from extended_seed_cohorts import evaluate
TARGET='primary_ndvi'
def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False); tr['is_synthetic_gap']=False; pr['is_synthetic_gap']=pr.is_synthetic_gap.fillna(False).astype(bool); d=pd.concat([tr,pr],ignore_index=True,sort=False); d.date=pd.to_datetime(d.date); d.year=d.year.fillna(d.date.dt.year).astype(int); d.doy=d.doy.fillna(d.date.dt.dayofyear).astype(int); d['_truth']=d[TARGET].astype(float); hid=d.is_synthetic_gap.to_numpy(bool); private=(d._origin if '_origin' in d else pd.Series(['train']*len(tr)+['private']*len(pr))).astype(str).eq('private').to_numpy(); known=(private & d[TARGET].notna().to_numpy() & ~hid); hold=np.zeros(len(d),bool); rng=np.random.default_rng(70404); tab=pd.DataFrame({'id':d.anon_polygon_id.astype(str),'yr':d.date.dt.year})
 for _,ix0 in tab.loc[known].groupby(['id','yr'],sort=False).groups.items():
  ix=np.asarray(ix0,int); n=max(1,int(round(.15*len(ix)))); hold[rng.choice(ix,size=min(n,len(ix)),replace=False)]=True
 met,p=evaluate(d,hold,hid|hold,4,'private_all15'); p['cohort']=np.where(p.anon_polygon_id.astype(str).isin(set(tr.anon_polygon_id.astype(str))),'shared','new'); p.to_csv(R/'extended_private_holdout_predictions.csv',index=False); pd.DataFrame([met]).to_csv(R/'extended_private_holdout_results.csv',index=False)
 c=p.groupby('cohort',as_index=False).apply(lambda g:pd.Series({'n':len(g),'rmse':float(np.sqrt(np.mean((g.pred.to_numpy()-g.truth.to_numpy())**2))),'mae':float(np.mean(abs(g.pred.to_numpy()-g.truth.to_numpy())))}),include_groups=False).reset_index(drop=True); c.to_csv(R/'extended_private_holdout_aggregate.csv',index=False); print(pd.DataFrame([met]).to_string(index=False)); print(c.to_string(index=False))
if __name__=='__main__': main()

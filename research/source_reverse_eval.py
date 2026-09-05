import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
ROOT=Path(r'C:/Users/kmaxc/PycharmProjects/hack/_1/_lezheboka'); DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'research')); from validate import make_fold
from overnight_source_eval import _mask_private_like,_source_labels
SRCMAP={'s2':0,'landsat':1,'modis':2,'none':-1}

def src(d): return np.select([d.s2_ndvi.notna(), d.landsat_ndvi.notna(), d.modis_ndvi.notna()],[0,1,2],-1)
def evaluate(d,q,y):
 d=d.copy().reset_index(drop=True); q=np.asarray(q); s=src(d); d['src']=s; d['idnum']=d.anon_polygon_id.str.extract(r'(\d+)',expand=False).astype(int);d['yr']=d.date.dt.year;d['doy']=d.date.dt.dayofyear
 vis=(~q)&(s>=0); out={}
 def mode(keys):
  z=d.loc[vis].groupby(keys+['src']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0);qi=pd.MultiIndex.from_frame(d.loc[q,keys]) if len(keys)>1 else pd.Index(d.loc[q,keys[0]]);a=z.reindex(qi).fillna(0).to_numpy(float);return a.argmax(1)
 for name,keys in [('date',['date']),('datecrop',['date','crop_type']),('doy',['doy']),('aoi_doy',['anon_polygon_id','doy']),('aoi_year',['anon_polygon_id','yr']),('crop_doy',['crop_type','doy'])]: out[name]=accuracy_score(y,mode(keys))
 qix=np.flatnonzero(q); visidx=np.flatnonzero(vis); dates=d.date.to_numpy(); ids=d.idnum.to_numpy(); crops=d.crop_type.astype(str).to_numpy()
 for typ,crop in [('n',False),('nc',True)]:
  for w in [1,2,3,4,6,8,12,16,24]:
   pred=[]
   for i in qix:
    sel=visidx[(dates[visidx]==dates[i]) & (np.abs(ids[visidx]-ids[i])<=w)]
    if crop: 
     ss=sel[crops[sel]==crops[i]]
     if len(ss): sel=ss
    if len(sel)==0: pred.append(-1);continue
    pred.append(np.bincount(s[sel],minlength=3).argmax())
   out[f'{typ}{w}']=accuracy_score(y,np.array(pred))
 return out,len(qix)
tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False);rows=[]
for yr in [2019,2020,2021,2022,2023,2024]:
 f,t=make_fold(tr.copy(),pr.copy(),yr); q=f.is_synthetic_gap.fillna(False).to_numpy(bool); y=np.array([SRCMAP[x] for x in _source_labels(tr)[q]]);r,n=evaluate(f,q,y); rows.append(pd.Series(r,name=f'exact{yr}')); print('exact',yr,n,sorted(r.items(),key=lambda x:-x[1])[:8],flush=True)
for seed in [0,1,2]:
 f,q=_mask_private_like(pr,seed);y=np.array([SRCMAP[x] for x in _source_labels(pr)[q]]);r,n=evaluate(f,q,y);rows.append(pd.Series(r,name=f'random{seed}'));print('random',seed,n,sorted(r.items(),key=lambda x:-x[1])[:8],flush=True)
o=pd.DataFrame(rows);o.to_csv(ROOT/'research/source_reverse_features.csv'); print('mean',o.mean().sort_values(ascending=False).head(20).to_string())

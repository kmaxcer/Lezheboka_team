"""Add leakage-safe nearest source-sequence features to the schedule classifier."""
from pathlib import Path
import sys, warnings
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1];DATA=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0,str(ROOT/'research'));sys.path.insert(0,str(ROOT/'src'))
from source_classifier_eval import source,features
from validate import make_fold
from overnight_source_eval import _mask_private_like
SRC=np.array(['s2','landsat','modis'])

def seq_features(frame,known,lab):
 d=frame.copy().reset_index(drop=True);d.date=pd.to_datetime(d.date).astype('datetime64[ns]');d['_year']=d.date.dt.year.astype(int);n=len(d);out=np.zeros((n,12),float)
 z=d.loc[known&(lab>=0),['anon_polygon_id','_year','date']].copy();z['_src']=lab[known&(lab>=0)]
 q=d[['anon_polygon_id','_year','date']].copy();q['_i']=np.arange(n)
 # Offset query timestamp by 1ns to avoid matching itself for labeled rows.
 l=q.sort_values('date');r=z.sort_values('date')
 rb=r.rename(columns={'date':'date_b','_src':'src_b'}).sort_values('date_b');rf=r.rename(columns={'date':'date_f','_src':'src_f'}).sort_values('date_f')
 lb=pd.merge_asof(l.assign(_key=l.date-pd.Timedelta(nanoseconds=1)).sort_values('_key'),rb,left_on='_key',right_on='date_b',by=['anon_polygon_id','_year'],direction='backward',tolerance=pd.Timedelta(days=90))
 lf=pd.merge_asof(l.assign(_key=l.date+pd.Timedelta(nanoseconds=1)).sort_values('_key'),rf,left_on='_key',right_on='date_f',by=['anon_polygon_id','_year'],direction='forward',tolerance=pd.Timedelta(days=90))
 # Restore original order via _i.
 lb=lb.set_index('_i').reindex(np.arange(n));lf=lf.set_index('_i').reindex(np.arange(n))
 bd=(lb.date-lb.date_b).dt.days.abs().to_numpy(float);fd=(lf.date_f-lf.date).dt.days.abs().to_numpy(float)
 bs=lb.src_b.to_numpy(float);fs=lf.src_f.to_numpy(float)
 bvalid=np.isfinite(bs);fvalid=np.isfinite(fs)
 usef=(~np.isfinite(bd))|(np.isfinite(fd)&(fd<bd));usef&=fvalid
 src=np.where(usef,fs,bs);dist=np.where(usef,fd,bd)
 for j in range(3):out[:,j]=(src==j);out[:,3+j]=np.where(src==j,dist,90.)
 out[:,6]=np.where(np.isfinite(bd),bs,-1);out[:,7]=np.where(np.isfinite(fd),fs,-1)
 out[:,8]=np.minimum(np.nan_to_num(bd,nan=90.),np.nan_to_num(fd,nan=90.))
 out[:,9]=np.isfinite(bd).astype(float);out[:,10]=np.isfinite(fd).astype(float);out[:,11]=(out[:,6]==out[:,7]).astype(float)
 return pd.DataFrame(out,index=np.arange(n),columns=[f'seq{i}' for i in range(12)])

def one(fr,q,true):
 lab=source(fr);known=fr.primary_ndvi.notna().to_numpy(bool)&~q
 X=features(fr,known,lab).join(seq_features(fr,known,lab));train=known&(lab>=0)
 rows=[]
 for name,m in [('hgb_seq',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=31,learning_rate=.06,l2_regularization=5.,random_state=42))]:
  m.fit(X.loc[train],lab[train]);p=m.predict_proba(X.loc[q]); rows.append((name,m.classes_[p.argmax(1)],p))
 return rows
def run():
 warnings.filterwarnings('ignore');tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=['date'],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False);R=[]
 for y in []: # full train folds are expensive; private-like checks are the relevant route audit
  f,_=make_fold(tr.copy(),pr.copy(),y);q=f.is_synthetic_gap.fillna(False).to_numpy(bool);true=source(tr)[q]
  for n,p,pp in one(f,q,true):R.append(dict(protocol='exact',part=y,method=n,N=len(true),acc=accuracy_score(true,p),ll=log_loss(true,pp,labels=[0,1,2])))
  print('exact',y,flush=True)
 for s in [0,1,2]:
  f,q=_mask_private_like(pr,s);true=source(pr)[q];q=f.is_synthetic_gap.to_numpy(bool)
  for n,p,pp in one(f,q,true):R.append(dict(protocol='random',part=s,method=n,N=len(true),acc=accuracy_score(true,p),ll=log_loss(true,pp,labels=[0,1,2])))
  print('random',s,flush=True)
 o=pd.DataFrame(R);o.to_csv(ROOT/'research/source_classifier_sequence_results.csv',index=False);print(o.to_string(index=False));print(o.groupby('method').apply(lambda g:pd.Series(N=g.N.sum(),acc=np.average(g.acc,weights=g.N),ll=np.average(g.ll,weights=g.N)),include_groups=False))
if __name__=='__main__':run()

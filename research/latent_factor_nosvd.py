"""Run only fast one-factor and multi-peer variants."""
import numpy as np, pandas as pd
import latent_factor_eval as lf
from latent_factor_eval import DATA,R,ID,DATE,TARGET,GAP,holdout_mask,score_frame

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False);tr[GAP]=False;pr[GAP]=pr[GAP].fillna(False).astype(bool)
 hold=holdout_mask(pr);gp=hold|pr[GAP].to_numpy(bool);tr2=tr.copy();p2=pr.copy();tr2['_origin']='train';p2['_origin']='private';dyn=[c for c in p2 if c not in [ID,DATE,'crop_type',GAP]];p2.loc[gp,dyn]=np.nan;p2.loc[gp,GAP]=True
 ref=pd.concat([tr2,p2],ignore_index=True,sort=False);ref[DATE]=pd.to_datetime(ref[DATE]);lab=pd.concat([tr[[ID,DATE,TARGET]],pr[[ID,DATE,TARGET]]],ignore_index=True).rename(columns={TARGET:'truth2'});ref=ref.merge(lab,on=[ID,DATE],how='left',validate='one_to_one');ref['_truth']=ref.truth2;ref.drop(columns='truth2',inplace=True);gaps=ref[GAP].fillna(False).to_numpy(bool);hk=set(map(tuple,pr.loc[hold,[ID,DATE]].to_numpy()));gaps=gaps|np.array([tuple(x) in hk for x in ref[[ID,DATE]].to_numpy()]);ref.loc[gaps,TARGET]=np.nan
 d,y,known=lf._panel_arrays(ref,gaps);key={(str(a),pd.Timestamp(b)):i for i,(a,b) in enumerate(zip(d[ID],d[DATE]))};q=pr.loc[hold,[ID,DATE]].copy();q[DATE]=pd.to_datetime(q[DATE]);q['truth']=pr.loc[hold,TARGET].to_numpy(float);qi=np.array([key[(str(a),pd.Timestamp(b))] for a,b in q[[ID,DATE]].itertuples(index=False,name=None)])
 rows=[];preds={};bases={}
 for bw in (8,16,24,32):
  base=lf._seasonal_baseline(d,y,known,bw=bw);bases[bw]=base
  for rob in ('median','trim','mean'):
   for sh in (0.,.15,1.5):
    nm=f'factor_bw{bw}_{rob}_sh{sh}';p=lf._date_factor_prediction(d,y,known,qi,base,robust=rob,shrink=sh);rows.append(score_frame(q,p,nm));preds[nm]=p
  for k in (3,5,8,12,16):
   nm=f'ridge_bw{bw}_k{k}';p=lf._multi_peer_ridge(d,y,known,qi,base,k=k,alpha=10.,min_common=30);rows.append(score_frame(q,p,nm));preds[nm]=p
 out=pd.DataFrame(rows).sort_values('rmse');out.to_csv(R/'latent_factor_nosvd_results.csv',index=False);train_ids=set(tr[ID].astype(str));q['cohort']=np.where(q[ID].astype(str).isin(train_ids),'shared','new');q['yearx']=q[DATE].dt.year;details=[]
 for nm in out.method:
  p=preds[nm]
  for gname,g in [('all',np.ones(len(q),bool)),('new',q.cohort.eq('new').to_numpy()),('shared',q.cohort.eq('shared').to_numpy()),('history',(q.yearx<2025).to_numpy()),('2025',(q.yearx==2025).to_numpy()),('new_history',((q.cohort=='new')&(q.yearx<2025)).to_numpy()),('new_2025',((q.cohort=='new')&(q.yearx==2025)).to_numpy()),('shared_2025',((q.cohort=='shared')&(q.yearx==2025)).to_numpy())]:
   z=score_frame(q.loc[g],p[g],nm);z['cohort']=gname;details.append(z)
 pd.DataFrame(details).to_csv(R/'latent_factor_nosvd_cohorts.csv',index=False);tab=q[[ID,DATE,'truth','cohort','yearx']].copy();
 for nm,p in preds.items():tab[nm]=p
 tab.to_csv(R/'latent_factor_nosvd_predictions.csv',index=False);print(out.head(30).to_string(index=False));print(pd.DataFrame(details).sort_values(['cohort','rmse']).head(80).to_string(index=False))
if __name__=='__main__':main()

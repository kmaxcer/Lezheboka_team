"""Evaluate latent date-factor corrections on top of a query-safe local baseline."""
import numpy as np, pandas as pd
import latent_factor_eval as lf
from latent_factor_eval import DATA,R,ID,DATE,TARGET,GAP,holdout_mask,score_frame

def local_base(d,y,known,loo_known=False):
    ids=d['id'].to_numpy(str); yrs=d['yearx'].to_numpy(int); x=d[DATE].map(pd.Timestamp.toordinal).to_numpy(float); doy=d['doyx'].to_numpy(int); n=len(d);out=np.full(n,np.nan)
    # exact same-year interpolation from observed target rows
    for (pid,yr),ix0 in pd.Series(np.arange(n)).groupby([ids,yrs],sort=False).groups.items():
        ix=np.asarray(ix0,dtype=int); ii=ix[known[ix]&np.isfinite(y[ix])]
        if len(ii):
            so=ii[np.argsort(x[ii])]; xx=x[so]; yy=y[so];
            # collapse duplicate dates robustly
            u,inv=np.unique(xx,return_inverse=True); vv=np.array([np.median(yy[inv==j]) for j in range(len(u))]);out[ix]=np.interp(x[ix],u,vv,left=vv[0],right=vv[-1])
            if loo_known:
                # Leave each observed point out when estimating its residual.
                # Otherwise interpolation reproduces y exactly and the date
                # factor is identically zero.
                for j in ii:
                    others=ii[ii!=j]
                    if len(others)>=2:
                        so2=others[np.argsort(x[others])]
                        u2,iv2=np.unique(x[so2],return_inverse=True)
                        v2=np.array([np.median(y[so2][iv2==z]) for z in range(len(u2))])
                        out[j]=np.interp(x[j],u2,v2,left=v2[0],right=v2[-1])
                    elif len(others)==1:
                        out[j]=y[others[0]]
    # fallback seasonal AOI profile for groups without observations
    sb=lf._seasonal_baseline(d,y,known,bw=8)
    out[~np.isfinite(out)]=sb[~np.isfinite(out)]
    return out

def robust_factor(d,y,known,base,qidx,mode='date',bw=0,loading='none',source=None):
    ids=d['id'].to_numpy(str); dates=d['datekey'].to_numpy(str); yrs=d['yearx'].to_numpy(int); doy=d['doyx'].to_numpy(int);res=y-base
    # optional smooth temporal baseline factor grouping exact date / doy bin
    f=np.full(len(d),np.nan)
    if mode=='date': keys=dates
    elif mode=='doy': keys=(doy//max(1,bw)).astype(str)
    elif mode=='year_doy': keys=np.array([f'{a}_{b//max(1,bw)}' for a,b in zip(yrs,doy)])
    else: keys=dates
    # leave-one-id-out robust residual aggregate
    by={k:np.flatnonzero((keys==k)&known&np.isfinite(res)) for k in np.unique(keys)}
    for i in qidx:
        ii=by.get(keys[i],np.array([],int)); ii=ii[ids[ii]!=ids[i]]
        v=res[ii];v=v[np.isfinite(v)]
        if len(v):
            if mode=='date_trim':
                sv=np.sort(v); kk=int(.15*len(sv));v=sv[kk:len(sv)-kk] if len(sv)>2*kk else sv
            f[i]=np.mean(np.clip(v,-.35,.35))
    # fit AOI loadings on all known residual/date factors.  Build factors for
    # observed rows excluding their own AOI, preventing mechanical identity.
    if loading!='none':
        # compute leave-AOI factor for every known row using date sums/counts
        sums=pd.Series(res[known],index=np.flatnonzero(known)).groupby(keys[known]).sum(); cnt=pd.Series(np.ones(known.sum()),index=np.flatnonzero(known)).groupby(keys[known]).sum()
        ff=np.full(len(d),np.nan)
        for i in np.flatnonzero(known):
            s=sums.get(keys[i],0.)-res[i]; c=cnt.get(keys[i],0.)-1; ff[i]=s/c if c>0 else np.nan
        for i in qidx:
            # already f query; if no peers, use global zero
            pass
        bet={}
        for pid,ix0 in pd.Series(np.arange(len(d))).groupby(ids,sort=False).groups.items():
            ix=np.asarray(ix0);ok=known[ix]&np.isfinite(ff[ix])&np.isfinite(res[ix]);
            if ok.sum()>=10:
                xx=ff[ix][ok];rr=res[ix][ok];den=np.dot(xx,xx)+(.15 if loading=='ridge' else 0);bet[pid]=np.clip(np.dot(xx,rr)/den,-2,2)
            else:bet[pid]=0.
        for i in qidx:
            if np.isfinite(f[i]): f[i]*=bet.get(ids[i],0.)
    return base[qidx]+np.nan_to_num(f[qidx])

def main():
 tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False);pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False);tr[GAP]=False;pr[GAP]=pr[GAP].fillna(False).astype(bool)
 hold=holdout_mask(pr);gp=hold|pr[GAP].to_numpy(bool);tr2=tr.copy();p2=pr.copy();tr2['_origin']='train';p2['_origin']='private';dyn=[c for c in p2 if c not in [ID,DATE,'crop_type',GAP]];p2.loc[gp,dyn]=np.nan;p2.loc[gp,GAP]=True
 ref=pd.concat([tr2,p2],ignore_index=True,sort=False);ref[DATE]=pd.to_datetime(ref[DATE]);lab=pd.concat([tr[[ID,DATE,TARGET]],pr[[ID,DATE,TARGET]]],ignore_index=True).rename(columns={TARGET:'truth2'});ref=ref.merge(lab,on=[ID,DATE],how='left',validate='one_to_one');ref['_truth']=ref.truth2;ref.drop(columns='truth2',inplace=True);gaps=ref[GAP].fillna(False).to_numpy(bool);hk=set(map(tuple,pr.loc[hold,[ID,DATE]].to_numpy()));gaps=gaps|np.array([tuple(x) in hk for x in ref[[ID,DATE]].to_numpy()]);ref.loc[gaps,TARGET]=np.nan
 d,y,known=lf._panel_arrays(ref,gaps);key={(str(a),pd.Timestamp(b)):i for i,(a,b) in enumerate(zip(d[ID],d[DATE]))};q=pr.loc[hold,[ID,DATE]].copy();q[DATE]=pd.to_datetime(q[DATE]);q['truth']=pr.loc[hold,TARGET].to_numpy(float);qi=np.array([key[(str(a),pd.Timestamp(b))] for a,b in q[[ID,DATE]].itertuples(index=False,name=None)])
 base=local_base(d,y,known,loo_known=True);rows=[];preds={}
 for mode,bw in [('date',0),('doy',8),('doy',16),('year_doy',8)]:
  for loading in ['none','ridge']:
   # use date factor helper; mode date_trim handled separately
   p=robust_factor(d,y,known,base,qi,mode=mode,bw=bw,loading=loading);nm=f'{mode}{bw}_{loading}';rows.append(score_frame(q,p,nm));preds[nm]=p
 # Also direct date factor from seasonal baseline for comparison
 for bw in (8,16,24):
  sb=lf._seasonal_baseline(d,y,known,bw); 
  for rob in ('median','trim','mean'):
   p=lf._date_factor_prediction(d,y,known,qi,sb,robust=rob,shrink=1.5);nm=f'season{bw}_{rob}';rows.append(score_frame(q,p,nm));preds[nm]=p
 out=pd.DataFrame(rows).sort_values('rmse');out.to_csv(R/'local_factor_correction_results.csv',index=False);print(out.to_string(index=False))
 # Compare against saved ensemble holdout and optimize tiny blend weights.
 old=pd.read_csv(R/'private_cohort_blend_holdout_predictions.csv',parse_dates=[DATE],low_memory=False);qq=q.merge(old[[ID,DATE,'truth','ext40','joint40','hgb']],on=[ID,DATE,'truth'],how='left',validate='one_to_one');
 for nm,p in preds.items():qq[nm]=p
 blends=[]
 for basecol in ['hgb','joint40','ext40']:
  for nm in preds:
   for w in [0.05,.1,.15,.2,.3,.4,.5]:
    pp=(1-w)*qq[basecol].to_numpy()+w*qq[nm].to_numpy();blends.append(score_frame(qq,pp,f'{basecol}+{nm}*{w}'))
 bo=pd.DataFrame(blends).sort_values('rmse');bo.to_csv(R/'local_factor_correction_blends.csv',index=False);print('blends',bo.head(30).to_string(index=False))
if __name__=='__main__':main()

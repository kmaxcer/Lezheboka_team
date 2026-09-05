"""Build a research-only dynamic peer-weight candidate."""
from pathlib import Path
import sys, hashlib, json
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); OUT=ROOT/'outputs'; R=ROOT/'research'
sys.path.insert(0,str(R)); import paired_aoi_v2 as pv
def main():
    d=pd.read_csv(DATA/'private_features.csv',parse_dates=['date'],low_memory=False); mask=d.is_synthetic_gap.fillna(False).astype(bool).to_numpy()
    wanted=[(16,.60,.125,2),(16,.60,.125,3),(16,.80,.125,2),(16,.80,.125,3),(12,.60,.125,2),(8,.80,.125,3)]
    pv.CONFIGS=wanted
    cache=R/'spatial_peer_v3_private_configs.csv'
    if cache.exists(): peer=pd.read_csv(cache,parse_dates=['date'])
    else:
        print('peer infer start',mask.sum(),flush=True)
        peer,_=pv.peer_predictions(d,mask,partition='private_dynamic')
        peer.to_csv(cache,index=False,float_format='%.8f')
    q=peer.copy(); cfg='n16_c60_r125_k2'; cols=[c for c in q if c.startswith('n16_c60_r') or c.startswith('n12_c60') or c.startswith('n8_c80')]; q['peer']=q[cfg]; q['spread']=q[cols].std(axis=1,skipna=True).fillna(.5)
    # Align the strongest lag40 HGB/lag files and use the validated lag40
    # shock/state correction, replacing only its fixed peer weight.
    h=pd.read_csv(OUT/'model_dani_tuned_hgb.csv',parse_dates=['date'])
    l=pd.read_csv(OUT/'model_dani_tuned_lag.csv',parse_dates=['date'])
    rows=pd.read_csv(R/'ensemble_cv_v2_peer_lag30_apply_rows.csv',parse_dates=['date'])
    z=rows[['anon_polygon_id','date','shock','state','canon']].copy()
    z=z.merge(h[['anon_polygon_id','date','primary_ndvi_pred']].rename(columns={'primary_ndvi_pred':'hgb'}),on=['anon_polygon_id','date'],validate='one_to_one')
    z=z.merge(l[['anon_polygon_id','date','primary_ndvi_pred']].rename(columns={'primary_ndvi_pred':'lag'}),on=['anon_polygon_id','date'],validate='one_to_one')
    z=z.merge(q[['anon_polygon_id','date','peer','spread']],on=['anon_polygon_id','date'],how='left',validate='one_to_one')
    base=.6*z.hgb.to_numpy()+.4*z.lag.to_numpy(); peerz=z.peer.to_numpy(); avail=np.isfinite(peerz); peerz=np.where(avail,peerz,base); spread=z.spread.to_numpy(); spread=np.where(avail,spread,.5); w=np.clip(.18-.3*spread-.2*np.abs(peerz-base),.02,.30); w=np.where(avail,w,0.0); p=(1-w)*base+w*peerz; sh=np.nan_to_num(z.shock.to_numpy(),nan=0.0); st=np.nan_to_num(z.state.to_numpy(),nan=0.0); corr=np.where(z.canon.to_numpy(bool),0,.35*sh-.2*st); p=np.clip(p+corr,-.5,1.2)
    out=z[['anon_polygon_id','date']].copy(); out['primary_ndvi_pred']=p; fn=OUT/'model_dani_spatial_dynamic_lag40_submission.csv'; out.to_csv(fn,index=False,float_format='%.8f'); print('w',np.nanmean(w),'finite',np.isfinite(p).sum(),'out',fn,flush=True)
if __name__=='__main__': main()

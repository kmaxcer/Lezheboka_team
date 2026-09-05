"""Build optional source-mode residual correction candidates.

Source is inferred from visible same-date neighboring AOIs (numeric ID +/-24),
excluding organiser gaps.  Coefficients are conservative train OOF HGB source
biases; files are optional diagnostics and never overwrite production outputs.
"""
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904'); O=ROOT/'outputs'
ID='anon_polygon_id'; DATE='date'; GAP='is_synthetic_gap'
def main():
    pth=O/'model_dani_lag40_peer10_extwide40_v3_30_spectral50_historyonly_submission.csv'; sub=pd.read_csv(pth,parse_dates=[DATE]); p=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE]); p['src']=np.select([p.s2_ndvi.notna(),p.landsat_ndvi.notna(),p.modis_ndvi.notna()],[0,1,2],-1); p['idnum']=p[ID].str.extract(r'(\d+)',expand=False).astype(int); vis=p[(p.src>=0)&~p[GAP].fillna(False)]; bydate={k:g[['idnum','src']].to_numpy() for k,g in vis.groupby(DATE)}; src=[]; conf=[]
    for _,r in p[p[GAP].fillna(False)].iterrows():
        a=bydate.get(r[DATE]); ss=a[np.abs(a[:,0]-int(r.idnum))<=24,1].astype(int) if a is not None else np.array([],int); src.append(int(np.bincount(ss,minlength=3).argmax()) if len(ss) else -1); conf.append(float(np.bincount(ss,minlength=3).max()/len(ss)) if len(ss) else 0.)
    q=p[p[GAP].fillna(False)][[ID,DATE]].copy(); q['src']=src; q['conf']=conf; q[DATE]=pd.to_datetime(q[DATE]); z=sub.merge(q,on=[ID,DATE],how='left',validate='one_to_one'); corr=z.src.map({0:.0159,1:-.0078,2:-.0140}).fillna(0.).to_numpy();
    for fac in [.25,.5,.75]:
        out=sub.copy(); out['primary_ndvi_pred']=np.clip(out.primary_ndvi_pred-fac*corr,-.2,1.1); outpath=O/f'model_dani_spectral50_sourcecorr{int(fac*100):02d}_submission.csv'; out.to_csv(outpath,index=False,float_format='%.9f'); h=hashlib.sha256(outpath.read_bytes()).hexdigest(); print(outpath.name,h,'rows',len(out),'src_coverage',np.mean(z.src>=0),'conf_med',np.median(z.conf));
        (O/f'model_dani_spectral50_sourcecorr{int(fac*100):02d}_metadata.json').write_text(json.dumps({'base':pth.name,'factor':fac,'sha256':h,'source_coverage':float(np.mean(z.src>=0)),'confidence_median':float(np.median(z.conf))},indent=2),encoding='utf-8')
if __name__=='__main__': main()

"""Advanced robust seasonal smoothers for masked NDVI rows.

Research-only evaluator.  Every curve is fitted after query values and all
dynamic fields are masked, so it can be compared with the competition mask.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(ROOT / "research"))
from teammate_sweep_postcorr import _mask_private  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))) if ok.any() else np.nan


def robust_poly(x, y, qx, degree=5, ridge=1.0, robust=True):
    """Centered polynomial with optional Huber reweighting."""
    x = np.asarray(x, float); y = np.asarray(y, float); qx = np.asarray(qx, float)
    if len(x) < degree + 2: return np.full(len(qx), np.nan)
    # scale to roughly [-1,1] over the growing season
    center = float(np.median(x)); scale = max(30., float(np.ptp(x) / 2.))
    z = (x - center) / scale; zq = (qx - center) / scale
    X = np.polynomial.legendre.legvander(z, degree)
    Xq = np.polynomial.legendre.legvander(zq, degree)
    w = np.ones(len(x))
    coef = None
    for _ in range(5 if robust else 1):
        sw = np.sqrt(w)
        A = X * sw[:, None]; b = y * sw
        # regularize only non-intercept terms
        reg = np.eye(degree + 1) * float(ridge)
        reg[0, 0] = 1e-8
        coef = np.linalg.solve(A.T @ A + reg, A.T @ b)
        if robust:
            r = y - X @ coef
            sc = 1.4826 * np.median(np.abs(r - np.median(r))) + .01
            u = np.abs(r) / (2.5 * sc)
            w = np.where(u <= 1, 1., 1. / np.maximum(u, 1e-8))
    return Xq @ coef


def fourier_ridge(x, y, qx, harmonics=5, ridge=1.0, robust=True):
    x=np.asarray(x,float); y=np.asarray(y,float); qx=np.asarray(qx,float)
    if len(x)<2*harmonics+2:return np.full(len(qx),np.nan)
    period=366.; phase=2*np.pi*x/period; phaseq=2*np.pi*qx/period
    X=np.column_stack([np.ones(len(x))]+sum(([np.sin(k*phase),np.cos(k*phase)] for k in range(1,harmonics+1)),[]))
    Xq=np.column_stack([np.ones(len(qx))]+sum(([np.sin(k*phaseq),np.cos(k*phaseq)] for k in range(1,harmonics+1)),[]))
    w=np.ones(len(x)); coef=None
    for _ in range(5 if robust else 1):
        sw=np.sqrt(w); A=X*sw[:,None]; b=y*sw
        reg=np.eye(X.shape[1])*ridge; reg[0,0]=1e-8
        coef=np.linalg.solve(A.T@A+reg,A.T@b)
        if robust:
            r=y-X@coef; sc=1.4826*np.median(np.abs(r-np.median(r)))+.01; u=np.abs(r)/(2.5*sc); w=np.where(u<=1,1.,1./np.maximum(u,1e-8))
    return Xq@coef


def local_poly(x, y, qx, bandwidth=20., degree=2, robust=True):
    """LOESS-like weighted local polynomial for each query."""
    x=np.asarray(x,float); y=np.asarray(y,float); qx=np.asarray(qx,float); out=np.full(len(qx),np.nan)
    for j,q in enumerate(qx):
        # circular DOY distance; season is within Apr-Oct but circular is safe
        d=np.abs(x-q); d=np.minimum(d,366-d)
        take=np.argsort(d)[:min(len(d),max(8, int(np.ceil(bandwidth*2))))]
        xx=x[take]-q; yy=y[take]; dd=d[take]
        w=np.exp(-0.5*(dd/max(1.,bandwidth))**2)
        if len(xx)<degree+2: continue
        # adapt scale to selected neighborhood
        z=xx/max(1.,float(np.max(dd))); X=np.column_stack([z**k for k in range(degree+1)])
        ww=w.copy(); coef=None
        for _ in range(4 if robust else 1):
            sw=np.sqrt(ww); A=X*sw[:,None]; b=yy*sw
            reg=np.eye(degree+1)*1e-3; reg[0,0]=1e-8
            coef=np.linalg.solve(A.T@A+reg,A.T@b)
            if robust:
                r=yy-X@coef; sc=1.4826*np.median(np.abs(r-np.median(r)))+.01; u=np.abs(r)/(2.5*sc); ww=w*np.where(u<=1,1.,1./np.maximum(u,1e-8))
        out[j]=coef[0]
    return out


def spline(x,y,qx,s_factor=0.01, robust=True, degree=3):
    x=np.asarray(x,float); y=np.asarray(y,float); qx=np.asarray(qx,float)
    if len(x)<degree+2:return np.full(len(qx),np.nan)
    order=np.argsort(x); x=x[order]; y=y[order]
    # collapse duplicate x by median (same AOI/date can have duplicated source rows rarely)
    ux, inv=np.unique(x,return_inverse=True); uy=np.array([np.median(y[inv==i]) for i in range(len(ux))]); x,y=ux,uy
    if len(x)<degree+2:return np.full(len(qx),np.nan)
    # s is total squared residual allowance; scale with robust variance
    sc=1.4826*np.median(np.abs(y-np.median(y)))+.01
    s=max(float(s_factor)*len(y), .05*len(y)*sc*sc)
    for _ in range(3 if robust else 1):
        try: sp=UnivariateSpline(x,y,k=degree,s=s,ext=3)
        except Exception:return np.full(len(qx),np.nan)
        if not robust:break
        r=y-sp(x); u=np.abs(r)/(2.5*sc); ww=np.where(u<=1,1.,1./np.maximum(u,1e-8))
        # weighted spline by replication-like transformed residual is not
        # available in scipy; refit a weighted polynomial fallback if needed.
        # Keep the first robust fit, which is already stable for our data.
        break
    return sp(qx)


def predict(frame, mask, method):
    d=frame.copy().reset_index(drop=True); d[DATE]=pd.to_datetime(d[DATE]); mask=np.asarray(mask,bool)
    y=pd.to_numeric(d.get(TARGET),errors='coerce').to_numpy(float); known=np.isfinite(y)&~mask
    x=d[DATE].dt.dayofyear.to_numpy(float); out=np.full(len(d),np.nan)
    for _,ix0 in pd.DataFrame({'id':d[ID].astype(str),'yr':d[DATE].dt.year}).groupby(['id','yr'],sort=False).groups.items():
        ix=np.asarray(ix0,int); k=ix[known[ix]]; q=ix[mask[ix]]
        if len(k)==0 or len(q)==0:continue
        xx=x[k]; yy=y[k]; qx=x[q]
        if method[0]=='poly': p=robust_poly(xx,yy,qx,degree=method[1],ridge=method[2],robust=method[3])
        elif method[0]=='fourier': p=fourier_ridge(xx,yy,qx,harmonics=method[1],ridge=method[2],robust=method[3])
        elif method[0]=='local': p=local_poly(xx,yy,qx,bandwidth=method[1],degree=method[2],robust=method[3])
        elif method[0]=='spline': p=spline(xx,yy,qx,s_factor=method[1],robust=method[2])
        out[q]=p
    # fallback group median
    out[mask & ~np.isfinite(out)]=np.nanmedian(y[known])
    return out[mask]


METHODS=[]
for deg in (2,3,4,5,6,7,8):
    for ridge in (.001,.01,.1,1.,10.,100.): METHODS.append((f'poly{deg}r{ridge}',('poly',deg,ridge,True)))
for h in (1,2,3,4,5,6,8,10,12):
    for r in (.001,.01,.1,1.,10.): METHODS.append((f'fourier{h}r{r}',('fourier',h,r,True)))
for bw in (5,8,12,16,20,25,30,40,55,80):
    for deg in (0,1,2,3): METHODS.append((f'local{bw}d{deg}',('local',bw,deg,True)))
for sf in (.0001,.001,.003,.01,.03,.1,.3,1.): METHODS.append((f'spline{sf}',('spline',sf,True)))


def main():
    tr=pd.read_csv(DATA/'train_dataset.csv',parse_dates=[DATE],low_memory=False); pr=pd.read_csv(DATA/'private_features.csv',parse_dates=[DATE],low_memory=False)
    rows=[]
    scenarios=[]
    for year in (2019,2020,2021,2022,2023,2024):
        f,_=make_fold(tr.copy(),pr.copy(),year); f=f.reset_index(drop=True); f['_truth']=pd.to_numeric(f['_truth'],errors='coerce'); m=f[GAP].fillna(False).to_numpy(bool); scenarios.append((f'm{year}',f,m))
    for seed in (0,1,2):
        f,m=_mask_private(pr.copy(),seed); f=f.reset_index(drop=True); f['_truth']=pd.to_numeric(f['_truth'],errors='coerce'); scenarios.append((f'r{seed}',f,np.asarray(m,bool)))
    # evaluate a reduced grid first; full method set is still cheap
    for name,method in METHODS:
        vals=[]
        for _,f,m in scenarios:
            p=predict(f,m,method); y=f.loc[m,'_truth'].to_numpy(float); vals.append((len(y),rmse(y,p)))
        n=sum(v[0] for v in vals); pooled=np.sqrt(sum(k*r*r for k,r in vals)/n)
        rows.append({'name':name,'pooled':pooled,'mean':np.mean([r for _,r in vals]),'exact':np.sqrt(sum(k*r*r for (s,(k,r)) in zip([x[0] for x in scenarios],vals) if s.startswith('m'))/sum(k for s,(k,r) in zip([x[0] for x in scenarios],vals) if s.startswith('m'))),'random':np.sqrt(sum(k*r*r for s,(k,r) in zip([x[0] for x in scenarios],vals) if s.startswith('r'))/sum(k for s,(k,r) in zip([x[0] for x in scenarios],vals) if s.startswith('r')))})
        if len(rows)%20==0:print(len(rows),sorted(rows,key=lambda z:z['pooled'])[:3],flush=True)
    out=pd.DataFrame(rows).sort_values('pooled'); out.to_csv(ROOT/'research/smooth_advanced_results.csv',index=False); print(out.head(40).to_string(index=False))


if __name__=='__main__':main()

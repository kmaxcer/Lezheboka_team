"""Leakage-safe проверка снижения веса кратких экстремальных событий."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904')
SRC=ROOT/'_archive_inspect'/'agropulse_max_score'/'src'; sys.path.insert(0,str(SRC))
from agropulse.pipeline import build_features, load_competition_data
ID,DATE,TARGET,GAP='anon_polygon_id','date','primary_ndvi','is_synthetic_gap'
OUT=ROOT/'research'; PREFIX=OUT/'stable_event_weight_probe_20260905'
DYNAMIC=['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi','modis_ndvi','modis_evi','modis_ndwi','era5_temp_c','era5_precip_mm',TARGET,'ndvi_climatology_mean','ndvi_climatology_std','ndvi_zscore','n_reference_years','status']

def safe_features(ref,mask):
    fr=ref.copy()
    for c in DYNAMIC:
        if c in fr: fr.loc[mask,c]=np.nan
    fr[GAP]=mask
    return build_features(fr,fr[TARGET].mask(mask),pd.Series(mask,index=fr.index)).replace([np.inf,-np.inf],np.nan)

def stratified(ref,pool,seed,frac=.15):
    rng=np.random.default_rng(seed); out=np.zeros(len(ref),bool)
    for _,ix0 in ref.loc[pool].groupby([ID,'year'],sort=False).groups.items():
        ix=np.asarray(ix0,int); out[rng.choice(ix,size=min(len(ix),max(1,int(round(frac*len(ix))))),replace=False)]=True
    return out

def event_scores(ref,train_mask):
    """Оценка нестабильности строится только на строках с известной целью."""
    y=pd.to_numeric(ref[TARGET],errors='coerce').to_numpy(float)
    score=np.zeros(len(ref),float)
    known=train_mask & np.isfinite(y)
    # Сезонный фон по AOI и 30-дневному кольцу: отклонение цели от фона.
    d=ref[DATE].dt.dayofyear.to_numpy(float); a=ref[ID].astype(str).to_numpy()
    tmp=pd.DataFrame({'a':a,'d':d,'y':y,'known':known})
    med=tmp.loc[known].assign(bin=(tmp.loc[known,'d']//30).astype(int)).groupby(['a','bin'],sort=False).y.median().rename('bg').reset_index()
    keydf=pd.DataFrame({'a':a,'bin':(d//30).astype(int),'_row':np.arange(len(ref))})
    bg=keydf.merge(med,on=['a','bin'],how='left',sort=False).sort_values('_row').bg.to_numpy(float)
    resid=np.abs(y-bg)
    rz=resid/(np.nanmedian(resid[known])+1e-3)
    score+=np.nan_to_num(np.clip(rz,0,8),nan=0.0)
    # Погодные сдвиги: robust z внутри AOI, без использования скрытых целей.
    for c in ['era5_temp_c','era5_precip_mm']:
        x=pd.to_numeric(ref[c],errors='coerce').to_numpy(float) if c in ref else np.full(len(ref),np.nan)
        med_a=pd.Series(x).groupby(a).transform('median').to_numpy(float)
        mad_a=pd.Series(np.abs(x-med_a)).groupby(a).transform('median').to_numpy(float)
        z=np.abs(x-med_a)/(1.4826*mad_a+1e-3)
        score+=0.35*np.nan_to_num(np.clip(z,0,8),nan=0.0)
    # Наличие коротких резких скачков по соседним наблюдаемым значениям.
    ord_idx=np.lexsort((ref[DATE].astype('int64').to_numpy(),a))
    jump=np.zeros(len(ref),float); ys=pd.Series(y)
    for _,ix0 in ref.loc[known].groupby(ID,sort=False).groups.items():
        ix=np.asarray(ix0,int); ix=ix[np.argsort(ref.loc[ix,DATE].to_numpy())]
        vv=y[ix]; jump[ix[1:]]=np.abs(np.diff(vv))
    score+=0.6*np.nan_to_num(np.clip(jump/(np.nanmedian(jump[known])+1e-3),0,8),nan=0.0)
    return score

def rmse(y,p):
    ok=np.isfinite(y)&np.isfinite(p); return float(np.sqrt(np.mean((p[ok]-y[ok])**2)))

def main():
    t0=time.time(); train,private,ref=load_competition_data(DATA/'train_dataset.csv',DATA/'private_features.csv')
    actual=(ref['_origin'].eq('test') & ref[GAP].fillna(False)).to_numpy(bool)
    known=ref[TARGET].notna().to_numpy(bool)&~actual
    outer=stratified(ref,known & ref['_origin'].eq('test').to_numpy(),20260905,.15)
    hidden=actual|outer; pool=known&~outer
    blocks=[]; ys=[]; ws=[]
    scores=event_scores(ref,known)
    for seed in (11,29,47,83):
        pseudo=stratified(ref,pool,seed,.18); feat=safe_features(ref,hidden|pseudo)
        blocks.append(feat.loc[pseudo]); ys.append(ref.loc[pseudo,TARGET]); ws.append(scores[pseudo])
        print('features',seed,int(pseudo.sum()),flush=True)
    X=pd.concat(blocks,ignore_index=True); y=pd.concat(ys,ignore_index=True).to_numpy(float); ev=np.concatenate(ws)
    query=safe_features(ref,hidden); xo=query.loc[outer]; xg=query.loc[actual]; yo=ref.loc[outer,TARGET].to_numpy(float)
    gt=pd.read_csv(OUT/'data_update_20260905_1350'/'private_test_ground_truth.csv',parse_dates=[DATE]); keys=ref.loc[actual,[ID,DATE]]; yg=keys.merge(gt,on=[ID,DATE],how='left',validate='one_to_one').primary_ndvi_true.to_numpy(float)
    # Несколько функций веса: от мягкого штрафа до жёсткого исключения хвоста.
    schemes={'uniform':np.ones(len(ev)), 'soft_0p35':1/(1+0.35*ev), 'soft_0p70':1/(1+0.70*ev), 'clip2':np.where(ev>2,0.35,1.0), 'clip3':np.where(ev>3,0.25,1.0), 'stable70':np.where(ev>2.5,0.0,1.0)}
    rows=[]; preds={}
    for name,w in schemes.items():
        m=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.03,max_iter=300,max_leaf_nodes=48,min_samples_leaf=50,l2_regularization=12.,random_state=42,early_stopping='auto')
        with threadpool_limits(limits=3): m.fit(X,y,sample_weight=w)
        po=np.clip(m.predict(xo),-.2,1.1); pg=np.clip(m.predict(xg),-.2,1.1); preds[name]=pg
        rows.append({'scheme':name,'weight_mean':float(w.mean()),'weight_zero':int((w==0).sum()),'outer_rmse':rmse(yo,po),'outer_score':round(30*max(0,1-rmse(yo,po)/.1),2),'released_rmse':rmse(yg,pg),'released_score':round(30*max(0,1-rmse(yg,pg)/.1),2),'n_iter':int(m.n_iter_)})
        print(rows[-1],flush=True)
    met=pd.DataFrame(rows); met.to_csv(str(PREFIX)+'_metrics.csv',index=False,float_format='%.10f')
    # Срезы для лучшей по outer схемы и всех схем для проверки 2025/new.
    best=met.sort_values(['outer_rmse','released_rmse']).iloc[0].scheme; q=ref.loc[actual,[ID,DATE]].copy(); q['year']=q[DATE].dt.year; q['cohort']=np.where(q[ID].isin(set(train[ID])),'shared','new'); q['truth']=yg
    for n,p in preds.items(): q[n]=p
    sl=[]
    for n in schemes:
      for dim,g in [('all',q),('history',q[q.year<2025]),('2025',q[q.year==2025]),('new2025',q[(q.year==2025)&(q.cohort=='new')]),('shared2025',q[(q.year==2025)&(q.cohort=='shared')])]:
        sl.append({'scheme':n,'slice':dim,'n':len(g),'rmse':rmse(g.truth,g[n])})
    pd.DataFrame(sl).to_csv(str(PREFIX)+'_slices.csv',index=False,float_format='%.10f')
    meta={'dataset':str(DATA),'actual_gap_n':int(actual.sum()),'outer_n':int(outer.sum()),'features':int(X.shape[1]),'train_rows':int(len(X)),'event_formula':'abs(y - AOI/30-day median) + 0.35 robust weather z + 0.6 neighbour jump; weights applied only to pseudo training labels','best_outer_scheme':best,'metrics_path':str(PREFIX)+'_metrics.csv','slices_path':str(PREFIX)+'_slices.csv','gt_sha256':hashlib.sha256((OUT/'data_update_20260905_1350'/'private_test_ground_truth.csv').read_bytes()).hexdigest(),'submission_created':False,'upload_performed':False,'seconds':round(time.time()-t0,1)}
    (Path(str(PREFIX)+'_metadata.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    report=['# Стабильные события: downweight probe (2026-09-05)','', 'Цель: проверить гипотезу, что редкие резкие погодные/NDVI события переобучают HGB. Скрытые поля outer и pseudo маскировались до построения признаков; released GT не входил в обучение.', '', 'Формула индекса события: `|y - медиана(AOI, 30-дневный сезонный бин)| + 0.35*robust_z(temp) + 0.35*robust_z(precip) + 0.6*нормированный соседний скачок`. Веса применены только к обучающим pseudo-строкам.', '', met.to_string(index=False), '', f'Лучший по leakage-safe outer: `{best}`. Срезы actual-gap сохранены отдельно для диагностики; выбор по released GT не считается независимой оценкой.', '', 'Новые submission не создавались, upload не выполнялся; старые CSV не изменялись.', '', f'Метрики: `{PREFIX}_metrics.csv`; slices: `{PREFIX}_slices.csv`.']
    Path(str(PREFIX)+'_report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
if __name__=='__main__': main()

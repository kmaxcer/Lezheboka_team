import pandas as pd,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
d=pd.read_csv(r'C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904/train_dataset.csv'); d=d[d.primary_ndvi.notna()].copy(); d['date']=pd.to_datetime(d.date)
cols=['s2_ndvi','landsat_ndvi','modis_ndvi','s2_evi','landsat_evi','modis_evi','s2_ndwi','landsat_ndwi','era5_temp_c','era5_precip_mm','doy','year']
X=d[cols].copy(); X=X.fillna(X.median());
for k in range(1,9): X[f's{k}']=np.sin(2*np.pi*k*d.doy/365.25); X[f'c{k}']=np.cos(2*np.pi*k*d.doy/365.25)
X=X.to_numpy(); y=d.primary_ndvi.to_numpy();
for seed in range(3):
 rng=np.random.default_rng(seed); te=rng.choice(len(d),int(.15*len(d)),False); tr=np.ones(len(d),bool); tr[te]=False
 m=ExtraTreesRegressor(n_estimators=220,min_samples_leaf=8,max_features=.8,n_jobs=-1,random_state=seed).fit(X[tr],y[tr]); p=m.predict(X[te]); print(seed,mean_squared_error(y[te],p)**.5)

from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")

def main():
    tr=pd.read_csv(ROOT/"train_dataset.csv",parse_dates=["date"])
    tr["_year"]=tr.date.dt.year
    for a,b in [("s2_ndvi","landsat_ndvi"),("s2_ndvi","modis_ndvi"),("landsat_ndvi","modis_ndvi")]:
        z=tr[[a,b,"primary_ndvi","crop_type","date","anon_polygon_id"]].dropna(subset=[a,b])
        x=z[a].to_numpy(float); y=z[b].to_numpy(float); coef=np.polyfit(x,y,1)
        print(a,b,"n",len(z),"corr",np.corrcoef(x,y)[0,1],"rmse raw",np.sqrt(np.mean((x-y)**2)),"fit",coef,"rmse fit",np.sqrt(np.mean((y-np.polyval(coef,x))**2)))
        print(" crop",z.groupby("crop_type").apply(lambda g: np.sqrt(np.mean((g[a]-g[b])**2)),include_groups=False).to_dict())
    for c in ["s2_ndvi","landsat_ndvi","modis_ndvi","primary_ndvi"]:
        vals=[]
        for (pid,year),g in tr.dropna(subset=[c]).sort_values("date").groupby(["anon_polygon_id","_year"]):
            d=g[["date",c]].copy(); d["dt"]=d.date.diff().dt.days; d["dy"]=d[c].diff(); vals.append(d[["dt","dy"]])
        q=pd.concat(vals)
        q["sq"]=q.dy*q.dy; q["abs"]=q.dy.abs()
        ag=q.groupby("dt").agg(n=("dy","count"),rmse=("sq",lambda x:np.sqrt(np.nanmean(x))),medabs=("abs","median"))
        print(c,"diff by dt\n",ag.head(20).to_string())
    print("overlap counts\n",tr[["s2_ndvi","landsat_ndvi","modis_ndvi"]].notna().value_counts().head(20).to_string())

if __name__=="__main__": main()

"""Correct source-accuracy sidecar for the private-only fixed-radius audit.

The original probe compared integer route indices to string source labels in
its diagnostic printout (RMSE artifacts are unaffected).  This tiny repair
writes a new suffixed table without modifying the original files.
"""
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]/"research"
def main():
    z=pd.read_csv(R/"source_expert_route_v2_fixed_radius_srcfix_rows.csv"); mp={"s2":0,"landsat":1,"modis":2}; t=z.true_src.map(mp).to_numpy(int); out=[]
    for s,g in z.groupby("seed"):
        tt=g.true_src.map(mp).to_numpy(int); nd=g.near_dist.to_numpy(float)
        for r in (1,2,3,4,5,6,8,16,32):
            cov=np.isfinite(nd)&(nd<=r); idx=g[f"route_fixed_r{r}"].to_numpy(int); out.append({"seed":int(s),"route":f"fixed_r{r}","n":len(g),"coverage":float(cov.mean()),"covered_n":int(cov.sum()),"source_accuracy_covered":float(np.mean(idx[cov]==tt[cov])) if cov.any() else np.nan,"source_accuracy_all":float(np.mean(idx==tt))})
    d=pd.DataFrame(out); d.to_csv(R/"source_expert_route_v2_fixed_radius_srcfix_accuracy_corrected.csv",index=False,float_format="%.10f"); print(d.to_string(index=False))
if __name__=="__main__": main()

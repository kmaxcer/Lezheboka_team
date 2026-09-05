"""Inventory old local Cosmo-like submission files without mutating them."""
from pathlib import Path
import hashlib
import numpy as np, pandas as pd

ROOT=Path(r"C:\Users\kmaxc\Documents\Codex\2026-09-04\ml")
PROJ=Path(__file__).resolve().parents[1]
files=[]
for p in list(ROOT.rglob("*.csv"))+list((PROJ/"outputs").glob("*.csv")):
    if any(x in p.parts for x in [".venv","venv","site-packages","node_modules"]): continue
    try: d=pd.read_csv(p)
    except Exception: continue
    if list(d.columns)!=["anon_polygon_id","date","primary_ndvi_pred"] or len(d)!=3112: continue
    y=pd.to_numeric(d.primary_ndvi_pred,errors="coerce").to_numpy(float)
    files.append({"path":str(p),"rows":len(d),"finite":bool(np.isfinite(y).all()),"unique_keys":int(d[["anon_polygon_id","date"]].drop_duplicates().shape[0]),"min":float(np.nanmin(y)),"max":float(np.nanmax(y)),"mean":float(np.nanmean(y)),"std":float(np.nanstd(y)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
out=pd.DataFrame(files).drop_duplicates("path").sort_values("path")
out.to_csv(PROJ/"research"/"old_cosmo_candidate_manifest_20260905.csv",index=False)
print(out.to_string(index=False))
leader=PROJ/"outputs"/"model_dani_extwide40_v3_30_peerblend12_history_submission.csv"
if leader.exists() and len(out):
    b=pd.read_csv(leader).primary_ndvi_pred.to_numpy(float); rows=[]
    for rec in files:
        d=pd.read_csv(rec["path"]); p=d.primary_ndvi_pred.to_numpy(float)
        rows.append({"path":rec["path"],"rmse_vs_leader":float(np.sqrt(np.mean((p-b)**2))),"corr_vs_leader":float(np.corrcoef(p,b)[0,1]),"maxabs":float(np.max(np.abs(p-b)))})
    pair=pd.DataFrame(rows).sort_values("rmse_vs_leader")
    pair.to_csv(PROJ/"research"/"old_cosmo_candidate_pairwise_20260905.csv",index=False)
    print("pairwise\n",pair.to_string(index=False))

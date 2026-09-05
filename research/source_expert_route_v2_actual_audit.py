"""Observable route coverage audit on the actual private gaps (diagnostic)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "research"; D = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(R))
import source_expert_q1 as q1
import source_expert_route_v2 as rv
from overnight_source_eval import _predict_matrix

def main():
    tr = pd.read_csv(D/"train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(D/"private_features.csv", parse_dates=["date"], low_memory=False)
    pr["is_synthetic_gap"] = pr["is_synthetic_gap"].fillna(False).astype(bool)
    ref, gref, sref, pm, gpr = q1._make_masked_ref(tr, pr, np.zeros(len(pr), bool))
    qkeys = pr.loc[gpr, ["anon_polygon_id", "date", "crop_type"]].copy().reset_index(drop=True)
    qkeys["date"] = pd.to_datetime(qkeys["date"])
    pmatrix, _ = _predict_matrix(pm, train=tr, family="base", k=8, degree=1, bin_days=30, date_weight=1.0)
    pmap = pmatrix.set_index("row_index"); qi = np.flatnonzero(gpr)
    post = np.column_stack([[pmap.loc[i,c] if i in pmap.index else 1/3 for i in qi] for c in ("p_s2", "p_landsat", "p_modis")])
    post = np.where(np.isfinite(post), post, 1/3); post /= post.sum(1, keepdims=True)
    cc, ac, near = rv._neighbor_counts(pm, gpr, qkeys); routes = rv._route_variants(cc, ac, post)
    yr = qkeys.date.dt.year.to_numpy(int); co = np.where(qkeys.anon_polygon_id.astype(str).isin(set(tr.anon_polygon_id.astype(str))), "shared", "new")
    rows = [{"slice":"all", "n":len(qkeys), "near":int(np.isfinite(near).sum()), "near_le2":int((near<=2).sum()), "mid_3_8":int(((near>2)&(near<=8)).sum()), "far_or_none":int(((~np.isfinite(near))|(near>8)).sum()), "r1_any":int((cc[:,0].sum(1)>0).sum()), "r2_any":int((cc[:,1].sum(1)>0).sum()), "r8_any":int((cc[:,3].sum(1)>0).sum())}]
    for sl, m in [("history",yr<2025),("2025",yr==2025),("new",co=="new"),("shared",co=="shared"),("new2025",(co=="new")&(yr==2025)),("shared2025",(co=="shared")&(yr==2025))]:
        if m.sum(): rows.append({"slice":sl,"n":int(m.sum()),"near":int(np.isfinite(near[m]).sum()),"near_le2":int((near[m]<=2).sum()),"mid_3_8":int(((near[m]>2)&(near[m]<=8)).sum()),"far_or_none":int(((~np.isfinite(near[m]))|(near[m]>8)).sum()),"r1_any":int((cc[m,0].sum(1)>0).sum()),"r2_any":int((cc[m,1].sum(1)>0).sum()),"r8_any":int((cc[m,3].sum(1)>0).sum())})
    # Route distributions are useful to ensure all branches have finite expert indices.
    dist=[]
    for name,v in routes.items():
        if v.ndim != 1: continue
        u,c=np.unique(v.astype(int),return_counts=True); dist.append({"route":name,"s2":int(c[u==0][0]) if 0 in u else 0,"landsat":int(c[u==1][0]) if 1 in u else 0,"modis":int(c[u==2][0]) if 2 in u else 0})
    pd.DataFrame(rows).to_csv(R/"source_expert_route_v2_actual_coverage.csv",index=False); pd.DataFrame(dist).to_csv(R/"source_expert_route_v2_actual_route_distribution.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(dist).to_string(index=False))

if __name__ == "__main__": main()

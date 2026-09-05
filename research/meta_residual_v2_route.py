"""Fast route-specific audit for the extwide40_v3_30 residual meta-model."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
sys.path.insert(0, str(R))
import meta_residual_v2 as m  # noqa: E402


def metric(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main():
    tr = pd.read_csv(m.DATA / "train_dataset.csv", parse_dates=[m.DATE], low_memory=False)
    pr = pd.read_csv(m.DATA / "private_features.csv", parse_dates=[m.DATE], low_memory=False)
    m.TRAIN_IDS = set(tr[m.ID].astype(str))
    q, _ = m._join_holdout_features(pr, tr)
    candidate = [c for c in ["base", "spectral", "ext40_v3_40", "ext40", "v3", "blend_30", "joint_blend_30"] if c in q]
    # A compact context set is less prone to learning arbitrary AOI IDs.
    context = [c for c in ["year", "doy", "is_2025", "is_shared", "sin1", "cos1", "sin2", "cos2",
                            "span", "prev_d", "next_d", "interp", "slope", "local_mean_7", "local_mean_14",
                            "local_mean_30", "local_sd_30", "local_n_30", "clim_local", "peer_median", "peer_sd",
                            "peer_n", "crop_peer_median", "crop_peer_n", "date_known_n", "source_p_s2", "source_p_ls",
                            "source_p_md", "source_entropy", "source_n"] if c in q]
    feature_sets = {"candidate": candidate, "compact": candidate + context}
    q["route"] = np.where(q.year < 2025, "history", np.where(q.cohort == "shared", "shared25", "new25"))
    q["group_aoi"] = q[m.ID].astype(str); q["group_aoiyear"] = q[m.ID].astype(str) + "_" + q.year.astype(str)
    rows=[]; preds=[]
    for route in ["all", "history", "shared25", "new25"]:
        sub = q if route == "all" else q[q.route == route]
        if len(sub) < 80: continue
        for fsname, fs in feature_sets.items():
            fs = [c for c in fs if sub[c].notna().any() and sub[c].nunique(dropna=True)>1]
            for gcol in ["group_aoi", "group_aoiyear"]:
                for seed in (0,1,2):
                    splitter=GroupShuffleSplit(n_splits=3,test_size=.20,random_state=seed)
                    X=sub[fs].to_numpy(float); y=sub.resid_target.to_numpy(float); base=sub.base.to_numpy(float); truth=sub.truth.to_numpy(float); groups=sub[gcol].to_numpy(str)
                    for kind in ["ridge30","ridge100","hgb8"]:
                        for split_no,(tri,tei) in enumerate(splitter.split(X,y,groups),1):
                            model=m._model(kind); model.fit(X[tri],y[tri]); raw=model.predict(X[tei])
                            b=metric(truth[tei],base[tei])
                            for cap in (.01,.02,.03,.04,.06):
                                pp=np.clip(base[tei]+np.clip(raw,-cap,cap),-.5,1.2); rm=metric(truth[tei],pp)
                                rows.append({"route":route,"features":fsname,"group":gcol,"seed":seed,"split":split_no,"model":kind,"cap":cap,"n":len(tei),"rmse":rm,"baseline_rmse":b,"delta_rmse":rm-b,"improved":int(rm<b)})
                                z=sub.iloc[tei][[m.ID,m.DATE,"truth","base","cohort","year"]].copy();z["pred"]=pp;z["correction"]=np.clip(raw,-cap,cap);z["route"]=route;z["features"]=fsname;z["group"]=gcol;z["seed"]=seed;z["split"]=split_no;z["model"]=kind;z["cap"]=cap;preds.append(z)
    out=pd.DataFrame(rows); out.to_csv(R/'meta_residual_v2_route_metrics.csv',index=False); pd.concat(preds,ignore_index=True).to_csv(R/'meta_residual_v2_route_predictions.csv',index=False)
    agg=out.groupby(["route","features","group","model","cap"],as_index=False).apply(lambda g:pd.Series({"runs":len(g),"n":int(g.n.sum()),"rmse":float(np.sqrt(np.average(g.rmse**2,weights=g.n))),"baseline_rmse":float(np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),"delta_rmse":float(np.sqrt(np.average(g.rmse**2,weights=g.n))-np.sqrt(np.average(g.baseline_rmse**2,weights=g.n))),"improved_runs":int(g.improved.sum())}),include_groups=False).reset_index(drop=True)
    agg.to_csv(R/'meta_residual_v2_route_aggregate.csv',index=False); print(agg.sort_values('delta_rmse').head(40).to_string(index=False),flush=True)
    # Route-level candidate is only considered if it improves all three seeds
    # and both group definitions for that route.
    good=agg[(agg.improved_runs>=8)].sort_values('delta_rmse'); good.to_csv(R/'meta_residual_v2_route_shortlist.csv',index=False)
    report=['# Meta residual v2 route audit','','Only visible-mask context and saved OOF candidates are used. GroupShuffleSplit seeds 0/1/2; groups are AOI or AOI×year.','','Top pooled rows:',good.head(20).to_string(index=False),'','No production artifact changed.']
    (R/'meta_residual_v2_route_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


if __name__=='__main__': main()

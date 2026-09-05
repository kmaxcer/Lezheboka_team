"""Source-aware experts on the private-like q1 mask (research only).

This script trains one leakage-safe HGB regressor per acquisition source
(S2/Landsat/MODIS), then routes/soft-blends the experts using the observable
acquisition schedule.  It also evaluates the existing source-interpolation
matrix.  No production file is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
ARCH = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ARCH))
sys.path.insert(0, str(ROOT / "src"))
from agropulse.pipeline import FULL_FEATURES, build_features  # type: ignore  # noqa: E402
from infer import _source_labels  # type: ignore  # noqa: E402
from overnight_source_eval import _predict_matrix  # type: ignore  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # type: ignore  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
SRC = np.array(["s2", "landsat", "modis"], dtype=object)
DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi",
    "era5_temp_c", "era5_precip_mm", "year", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "n_reference_years",
    "ndvi_zscore", "status", TARGET,
]


def source_labels(d: pd.DataFrame) -> np.ndarray:
    """Priority source from unmasked sensor columns."""
    return _source_labels(d)


def _expert_model(seed: int, source: str) -> HistGradientBoostingRegressor:
    # Slightly more regular than the archive model: source partitions are
    # smaller and MODIS has appreciably heavier tails.
    pars = dict(
        loss="squared_error", learning_rate=0.035, max_iter=260,
        max_leaf_nodes=40, min_samples_leaf=40, l2_regularization=10.0,
        random_state=seed,
    )
    if source == "modis":
        pars.update(max_leaf_nodes=28, min_samples_leaf=55, l2_regularization=14.0)
    return HistGradientBoostingRegressor(**pars)


def _make_masked_ref(tr: pd.DataFrame, pr: pd.DataFrame, hold: np.ndarray):
    """Build sorted train+private reference and truth/source sidecars."""
    tr = tr.copy(); pr = pr.copy()
    tr[GAP] = False
    pr[GAP] = pr[GAP].fillna(False).astype(bool)
    real_gap = pr[GAP].to_numpy(bool)
    gaps_pr = real_gap | hold
    # Keep an untouched source/truth dictionary before masking.
    orig = pd.concat([tr, pr], ignore_index=True, sort=False)
    orig[DATE] = pd.to_datetime(orig[DATE])
    orig["_truth_key"] = orig[TARGET].astype(float)
    orig["_src_key"] = source_labels(orig)
    # If a key somehow occurs twice, prefer train then first finite value.
    srcmap = {}
    trumap = {}
    for row in orig[[ID, DATE, "_truth_key", "_src_key"]].itertuples(index=False):
        k = (str(row[0]), pd.Timestamp(row[1]))
        if k not in srcmap or srcmap[k] == "none": srcmap[k] = row[3]
        if k not in trumap or not np.isfinite(trumap[k]): trumap[k] = row[2]

    pm = pr.copy()
    for c in DYNAMIC:
        if c in pm: pm.loc[gaps_pr, c] = np.nan
    pm.loc[gaps_pr, GAP] = True
    tr2 = tr.copy(); p2 = pm.copy()
    tr2["_origin"] = "train"; p2["_origin"] = "test"
    tr2["_test_order"] = np.nan; p2["_test_order"] = np.arange(len(p2))
    ref = pd.concat([tr2, p2], ignore_index=True, sort=False)
    ref = ref.sort_values([ID, DATE, "_origin"]).reset_index(drop=True)
    ref[DATE] = pd.to_datetime(ref[DATE])
    ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int)
    ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    ref["_truth"] = [trumap.get((str(i), pd.Timestamp(d)), np.nan)
                     for i, d in zip(ref[ID], ref[DATE])]
    # hidden flags after sorting (all actual gaps plus requested holdout keys)
    hk = {(str(i), pd.Timestamp(d)) for i, d in pr.loc[hold, [ID, DATE]].itertuples(index=False, name=None)}
    gaps_ref = ref[GAP].fillna(False).to_numpy(bool).copy()
    gaps_ref |= np.array([(str(i), pd.Timestamp(d)) in hk for i, d in zip(ref[ID], ref[DATE])])
    ref.loc[gaps_ref, TARGET] = np.nan
    sref = np.array([srcmap.get((str(i), pd.Timestamp(d)), "none") for i, d in zip(ref[ID], ref[DATE])], dtype=object)
    return ref, gaps_ref, sref, pm, gaps_pr


def _fit_experts(ref: pd.DataFrame, gaps_ref: np.ndarray, sref: np.ndarray):
    """Construct OOF features, then fit three source experts."""
    known = ref[TARGET].notna().to_numpy(bool) & ~gaps_ref
    folds = np.full(len(ref), -1, dtype=int)
    rng = np.random.default_rng(42042)
    for _, ix0 in ref.loc[known].groupby(ID, sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int).copy(); rng.shuffle(ix); folds[ix] = np.arange(len(ix)) % 5
    xb, yb, sb = [], [], []
    for f in range(5):
        pseudo = folds == f
        if not pseudo.any(): continue
        hidden = gaps_ref | pseudo
        obs = ref[TARGET].mask(hidden)
        xx = build_features(ref, obs, pd.Series(hidden, index=ref.index))
        keep = pseudo & np.isin(sref, SRC)
        xb.append(xx.loc[keep, FULL_FEATURES]); yb.append(ref.loc[keep, "_truth"].astype(float)); sb.append(sref[keep])
        print(f"OOF fold {f}: {int(keep.sum())}", flush=True)
    X = pd.concat(xb, ignore_index=True); y = pd.concat(yb, ignore_index=True); s = np.concatenate(sb)
    # Query features are built once from genuinely observable rows.
    obsq = ref[TARGET].mask(gaps_ref)
    xq = build_features(ref, obsq, pd.Series(gaps_ref, index=ref.index)).loc[gaps_ref, FULL_FEATURES]
    models = {}; pred = np.full((len(xq), 3), np.nan)
    for j, name in enumerate(SRC):
        take = s == name
        print(f"fit expert {name}: {int(take.sum())}", flush=True)
        m = _expert_model(42 + j, str(name)); m.fit(X.loc[take], y.loc[take]); models[name] = m
        pred[:, j] = m.predict(xq)
    return models, pred, xq


def _metric(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def main():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    hold = make_holdout(pr, seed=70404)
    ref, gaps_ref, sref, pm, gaps_pr = _make_masked_ref(tr, pr, hold)
    models, ep_ref, _ = _fit_experts(ref, gaps_ref, sref)
    # Map expert predictions in sorted-ref gap order to private holdout keys.
    qref = ref.loc[gaps_ref, [ID, DATE]].copy(); qref["_e0"] = ep_ref[:, 0]; qref["_e1"] = ep_ref[:, 1]; qref["_e2"] = ep_ref[:, 2]
    qkeys = pr.loc[hold, [ID, DATE]].copy().reset_index(names="_pr_i")
    q = qkeys.merge(qref, on=[ID, DATE], how="left", validate="one_to_one")
    q["truth"] = pr.loc[hold, TARGET].to_numpy(float)
    q["true_src"] = source_labels(pr)[hold]
    # Observable source posterior and source-interpolation matrix.
    pmatrix, _ = _predict_matrix(pm, train=tr, family="base", k=8, degree=1, bin_days=30, date_weight=1.0)
    pmap = pmatrix.set_index("row_index")
    # _predict_matrix row_index is private positional index.
    qi = q["_pr_i"].to_numpy(int)
    q["p_s2"] = [pmap.loc[i, "pred_s2"] if i in pmap.index else np.nan for i in qi]
    q["p_ls"] = [pmap.loc[i, "pred_landsat"] if i in pmap.index else np.nan for i in qi]
    q["p_md"] = [pmap.loc[i, "pred_modis"] if i in pmap.index else np.nan for i in qi]
    q["post_s2"] = [pmap.loc[i, "p_s2"] if i in pmap.index else 1/3 for i in qi]
    q["post_ls"] = [pmap.loc[i, "p_landsat"] if i in pmap.index else 1/3 for i in qi]
    q["post_md"] = [pmap.loc[i, "p_modis"] if i in pmap.index else 1/3 for i in qi]
    # Existing q1 ext40 baseline.
    basefile = RESEARCH / "private_cohort_blend_holdout_predictions.csv"
    if basefile.exists():
        b = pd.read_csv(basefile, parse_dates=[DATE]); b[DATE] = pd.to_datetime(b[DATE])
        q = q.merge(b[[ID, DATE, "ext40"]], on=[ID, DATE], how="left", validate="one_to_one")
    else: q["ext40"] = np.nan
    E = q[["_e0", "_e1", "_e2"]].to_numpy(float)
    P = q[["p_s2", "p_ls", "p_md"]].to_numpy(float)
    W = q[["post_s2", "post_ls", "post_md"]].to_numpy(float)
    W = np.where(np.isfinite(W), W, 1/3); W = W / W.sum(1, keepdims=True)
    # Expert soft/hard, source matrix soft/hard, oracle bounds, and blends.
    q["expert_soft"] = np.sum(E * W, axis=1)
    q["expert_hard"] = E[np.arange(len(q)), W.argmax(1)]
    q["matrix_soft"] = np.sum(P * W, axis=1)
    q["matrix_hard"] = P[np.arange(len(q)), W.argmax(1)]
    srcidx = {s:i for i,s in enumerate(SRC)}
    q["expert_oracle"] = [E[i, srcidx.get(s, 0)] for i,s in enumerate(q.true_src)]
    q["matrix_oracle"] = [P[i, srcidx.get(s, 0)] for i,s in enumerate(q.true_src)]
    rows=[]
    for name in ["ext40", "expert_soft", "expert_hard", "expert_oracle", "matrix_soft", "matrix_hard", "matrix_oracle"]:
        rows.append({"method":name,"n":len(q),"rmse":_metric(q.truth,q[name])})
    # Blend expert family into ext40 and source matrix; search conservative grids.
    for srcname in ["expert_soft", "expert_hard", "matrix_soft", "matrix_hard"]:
        for w in np.arange(0, 1.01, .05):
            name=f"blend_ext40_{srcname}_{w:.2f}"; p=(1-w)*q.ext40.to_numpy(float)+w*q[srcname].to_numpy(float)
            q[name]=p; rows.append({"method":name,"n":len(q),"rmse":_metric(q.truth,p)})
    # Cohort-aware score table for promising candidates.
    q["year"] = q[DATE].dt.year.astype(int); q["cohort"] = np.where(q[ID].isin(set(tr[ID])), "shared", "new")
    score=[]
    for cohort, g in [("all",q),("history",q[q.year<2025]),("2025",q[q.year==2025]),("new2025",q[(q.year==2025)&(q.cohort=="new")]),("shared2025",q[(q.year==2025)&(q.cohort=="shared")])]:
        for name in ["ext40","expert_soft","expert_hard","expert_oracle","matrix_soft","matrix_hard","matrix_oracle"]:
            score.append({"cohort":cohort,"method":name,"n":len(g),"rmse":_metric(g.truth,g[name])})
        for name in ["blend_ext40_expert_soft_0.10","blend_ext40_expert_soft_0.20","blend_ext40_expert_hard_0.10","blend_ext40_matrix_soft_0.05","blend_ext40_matrix_soft_0.10"]:
            score.append({"cohort":cohort,"method":name,"n":len(g),"rmse":_metric(g.truth,g[name])})
    q.to_csv(RESEARCH / "source_expert_q1_predictions.csv", index=False)
    pd.DataFrame(rows).sort_values("rmse").to_csv(RESEARCH / "source_expert_q1_results.csv", index=False)
    pd.DataFrame(score).to_csv(RESEARCH / "source_expert_q1_cohort_results.csv", index=False)
    print(pd.DataFrame(rows).sort_values("rmse").head(20).to_string(index=False), flush=True)
    print(pd.DataFrame(score).query("method in ['ext40','expert_soft','expert_hard','expert_oracle','matrix_soft','matrix_hard','matrix_oracle']").to_string(index=False), flush=True)


if __name__ == "__main__": main()

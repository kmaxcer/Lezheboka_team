"""Add a leakage-safe v3 component to the independent-mask residual audit.

The expensive ext40-core predictions are read from the preceding independent
audit.  Only v3 is refit on each fresh mask, using a target-masked reference;
then the requested ``.7*ext40 + .3*v3`` anchor is scored with the same AOI
grouped Ridge protocol.  Research outputs only.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
sys.path.insert(0, str(R)); sys.path.insert(0, str(ROOT / "_archive_inspect" / "agropulse_max_score" / "src"))
import meta_residual_v2 as meta  # noqa: E402
from evaluate_v3_private_quick import fit_v3  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_ref(train: pd.DataFrame, private: pd.DataFrame, hold: np.ndarray):
    """Construct the same target-masked reference used by the core audit."""
    p = private.copy().reset_index(drop=True); p[DATE] = pd.to_datetime(p[DATE])
    p["_truth"] = pd.to_numeric(p[TARGET], errors="coerce")
    hidden = p[GAP].fillna(False).astype(bool).to_numpy(); gaps = hidden | np.asarray(hold, bool)
    pm = p.copy(); pm[GAP] = gaps
    for c in meta.DYNAMIC:
        if c in pm: pm.loc[gaps, c] = np.nan
    tr = train.copy(); tr[GAP] = False; tr["_origin"] = "train"; pm["_origin"] = "private"
    # `_truth` is sidecar only and must not be included before label merge.
    pm_model = pm.drop(columns=["_truth"], errors="ignore")
    ref = pd.concat([tr, pm_model], ignore_index=True, sort=False)
    ref[DATE] = pd.to_datetime(ref[DATE]); ref = ref.sort_values([ID, DATE, "_origin"]).reset_index(drop=True)
    ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int); ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    ltr = train[[ID, DATE, TARGET]].rename(columns={TARGET: "_truth"})
    lpr = p[[ID, DATE, "_truth"]]
    labels = pd.concat([ltr, lpr], ignore_index=True); labels[DATE] = pd.to_datetime(labels[DATE])
    ref = ref.merge(labels, on=[ID, DATE], how="left", validate="one_to_one")
    keyset = set(map(tuple, p.loc[gaps, [ID, DATE]].to_numpy()))
    gr = np.array([tuple(x) in keyset for x in ref[[ID, DATE]].to_numpy()], bool)
    ref.loc[gr, TARGET] = np.nan
    return p, ref, gr


def score(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def evaluate(q: pd.DataFrame, mask_seed: int):
    cand = [c for c in ["base", "ext40", "v3", "hgb", "lag", "joint40", "extended"] if c in q]
    ctx = [c for c in ["year", "doy", "is_2025", "is_shared", "sin1", "cos1", "sin2", "cos2", "span", "prev_d", "next_d", "interp", "slope", "local_mean_7", "local_mean_14", "local_mean_30", "local_sd_30", "local_n_30", "clim_local", "peer_median", "peer_sd", "peer_n", "crop_peer_median", "crop_peer_n", "date_known_n", "source_p_s2", "source_p_ls", "source_p_md", "source_entropy", "source_n"] if c in q]
    fs = [c for c in cand + ctx if q[c].notna().any() and q[c].nunique(dropna=True) > 1]
    y = q.resid_target.to_numpy(float); base = q.base.to_numpy(float); truth = q.truth.to_numpy(float); X = q[fs].to_numpy(float); groups = q[ID].astype(str).to_numpy()
    rows = []
    for split, (ti, ei) in enumerate(GroupShuffleSplit(n_splits=3, test_size=.2, random_state=9000 + mask_seed).split(X, y, groups), 1):
        model = meta._model("ridge30"); model.fit(X[ti], y[ti]); raw = model.predict(X[ei]); b = score(truth[ei], base[ei])
        for cap in (.005, .01, .015, .02, .03):
            p = np.clip(base[ei] + np.clip(raw, -cap, cap), -.5, 1.2); r = score(truth[ei], p)
            rows.append({"mask_seed": mask_seed, "split": split, "n": len(ei), "features": len(fs), "cap": cap, "baseline_rmse": b, "rmse": r, "delta_rmse": r-b, "improved": int(r < b)})
    return rows


def main() -> None:
    t0 = time.time(); train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False); private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    train[GAP] = False; private[GAP] = private[GAP].fillna(False).astype(bool)
    core = pd.read_csv(R / "meta_residual_v2_independent_predictions.csv", parse_dates=[DATE], low_memory=False)
    outputs = []; metrics = []
    for seed in (0, 1):
        q = core[core.mask_seed == seed].copy().reset_index(drop=True)
        hold = make_holdout(private, seed)
        p, ref, gaps_ref = make_ref(train, private, hold)
        print(f"v3 mask {seed}: fitting leakage-safe v3", flush=True)
        vp, info = fit_v3(ref, gaps_ref, n_masks=2)
        vp[DATE] = pd.to_datetime(vp[DATE]); vp = vp[[ID, DATE, "v3"]]
        q = q.drop(columns=["v3"], errors="ignore").merge(vp, on=[ID, DATE], how="left", validate="one_to_one")
        q["base"] = .7 * q.ext40.to_numpy(float) + .3 * q.v3.to_numpy(float)
        q["resid_target"] = q.truth.to_numpy(float) - q.base.to_numpy(float)
        metrics.extend(evaluate(q, seed)); outputs.append(q.assign(v3_mask_seed=seed))
        print(f"v3 mask {seed}: rows={len(q)} history={int((q.year<2025).sum())}", flush=True)
    qq = pd.concat(outputs, ignore_index=True); mm = pd.DataFrame(metrics)
    qq.to_csv(R / "meta_residual_v2_independent_v3_predictions.csv", index=False, float_format="%.8f")
    mm.to_csv(R / "meta_residual_v2_independent_v3_metrics.csv", index=False, float_format="%.10f")
    ss = mm.groupby("cap", as_index=False).apply(lambda z: pd.Series({"runs": len(z), "n": int(z.n.sum()), "baseline_rmse": float(np.sqrt(np.average(z.baseline_rmse**2, weights=z.n))), "rmse": float(np.sqrt(np.average(z.rmse**2, weights=z.n))), "delta_rmse": float(np.sqrt(np.average(z.rmse**2, weights=z.n))-np.sqrt(np.average(z.baseline_rmse**2, weights=z.n))), "wins": int(z.improved.sum())}), include_groups=False).reset_index(drop=True)
    ss.to_csv(R / "meta_residual_v2_independent_v3_summary.csv", index=False, float_format="%.10f")
    info = {"mask_seeds": [0, 1], "rows": int(len(qq)), "hidden_rows": int(private[GAP].sum()), "private_sha256": sha(DATA / "private_features.csv"), "train_sha256": sha(DATA / "train_dataset.csv"), "v3_component": ".7*ext40+.3*v3", "seconds": round(time.time()-t0, 1), "production_baseline_overwritten": False}
    (R / "meta_residual_v2_independent_v3_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (R / "meta_residual_v2_independent_v3_report.md").write_text("# meta residual v2 independent v3 audit\n\nFresh masks 0/1; v3 fit sees target-masked reference only.\n\n" + ss.to_string(index=False) + "\n\n" + json.dumps(info, indent=2) + "\n\nNo production artifact changed.\n", encoding="utf-8")
    print(ss.to_string(index=False), flush=True); print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__": main()

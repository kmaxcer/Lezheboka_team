"""Fresh seed=2 source-expert route audit.

This is deliberately separate from ``source_expert_route_v2.py``.  It builds
the independent-mask baseline with the existing leakage-safe helper, fits the
three source experts on masked train/private reference rows, and evaluates
observable same-date/crop routing.  All source labels are retained only as a
scoring sidecar.  Existing artifacts are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
sys.path.insert(0, str(R))

import meta_residual_v2_independent as indep  # noqa: E402
import source_expert_q1 as q1  # noqa: E402
from evaluate_private_cohort_blend import make_holdout  # noqa: E402
from overnight_source_eval import _predict_matrix, _source_labels  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
SOURCES = ("s2", "landsat", "modis")
ROUTE_RADII = (1, 2, 4, 8, 16, 32)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def neighbor_counts(pm: pd.DataFrame, gaps: np.ndarray, qkeys: pd.DataFrame):
    """Observable same-date source counts by numeric AOI radius and crop."""
    d = pm.reset_index(drop=True).copy(); d[DATE] = pd.to_datetime(d[DATE])
    src = np.select([d["s2_ndvi"].notna(), d["landsat_ndvi"].notna(), d["modis_ndvi"].notna()], [0, 1, 2], -1)
    idnum = pd.to_numeric(d[ID].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(-1).to_numpy(int)
    dates = d[DATE].to_numpy(); crops = d["crop_type"].fillna("unknown").astype(str).to_numpy()
    vis = np.flatnonzero((~np.asarray(gaps, bool)) & (src >= 0))
    bydate = {dt: np.asarray(ix, int) for dt, ix in pd.Series(vis, index=vis).groupby(dates[vis])}
    q = qkeys.reset_index(drop=True); qdates = pd.to_datetime(q[DATE]).to_numpy()
    qids = pd.to_numeric(q[ID].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(-1).to_numpy(int)
    qcrops = q.get("crop_type", pd.Series("unknown", index=q.index)).fillna("unknown").astype(str).to_numpy()
    cc = np.zeros((len(q), len(ROUTE_RADII), 3), float); ac = np.zeros_like(cc); near = np.full(len(q), np.inf)
    for n, (dt, aid, crop) in enumerate(zip(qdates, qids, qcrops)):
        z0 = np.asarray(bydate.get(dt, np.empty(0, int)), dtype=int)
        for rj, rad in enumerate(ROUTE_RADII):
            z = z0[np.abs(idnum[z0] - aid) <= rad]
            if len(z): ac[n, rj] = np.bincount(src[z], minlength=3)
            zz = z[crops[z] == crop] if len(z) else z
            if len(zz):
                cc[n, rj] = np.bincount(src[zz], minlength=3)
                near[n] = min(near[n], float(np.min(np.abs(idnum[zz] - aid))))
    return cc, ac, near


def route_variants(cc: np.ndarray, ac: np.ndarray, post: np.ndarray):
    n = len(post); out = {"post_mode": np.argmax(post, axis=1).astype(int)}
    for min_n in (1, 2, 3):
        for min_p in (0.0, .67, .8):
            route = out["post_mode"].copy(); used = np.zeros(n, bool)
            for rj in range(cc.shape[1]):
                c = cc[:, rj]; nn = c.sum(1); pur = c.max(1) / np.maximum(1., nn)
                take = (~used) & (nn >= min_n) & (pur >= min_p)
                route[take] = np.argmax(c[take], axis=1); used |= take
            out[f"crop_hier_n{min_n}_p{int(min_p*100):02d}"] = route
    for rj, rad in enumerate(ROUTE_RADII):
        for typ, c0 in (("crop", cc[:, rj]), ("all", ac[:, rj])):
            for lam in (0., .25, 1.):
                q = c0.copy() + lam * post
                bad = q.sum(1) <= 0; q[bad] = post[bad]; q /= q.sum(1, keepdims=True)
                for power in (1., 2., 4.):
                    w = np.power(np.clip(q, 1e-9, 1.), power); w /= w.sum(1, keepdims=True)
                    out[f"soft_{typ}_r{rad}_l{lam:g}_p{power:g}"] = w
    return out


def fit_seed2(tr: pd.DataFrame, pr: pd.DataFrame):
    seed = 2
    hold = make_holdout(pr, seed=seed)
    # Independent baseline reconstruction (all dynamic columns are masked by
    # _make_q before any helper sees the query rows).
    base_q = indep._make_q(tr, pr, seed)
    if len(base_q) != int(hold.sum()): raise RuntimeError("baseline holdout size mismatch")
    base_map = base_q.set_index([ID, DATE])["ext40"]
    # Source experts use the same masking and OOF protocol as route v2.
    ref, gaps_ref, sref, pm, gaps_pr = q1._make_masked_ref(tr, pr, hold)
    _, ep_ref, _ = q1._fit_experts(ref, gaps_ref, sref)
    qref = ref.loc[gaps_ref, [ID, DATE]].copy().reset_index(drop=True)
    qref[["e_s2", "e_landsat", "e_modis"]] = ep_ref
    qkeys = pr.loc[hold, [ID, DATE, "crop_type"]].copy().reset_index(drop=True); qkeys[DATE] = pd.to_datetime(qkeys[DATE])
    q = qkeys.merge(qref, on=[ID, DATE], how="left", validate="one_to_one")
    q["truth"] = pr.loc[hold, TARGET].to_numpy(float); q["true_src"] = _source_labels(pr)[hold]
    q["baseline"] = [base_map.get((i, d), np.nan) for i, d in q[[ID, DATE]].itertuples(index=False, name=None)]
    if q.baseline.isna().any(): raise RuntimeError("baseline alignment failed")
    # Observable schedule posterior.
    pmatrix, _ = _predict_matrix(pm, train=tr, family="base", k=8, degree=1, bin_days=30, date_weight=1.0)
    pmap = pmatrix.set_index("row_index"); qi = np.flatnonzero(hold)
    post = np.column_stack([[pmap.loc[i, c] if i in pmap.index else 1/3 for i in qi] for c in ("p_s2", "p_landsat", "p_modis")])
    post = np.where(np.isfinite(post), post, 1/3); post /= post.sum(1, keepdims=True)
    cc, ac, near = neighbor_counts(pm, gaps_pr, qkeys)
    vars_ = route_variants(cc, ac, post)
    E = q[["e_s2", "e_landsat", "e_modis"]].to_numpy(float); B = q.baseline.to_numpy(float); y = q.truth.to_numpy(float)
    rows = []
    for name, v in vars_.items():
        psrc = E[np.arange(len(E)), v] if v.ndim == 1 else np.sum(v * E, axis=1)
        # Keep a broad alpha curve for independent validation and policy audit.
        for a in np.arange(0., .81, .05):
            rows.append({"seed": seed, "method": name, "alpha": float(a), "n": len(y), "rmse": rmse(y, (1-a)*B+a*psrc)})
    # Compact row sidecar contains route predictions at alpha=.40 and source
    # expert itself; this makes later alpha/LOO analyses reproducible.
    keep = ["post_mode", "crop_hier_n1_p67", "soft_all_r1_l0_p4", "soft_crop_r1_l0_p4", "soft_all_r2_l0_p4", "soft_all_r32_l0_p4"]
    out = q[[ID, DATE, "truth", "true_src"]].copy(); out["year"] = out[DATE].dt.year.astype(int)
    out["cohort"] = np.where(out[ID].astype(str).isin(set(tr[ID].astype(str))), "shared", "new"); out["near_dist"] = near; out["baseline"] = B
    for name in keep:
        v = vars_[name]; psrc = E[np.arange(len(E)), v] if v.ndim == 1 else np.sum(v * E, axis=1)
        out[f"expert_{name}"] = psrc; out[f"blend_{name}_0.40"] = .6*B+.4*psrc
    # Slice metrics for fixed alpha and analytically optimal alpha are kept in
    # a separate table, with source labels marked as scoring-only.
    slices = [("all", np.ones(len(y), bool)), ("history", out.year.to_numpy() < 2025), ("2025", out.year.to_numpy() == 2025),
              ("new", out.cohort.to_numpy() == "new"), ("shared", out.cohort.to_numpy() == "shared"),
              ("new_2025", (out.cohort == "new").to_numpy() & (out.year == 2025)),
              ("shared_2025", (out.cohort == "shared").to_numpy() & (out.year == 2025)),
              ("near_0_2", np.isfinite(near) & (near <= 2)), ("mid_2_8", np.isfinite(near) & (near > 2) & (near <= 8)),
              ("far_or_none", (~np.isfinite(near)) | (near > 8))]
    for s in SOURCES: slices.append((f"source_{s}", out.true_src.to_numpy() == s))
    sm = []
    for name in keep:
        e = out[f"expert_{name}"].to_numpy(float)
        for sl, m in slices:
            if int(m.sum()) < 10: continue
            d = e[m]-B[m]; den = float(np.dot(d,d)); ao = np.clip(float(np.dot(d, y[m]-B[m])/den), 0., 1.) if den > 1e-12 else 0.
            sm.append({"seed": seed, "method": name, "slice": sl, "n": int(m.sum()), "alpha_opt": ao,
                       "rmse_opt": rmse(y[m], (1-ao)*B[m]+ao*e[m]), "rmse_a040": rmse(y[m], .6*B[m]+.4*e[m]), "rmse_baseline": rmse(y[m], B[m])})
    return out, pd.DataFrame(rows), pd.DataFrame(sm)


def main() -> None:
    t0 = time.time(); tr = pd.read_csv(DATA/"train_dataset.csv", parse_dates=[DATE], low_memory=False); pr = pd.read_csv(DATA/"private_features.csv", parse_dates=[DATE], low_memory=False); tr[GAP] = False; pr[GAP] = pr[GAP].fillna(False).astype(bool)
    print("fitting fresh source route seed=2", flush=True)
    out, metrics, slices = fit_seed2(tr, pr)
    out.to_csv(R/"source_expert_route_v2_seed2_rows.csv", index=False, float_format="%.9f"); metrics.to_csv(R/"source_expert_route_v2_seed2_metrics.csv", index=False, float_format="%.10f"); slices.to_csv(R/"source_expert_route_v2_seed2_slices.csv", index=False, float_format="%.10f")
    # Concise report and metadata; no production/output candidate is created.
    shortlist = metrics[metrics.method.isin(["post_mode", "crop_hier_n1_p67", "soft_all_r1_l0_p4", "soft_all_r2_l0_p4", "soft_all_r32_l0_p4"])].sort_values("rmse").groupby("method", sort=False).head(1)
    lines = ["# Source-expert route v2 fresh seed=2", "", "Independent private-like mask seed=2; source labels are scoring-only. Baseline is rebuilt by `meta_residual_v2_independent._make_q`; all sensor/dynamic fields are masked on query rows.", "", "## Best alpha per route", "", shortlist.to_string(index=False), "", "## Cohort/source/distance alpha audit", "", slices.sort_values(["slice", "rmse_opt"]).to_string(index=False), "", "No existing candidate was overwritten."]
    (R/"source_expert_route_v2_seed2_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    meta = {"seed": 2, "rows": int(len(out)), "private_hidden": int(pr[GAP].sum()), "private_sha256": sha(DATA/"private_features.csv"), "seconds": round(time.time()-t0, 1), "artifacts": ["source_expert_route_v2_seed2_rows.csv", "source_expert_route_v2_seed2_metrics.csv", "source_expert_route_v2_seed2_slices.csv"]}
    (R/"source_expert_route_v2_seed2_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(shortlist.to_string(index=False), flush=True); print(slices.sort_values(["slice", "rmse_opt"]).groupby("slice", sort=False).head(5).to_string(index=False), flush=True)


if __name__ == "__main__": main()

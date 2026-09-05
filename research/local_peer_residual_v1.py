"""Leakage-safe local same-date peer residual correction audit.

This diagnostic asks whether directly using visible same-date/same-crop AOI
targets can correct the source-route baseline.  A seasonal AOI profile is
estimated from visible rows only; peer residuals are then aggregated in
numeric AOI-ID radii.  All correction coefficients are fitted leave-mask-out
over four independent 15% private-like masks.  No production candidate is
overwritten.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
sys.path.insert(0, str(R))
from teammate_sweep_postcorr import _mask_private  # noqa: E402
from source_expert_route_v2_seed2_policy_audit import load as load_route, policy_pred  # noqa: E402

ID, DATE = "anon_polygon_id", "date"
SEEDS = (0, 1, 2, 70404)
RADII = (1, 2, 4, 8, 16, 32, 64)


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def _source(d: pd.DataFrame) -> np.ndarray:
    return np.select([d.s2_ndvi.notna(), d.landsat_ndvi.notna(), d.modis_ndvi.notna()],
                     ["s2", "landsat", "modis"], default="none")


def _profile(d: pd.DataFrame, known: np.ndarray, width: int = 24,
             source_level: bool = False) -> np.ndarray:
    """Hierarchical robust seasonal profile for each row."""
    x = d[[ID, DATE]].copy(); x[DATE] = pd.to_datetime(x[DATE])
    x["year_calc"] = x[DATE].dt.year.astype(int)
    x["bin"] = ((x[DATE].dt.dayofyear.astype(int) - 1) // int(width)).astype(int)
    x["src"] = _source(d)
    y = pd.to_numeric(d["_truth"], errors="coerce").to_numpy(float)
    k = np.asarray(known, bool) & np.isfinite(y)
    oi = np.flatnonzero(k)
    if len(oi) == 0:
        return np.full(len(d), np.nan)
    ob = x.iloc[oi].copy(); ob["y"] = y[oi]
    # Source-specific profile can reduce sensor offsets, but retains all-source
    # fallbacks for sparse AOI/year bins.
    keys = []
    if source_level:
        keys.append([ID, "year_calc", "bin", "src"])
    keys += [[ID, "year_calc", "bin"], [ID, "bin"], ["year_calc", "bin"], ["bin"]]
    prof = x[[ID, "year_calc", "bin", "src"]].copy()
    for j, key in enumerate(keys):
        g = ob.groupby(key, observed=True).y.median().rename(f"p{j}").reset_index()
        prof = prof.merge(g, on=key, how="left")
    p = np.full(len(x), np.nan)
    for j in range(len(keys)):
        p = np.where(np.isfinite(p), p, prof[f"p{j}"].to_numpy(float))
    p = np.where(np.isfinite(p), p, float(np.nanmedian(ob.y)))
    return p


def _peer_features(d: pd.DataFrame, known: np.ndarray, qidx: np.ndarray,
                   width: int = 24) -> pd.DataFrame:
    """Build same-date/crop local residual and raw peer features."""
    d = d.reset_index(drop=True).copy(); d[DATE] = pd.to_datetime(d[DATE])
    ids = pd.to_numeric(d[ID].astype(str).str.extract(r"(\d+)", expand=False),
                        errors="coerce").fillna(-1).to_numpy(int)
    crops = d.crop_type.fillna("unknown").astype(str).to_numpy()
    src = _source(d)
    y = pd.to_numeric(d["_truth"], errors="coerce").to_numpy(float)
    prof = _profile(d, known, width=width, source_level=False)
    prof_src = _profile(d, known, width=width, source_level=True)
    res = np.clip(y - prof, -0.5, 0.5); res_src = np.clip(y - prof_src, -0.5, 0.5)
    k = np.asarray(known, bool) & np.isfinite(y)
    # Deduplicate visible rows by AOI/date: each source observation is an
    # independent row in the sparse frame, but should count once as a peer.
    vi = np.flatnonzero(k)
    bydate = {}
    for dt, gi in pd.Series(vi, index=vi).groupby(d.loc[vi, DATE]):
        bydate[dt] = np.asarray(gi, dtype=int)
    # query metadata
    out = d.loc[qidx, [ID, DATE, "crop_type"]].copy().reset_index(drop=True)
    out["idx"] = qidx
    out["year_calc"] = out[DATE].dt.year.astype(int)
    # Build a moderate set of robust aggregators.  Empty/low-count groups are
    # represented by NaN, preserving leakage-safe fallback behavior.
    for rad in RADII:
        for typ in ("all", "crop"):
            out[f"r{rad}_{typ}_resmed"] = np.nan
            out[f"r{rad}_{typ}_resmean"] = np.nan
            out[f"r{rad}_{typ}_rawmed"] = np.nan
            out[f"r{rad}_{typ}_n"] = 0
            for ss in ("s2", "landsat", "modis"):
                out[f"r{rad}_{typ}_{ss}_resmed"] = np.nan
    # Also exact-date global/crop residuals (not radius-limited).
    out["date_resmed"] = np.nan; out["crop_resmed"] = np.nan
    out["date_resmean"] = np.nan; out["crop_resmean"] = np.nan
    out["date_n"] = 0; out["crop_n"] = 0
    for j, qi in enumerate(qidx):
        z0 = bydate.get(d.at[qi, DATE], np.empty(0, dtype=int))
        # Exclude same AOI; this matters if a visible duplicate is present.
        z0 = z0[ids[z0] != ids[qi]]
        if len(z0) == 0: continue
        zcrop = z0[crops[z0] == crops[qi]]
        for name, z in (("date", z0), ("crop", zcrop)):
            rr = res[z]; yy = y[z]
            ok = np.isfinite(rr) & np.isfinite(yy)
            if ok.any():
                out.at[j, f"{name}_resmed"] = float(np.nanmedian(rr[ok]))
                out.at[j, f"{name}_resmean"] = float(np.nanmean(rr[ok]))
                out.at[j, f"{name}_n"] = int(ok.sum())
        for rad in RADII:
            zr = z0[np.abs(ids[z0] - ids[qi]) <= rad]
            for typ, z in (("all", zr), ("crop", zr[crops[zr] == crops[qi]] if len(zr) else zr)):
                rr = res[z]; yy = y[z]; ok = np.isfinite(rr) & np.isfinite(yy)
                if not ok.any(): continue
                rr = rr[ok]; yy = yy[ok]; dist = np.abs(ids[z[ok]] - ids[qi]).astype(float)
                out.at[j, f"r{rad}_{typ}_resmed"] = float(np.nanmedian(rr))
                # Inverse-distance weighted mean is useful for near AOIs.
                w = 1.0 / np.maximum(1.0, dist)
                out.at[j, f"r{rad}_{typ}_resmean"] = float(np.sum(w * rr) / np.sum(w))
                out.at[j, f"r{rad}_{typ}_rawmed"] = float(np.nanmedian(yy - prof[z[ok]]))
                out.at[j, f"r{rad}_{typ}_n"] = int(ok.sum())
                for ss in ("s2", "landsat", "modis"):
                    zs = z[ok][src[z[ok]] == ss]
                    if len(zs): out.at[j, f"r{rad}_{typ}_{ss}_resmed"] = float(np.nanmedian(res[zs]))
    # Preserve direct profile values for diagnostics.
    out["query_profile"] = prof[qidx]
    out["query_profile_src"] = prof_src[qidx]
    return out


def _parts():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    routes = load_route(); parts = []
    for seed in SEEDS:
        f, mask = _mask_private(pr, seed)
        # Keep side-car truth before masking, and append train observations.
        tr0 = tr.copy(); tr0["_truth"] = pd.to_numeric(tr0.primary_ndvi, errors="coerce")
        tr0["_hidden"] = False
        f["_truth"] = pd.to_numeric(f.primary_ndvi, errors="coerce")
        f["_hidden"] = mask
        combo = pd.concat([tr0, f], ignore_index=True, sort=False)
        combo[DATE] = pd.to_datetime(combo[DATE])
        known = combo.primary_ndvi.notna().to_numpy(bool) & ~combo._hidden.to_numpy(bool)
        # Query rows are held rows only (actual organiser gaps are context,
        # never scored in this private-like audit).
        qidx = np.flatnonzero(np.r_[np.zeros(len(tr), bool), mask])
        q = routes[routes.seed.astype(int).eq(int(seed))].copy()
        q[DATE] = pd.to_datetime(q[DATE]); q = q.sort_values([ID, DATE]).reset_index(drop=True)
        q["route_base"] = policy_pred(q, "crop_hier_n1_p67", "cohort_year_dist")
        q["truth"] = q["truth"].astype(float)
        ft = _peer_features(combo, known, qidx, width=24)
        # Match feature rows by key; combo query order equals private sorted
        # order used by _mask_private, while route rows have same keys.
        qkeys = f.loc[mask, [ID, DATE]].copy(); qkeys[DATE] = pd.to_datetime(qkeys[DATE])
        ft = qkeys.reset_index(drop=True).join(ft.drop(columns=[ID, DATE], errors="ignore"))
        z = q.merge(ft, on=[ID, DATE], how="left", validate="one_to_one")
        z["seed"] = int(seed); parts.append(z)
    return parts


def fit_alpha(train, col):
    x = train[col].to_numpy(float); r = train.truth.to_numpy(float) - train.route_base.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(r)
    den = float(np.dot(x[ok], x[ok])) if ok.any() else 0.0
    return float(np.clip(np.dot(x[ok], r[ok]) / den, -1.0, 1.0)) if den > 1e-10 else 0.0


def run():
    parts = _parts(); allrows = []
    # Evaluate feature grid with leave-mask-out coefficients and fixed robust
    # weights.  Keep rows compact (metrics only; full feature table is useful
    # for later work and saved separately).
    feature_cols = []
    for rad in RADII:
        for typ in ("all", "crop"):
            feature_cols += [f"r{rad}_{typ}_resmed", f"r{rad}_{typ}_resmean", f"r{rad}_{typ}_rawmed"]
            for ss in ("s2", "landsat", "modis"): feature_cols.append(f"r{rad}_{typ}_{ss}_resmed")
    feature_cols += ["date_resmed", "crop_resmed", "date_resmean", "crop_resmean"]
    for i, test in enumerate(parts):
        train = pd.concat([p for j, p in enumerate(parts) if j != i], ignore_index=True)
        for col in feature_cols:
            a = fit_alpha(train, col); x = test[col].to_numpy(float); b = test.route_base.to_numpy(float); y = test.truth.to_numpy(float)
            good = np.isfinite(x); p = np.clip(b + a * np.nan_to_num(x, nan=0.0), -0.5, 1.2)
            allrows.append({"held_seed":int(test.seed.iloc[0]), "feature":col, "alpha":a,
                            "n":len(test), "coverage":float(good.mean()), "rmse":rmse(y,p),
                            "base_rmse":rmse(y,b), "delta":rmse(y,p)-rmse(y,b)})
    res = pd.DataFrame(allrows)
    res.to_csv(R / "local_peer_residual_v1_results.csv", index=False, float_format="%.10f")
    # Aggregate pooled masks with per-mask LOO coefficients.
    agg = res.groupby("feature", as_index=False).apply(lambda g: pd.Series({
        "n": int(g.n.sum()), "rmse_pooled": float(np.sqrt(np.average(g.rmse**2, weights=g.n))),
        "base_rmse_pooled": float(np.sqrt(np.average(g.base_rmse**2, weights=g.n))),
        "delta_pooled": float(np.sqrt(np.average(g.rmse**2, weights=g.n))-np.sqrt(np.average(g.base_rmse**2, weights=g.n))),
        "coverage_mean": float(np.average(g.coverage, weights=g.n)), "wins": int((g.rmse < g.base_rmse).sum())
    }), include_groups=False).reset_index(drop=True).sort_values("rmse_pooled")
    agg.to_csv(R / "local_peer_residual_v1_aggregate.csv", index=False, float_format="%.10f")
    # Save a compact feature sidecar for strongest variants, including truth
    # only because this is a local audit (never used in production inference).
    keep = [ID, DATE, "seed", "truth", "route_base", "year", "cohort", "near_dist"] + feature_cols
    # year/cohort may be present in route rows; avoid duplicate names.
    keep = list(dict.fromkeys([c for c in keep if c in pd.concat(parts, ignore_index=True).columns]))
    pd.concat(parts, ignore_index=True)[keep].to_csv(R / "local_peer_residual_v1_features.csv", index=False, float_format="%.8f")
    lines = ["# Local same-date peer residual audit v1", "", "Visible train + unmasked private rows build AOI seasonal profiles and peer residuals. Coefficients are leave-mask-out across seeds 0,1,2,70404.", "", agg.head(50).to_string(index=False), "", "No output candidate overwritten."]
    (R / "local_peer_residual_v1_report.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print(agg.head(40).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()

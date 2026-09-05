"""Leakage-safe post-correction audit for source-expert route v2.

Uses only the already persisted independent private-like mask rows.  For each
held-out mask, correction parameters are fit on the other two masks and then
applied to the held-out rows.  ``true_src`` is retained only for diagnostics;
all candidate corrections use observable keys (year/cohort/near distance,
prediction disagreement and date).  No existing candidate is overwritten.
"""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
INFILE = R / "source_expert_route_v2_rows.csv"
OUT_ROWS = R / "source_expert_route_v2_postcorr_rows_20260905.csv"
OUT_METRICS = R / "source_expert_route_v2_postcorr_metrics_20260905.csv"
OUT_GRID = R / "source_expert_route_v2_postcorr_grid_20260905.csv"
OUT_REPORT = ROOT / "reports" / "source_expert_route_v2_postcorr_report_20260905.md"

ID, DATE = "anon_polygon_id", "date"
BASE = "baseline"
EXPERT = "blend_crop_hier_n1_p67_0.40"
SEEDS = (0, 1, 70404)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features available at inference from the persisted audit rows."""
    x = pd.DataFrame(index=df.index)
    x["year_group"] = np.where(df.year.to_numpy(int) == 2025, "y2025", "history")
    x["cohort"] = df.cohort.astype(str).to_numpy()
    nd = df.near_dist.to_numpy(float)
    x["dist_group"] = pd.cut(
        nd, [-np.inf, 2, 8, 16, np.inf], labels=["near", "mid", "far", "none"],
    ).astype(str)
    # The expert disagreement is directly observable at inference.
    b = df[BASE].to_numpy(float); e = df[EXPERT].to_numpy(float)
    if "_delta_bin" in df:
        x["delta_abs_bin"] = df["_delta_bin"].to_numpy()
    else:
        x["delta_abs_bin"] = pd.qcut(np.abs(e - b), 4, labels=False, duplicates="drop")
    # Calendar phase is observable; coarse bins avoid high-variance tiny groups.
    dt = pd.to_datetime(df[DATE]); x["doy_bin"] = (dt.dt.dayofyear // 30).astype(int).to_numpy()
    # Stable AOI numeric bins are used only in a deliberately strongly shrunk
    # diagnostic family, never as exact per-AOI free parameters.
    anum = pd.to_numeric(df[ID].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(-1)
    x["aoi_bin"] = (anum // 10).astype(int).to_numpy()
    x["aoi_id"] = df[ID].astype(str).to_numpy()
    x["aoi_year_group"] = x["aoi_id"].astype(str) + "_" + x["year_group"].astype(str)
    x["aoi_dist_group"] = x["aoi_id"].astype(str) + "_" + x["dist_group"].astype(str)
    x["year_dist_group"] = x["year_group"].astype(str) + "_" + x["dist_group"].astype(str)
    x["base_bin"] = np.clip(np.floor(np.nan_to_num(b, nan=0.4) * 5).astype(int), -2, 6)
    return x


def _fit_global(cal: pd.DataFrame, bound=(0.0, 2.0), ridge=0.0) -> float:
    d = cal[EXPERT].to_numpy(float) - cal[BASE].to_numpy(float)
    r = cal.truth.to_numpy(float) - cal[BASE].to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(r)
    if not ok.any(): return 0.0
    # Minimise sum (r - a*d)^2; ridge shrinks towards zero (base).
    a = float(np.sum(d[ok] * r[ok]) / (np.sum(d[ok] ** 2) + ridge))
    return float(np.clip(a, *bound))


def _fit_group_map(cal: pd.DataFrame, key: str, global_a: float,
                   min_n=40, shrink=100.0, bound=(0.0, 2.0)) -> dict:
    """Empirical slope per group with James-Stein-like denominator shrinkage."""
    x = make_features(cal)
    out = {}
    for g, ix in x.groupby(key, dropna=False).groups.items():
        z = cal.loc[np.asarray(ix)]
        d = z[EXPERT].to_numpy(float) - z[BASE].to_numpy(float)
        r = z.truth.to_numpy(float) - z[BASE].to_numpy(float)
        ok = np.isfinite(d) & np.isfinite(r)
        n = int(ok.sum())
        if n < min_n: continue
        den = float(np.sum(d[ok] ** 2)); num = float(np.sum(d[ok] * r[ok]))
        # Prior equivalent to ``shrink`` observations at global slope.
        a = (num + shrink * global_a * max(den / max(n, 1), 1e-8)) / (den + shrink * max(den / max(n, 1), 1e-8))
        out[g] = float(np.clip(a, *bound))
    return out


def _fit_group_bias(cal: pd.DataFrame, key: str, global_b: float,
                    min_n=40, shrink_n=100.0, bound=(-0.05, 0.05)) -> dict:
    x = make_features(cal); out = {}
    for g, ix in x.groupby(key, dropna=False).groups.items():
        z = cal.loc[np.asarray(ix)]
        res = z.truth.to_numpy(float) - z[EXPERT].to_numpy(float)
        res = res[np.isfinite(res)]
        if len(res) < min_n: continue
        b = (float(res.sum()) + shrink_n * global_b) / (len(res) + shrink_n)
        out[g] = float(np.clip(b, *bound))
    return out


def apply_family(cal: pd.DataFrame, te: pd.DataFrame, family: str) -> np.ndarray:
    b = te[BASE].to_numpy(float); d = te[EXPERT].to_numpy(float) - b
    ga = _fit_global(cal, bound=(0.0, 2.0))
    if family == "global":
        return b + ga * d
    xcal, xte = make_features(cal), make_features(te)
    # Additive bias variants are handled before the generic ``group_`` slope
    # branch because their names also carry that prefix.
    if family == "global_bias":
        ga_b = float(np.mean(cal.truth.to_numpy(float)-cal[EXPERT].to_numpy(float)))
        return b + ga*d + np.clip(ga_b, -0.05, 0.05)
    if family == "group_bias_year":
        ga_b = float(np.mean(cal.truth.to_numpy(float)-cal[EXPERT].to_numpy(float)))
        mp = _fit_group_bias(cal, "year_group", ga_b, min_n=100, shrink_n=200.0)
        bb=np.asarray([mp.get(v,ga_b) for v in xte.year_group],float)
        return b + ga*d + bb
    if family == "group_bias_dist":
        ga_b = float(np.mean(cal.truth.to_numpy(float)-cal[EXPERT].to_numpy(float)))
        mp = _fit_group_bias(cal, "dist_group", ga_b, min_n=100, shrink_n=200.0)
        bb=np.asarray([mp.get(v,ga_b) for v in xte.dist_group],float)
        return b + ga*d + bb
    if family == "group_bias_cohort":
        ga_b = float(np.mean(cal.truth.to_numpy(float)-cal[EXPERT].to_numpy(float)))
        mp = _fit_group_bias(cal, "cohort", ga_b, min_n=100, shrink_n=200.0)
        bb=np.asarray([mp.get(v,ga_b) for v in xte.cohort],float)
        return b + ga*d + bb
    if family.startswith("group_"):
        key = family.split("_", 1)[1]
        mp = _fit_group_map(cal, key, ga, min_n=40, shrink=100.0)
        a = np.asarray([mp.get(v, ga) for v in xte[key]], float)
        return b + a * d
    if family.startswith("groupweak_"):
        key = family.split("_", 1)[1]
        mp = _fit_group_map(cal, key, ga, min_n=80, shrink=300.0)
        a = np.asarray([mp.get(v, ga) for v in xte[key]], float)
        return b + a * d
    if family == "bin_delta":
        # Confidence-adaptive alpha from disagreement quartiles.
        mp = _fit_group_map(cal, "delta_abs_bin", ga, min_n=100, shrink=150.0)
        a = np.asarray([mp.get(v, ga) for v in xte.delta_abs_bin], float)
        return b + a * d
    if family == "bin_dist":
        mp = _fit_group_map(cal, "dist_group", ga, min_n=100, shrink=150.0)
        a = np.asarray([mp.get(v, ga) for v in xte.dist_group], float)
        return b + a * d
    if family == "bin_year_dist":
        xx = xcal.copy(); yy = xte.copy()
        xx["combo"] = xx.year_group.astype(str) + "_" + xx.dist_group.astype(str)
        yy["combo"] = yy.year_group.astype(str) + "_" + yy.dist_group.astype(str)
        # Reuse helper by injecting key column into temporary data is easiest.
        cc = cal.copy(); tt = te.copy(); cc["_combo"] = xx.combo.values; tt["_combo"] = yy.combo.values
        # local helper sees make_features and cannot use injected key, so fit manually
        mp = {}
        for g, ix in xx.groupby("combo", dropna=False).groups.items():
            z = cal.loc[np.asarray(ix)]; dz=z[EXPERT].to_numpy(float)-z[BASE].to_numpy(float); rz=z.truth.to_numpy(float)-z[BASE].to_numpy(float); ok=np.isfinite(dz)&np.isfinite(rz)
            if ok.sum()<100: continue
            den=float(np.sum(dz[ok]**2)); num=float(np.sum(dz[ok]*rz[ok])); mp[g]=float(np.clip((num+150*ga*max(den/ok.sum(),1e-8))/(den+150*max(den/ok.sum(),1e-8)),0,2))
        a=np.asarray([mp.get(v,ga) for v in yy.combo],float); return b+a*d
    if family == "aoi_shrink":
        mp = _fit_group_map(cal, "aoi_id", ga, min_n=80, shrink=1000.0)
        a = np.asarray([mp.get(v, ga) for v in xte.aoi_id], float)
        return b + a*d
    if family == "aoi_year_shrink":
        mp = _fit_group_map(cal, "aoi_year_group", ga, min_n=60, shrink=1000.0)
        a = np.asarray([mp.get(v, ga) for v in xte.aoi_year_group], float)
        return b + a*d
    if family == "aoi_dist_shrink":
        mp = _fit_group_map(cal, "aoi_dist_group", ga, min_n=60, shrink=1000.0)
        a = np.asarray([mp.get(v, ga) for v in xte.aoi_dist_group], float)
        return b + a*d
    if family == "base_bin":
        mp = _fit_group_map(cal, "base_bin", ga, min_n=100, shrink=200.0)
        a = np.asarray([mp.get(v, ga) for v in xte.base_bin], float)
        return b + a*d
    if family == "delta_dist":
        xx=xcal.copy(); yy=xte.copy()
        xx["combo"]=xx.delta_abs_bin.astype(str)+"_"+xx.dist_group.astype(str)
        yy["combo"]=yy.delta_abs_bin.astype(str)+"_"+yy.dist_group.astype(str)
        mp={}
        for g,ix in xx.groupby("combo",dropna=False).groups.items():
            z=cal.loc[np.asarray(ix)]
            dz=z[EXPERT].to_numpy(float)-z[BASE].to_numpy(float)
            rz=z.truth.to_numpy(float)-z[BASE].to_numpy(float)
            ok=np.isfinite(dz)&np.isfinite(rz)
            if ok.sum()<100: continue
            den=float(np.sum(dz[ok]**2)); num=float(np.sum(dz[ok]*rz[ok]))
            prior=max(den/ok.sum(),1e-8)
            mp[g]=float(np.clip((num+250*ga*prior)/(den+250*prior),0,2))
        a=np.asarray([mp.get(v,ga) for v in yy.combo],float)
        return b+a*d
    if family == "aoi_bias":
        gb=float(np.mean(cal.truth.to_numpy(float)-cal[EXPERT].to_numpy(float)))
        mp=_fit_group_bias(cal,"aoi_id",gb,min_n=80,shrink_n=500.0)
        bb=np.asarray([mp.get(v,gb) for v in xte.aoi_id],float)
        return b+ga*d+bb
    if family == "aoi_year_bias":
        gb=float(np.mean(cal.truth.to_numpy(float)-cal[EXPERT].to_numpy(float)))
        mp=_fit_group_bias(cal,"aoi_year_group",gb,min_n=60,shrink_n=500.0)
        bb=np.asarray([mp.get(v,gb) for v in xte.aoi_year_group],float)
        return b+ga*d+bb
    raise ValueError(family)


def main() -> None:
    df = pd.read_csv(INFILE)
    df[DATE] = pd.to_datetime(df[DATE])
    # Ensure qcut bins are globally defined and stable across folds.
    # Recompute manually from all rows, then use these persisted helper values.
    absd = np.abs(df[EXPERT] - df[BASE]); edges = np.nanquantile(absd, [0,.25,.5,.75,1]); edges=np.maximum.accumulate(edges); edges[-1] += 1e-12
    df["_delta_bin"] = np.clip(np.searchsorted(edges[1:-1], absd, side="right"),0,3)
    # monkey patch make_features's qcut instability by retaining enough rows;
    # qcut on each fold is close but this fixed column is used below manually.
    families = ["base", "global", "group_year_group", "group_cohort", "group_dist_group", "groupweak_year_group", "groupweak_cohort", "groupweak_dist_group", "bin_delta", "bin_dist", "bin_year_dist", "aoi_shrink", "aoi_year_shrink", "aoi_dist_shrink", "base_bin", "delta_dist", "aoi_bias", "aoi_year_bias", "global_bias", "group_bias_year", "group_bias_dist", "group_bias_cohort"]
    rows=[]; metrics=[]; grids=[]
    for te_seed in SEEDS:
        te=df[df.seed==te_seed].copy(); cal=df[df.seed!=te_seed].copy()
        for fam in families:
            if fam=="base": pred=te[EXPERT].to_numpy(float)
            else: pred=apply_family(cal,te,fam)
            rr=te.truth.to_numpy(float)
            rec=te[[ID,DATE,"seed","truth","year","cohort","near_dist","true_src",BASE,EXPERT]].copy(); rec["family"]=fam; rec["pred"]=pred; rows.append(rec)
            metrics.append({"test_seed":te_seed,"family":fam,"n":len(te),"rmse":rmse(rr,pred),"bias":float(np.mean(pred-rr)),"mae":float(np.mean(np.abs(pred-rr)))})
        # Oracle in-fold slope only diagnostic, not used for recommendation.
        for a in np.arange(0,2.001,.025):
            p=te[BASE].to_numpy(float)+a*(te[EXPERT].to_numpy(float)-te[BASE].to_numpy(float)); grids.append({"test_seed":te_seed,"alpha":a,"rmse":rmse(te.truth.to_numpy(float),p)})
    outrows=pd.concat(rows,ignore_index=True); outrows.to_csv(OUT_ROWS,index=False,float_format="%.9f")
    met=pd.DataFrame(metrics); met.to_csv(OUT_METRICS,index=False,float_format="%.9f")
    pooled=[]
    for fam,g in outrows.groupby("family"):
        pooled.append({"family":fam,"n":len(g),"pooled_rmse":rmse(g.truth.to_numpy(float),g.pred.to_numpy(float)),"per_seed":";".join(f"{s}:{rmse(z.truth.to_numpy(float),z.pred.to_numpy(float)):.6f}" for s,z in g.groupby("seed"))})
    pool=pd.DataFrame(pooled).sort_values("pooled_rmse"); pool.to_csv(OUT_GRID,index=False,float_format="%.9f")
    # Best alpha per heldout seed as a stability diagnostic.
    gd=pd.DataFrame(grids); best=gd.loc[gd.groupby("test_seed").rmse.idxmin()].sort_values("test_seed")
    report=["# Source-expert route v2 post-correction audit (2026-09-05)","","Input: `research/source_expert_route_v2_rows.csv`; independent masks 0, 1, 70404.","For each test seed, parameters fit on the other two masks only. Features use year/cohort/near distance/prediction disagreement/calendar; `true_src` is evaluation-only.","","## Pooled LOO results", "", pool.to_string(index=False),"","## Per-seed LOO metrics","",met.to_string(index=False),"","## In-sample alpha diagnostic (not deployable)","",best.to_string(index=False),"","Artifacts:",f"- `{OUT_ROWS.relative_to(ROOT).as_posix()}`",f"- `{OUT_METRICS.relative_to(ROOT).as_posix()}`",f"- `{OUT_GRID.relative_to(ROOT).as_posix()}`", "No existing candidate overwritten; no submission emitted."]
    OUT_REPORT.write_text("\n".join(report)+"\n",encoding="utf-8")
    print(pool.to_string(index=False)); print("\nLOO:\n",met.to_string(index=False)); print("\nOracle alpha:\n",best.to_string(index=False))


if __name__ == "__main__": main()

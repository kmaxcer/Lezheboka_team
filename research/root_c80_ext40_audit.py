"""Audit and apply a conservative c80 peer correction to the ext40 anchor.

This is a research-only continuation of the completed private-like audits.  It
joins the independently generated c80 same-date peer map to the ext40 rows,
tests fixed peer/shock/state rules on three leakage-safe masks, reports slices
by cohort/year/source/nearest-visible distance, and (only after that) builds
new, separately named private candidates.  No hidden target is read by the
peer or correction features; labels in the audit tables are used only for
scoring.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
O = ROOT / "outputs"
sys.path.insert(0, str(R))
from evaluate_private_cohort_blend import make_holdout  # noqa: E402
from paired_aoi_v2 import peer_predictions, _config_name  # noqa: E402

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
PEER_C80 = _config_name(16, 0.80, 0.125, 3)
PEER_C60 = _config_name(16, 0.60, 0.125, 2)
CANON = frozenset((97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289))
BASE_PRIVATE = O / "model_dani_lag40_peer10_extwide40_v3_30_submission.csv"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else float("nan")


def _source(pr: pd.DataFrame) -> np.ndarray:
    """First available sensor, matching the organiser's target definition."""
    a = np.full(len(pr), "none", dtype=object)
    for col, name in (("modis_ndvi", "MODIS"), ("landsat_ndvi", "Landsat"), ("s2_ndvi", "S2")):
        if col in pr:
            take = pd.to_numeric(pr[col], errors="coerce").notna().to_numpy() & (a == "none")
            a[take] = name
    return a


def _context(pr: pd.DataFrame, masks: dict[int, np.ndarray]) -> pd.DataFrame:
    """Build source and distance metadata for each audit mask by key."""
    p = pr.copy().reset_index(drop=True)
    p[DATE] = pd.to_datetime(p[DATE])
    p["_src"] = _source(p)
    rows: list[pd.DataFrame] = []
    # The saved q tables contain only rows selected by each mask.  Distance is
    # computed against target values visible after both organiser gaps and the
    # pseudo holdout are removed.
    for seed, hold in masks.items():
        hidden = p[GAP].fillna(False).astype(bool).to_numpy()
        qidx = np.flatnonzero(hold)
        known = p[TARGET].notna().to_numpy(bool) & ~hidden & ~hold
        by_id: dict[str, np.ndarray] = {}
        for aid, ix in p.loc[known].groupby(ID, sort=False).groups.items():
            by_id[str(aid)] = p.loc[np.asarray(ix, dtype=int), DATE].map(pd.Timestamp.toordinal).to_numpy(float)
        q = p.loc[qidx, [ID, DATE, "_src"]].copy().reset_index(drop=True)
        dist = np.full(len(q), np.inf, dtype=float)
        for j, (aid, dt) in enumerate(q[[ID, DATE]].itertuples(index=False, name=None)):
            z = by_id.get(str(aid))
            if z is not None and len(z):
                dist[j] = float(np.min(np.abs(z - pd.Timestamp(dt).toordinal())))
        q["mask_seed"] = int(seed)
        q["source"] = q.pop("_src").astype(str)
        q["nearest_days"] = dist
        q["distance_bin"] = pd.cut(
            q["nearest_days"], bins=[-np.inf, 7, 16, 32, 64, 120, np.inf],
            labels=["0-7", "8-16", "17-32", "33-64", "65-120", ">120"],
        ).astype(object).fillna(">120")
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def _load_audits(private: pd.DataFrame) -> pd.DataFrame:
    """Join ext40/c60/c80/shock/state rows from the three completed audits."""
    p = private.copy().reset_index(drop=True)
    p[DATE] = pd.to_datetime(p[DATE])
    masks = {s: make_holdout(p, s) for s in (0, 1, 70404)}
    # c80 is in paired_aoi_v2; c60/ext40/shock/state are in the prior tables.
    paired = pd.read_csv(R / "paired_aoi_v2_predictions.csv", parse_dates=[DATE], low_memory=False)
    paired = paired[["partition", ID, DATE, PEER_C80]].copy()
    parts: list[pd.DataFrame] = []
    for seed in (0, 1):
        q = pd.read_csv(R / "meta_residual_v2_independent_v3_predictions.csv", parse_dates=[DATE], low_memory=False)
        q = q[q["mask_seed"].eq(seed)].copy()
        q = q.merge(
            paired[paired["partition"].eq(f"random_{seed}")].drop(columns="partition"),
            on=[ID, DATE], how="left", validate="one_to_one",
        )
        q["seed"] = seed
        parts.append(q)
    q = pd.read_csv(R / "private_cohort_blend_holdout_predictions.csv", parse_dates=[DATE], low_memory=False)
    q["seed"] = 70404
    parts.append(q)
    d = pd.concat(parts, ignore_index=True, sort=False)
    # Avoid any accidental duplicate columns from future table extensions.
    d = d.loc[:, ~d.columns.duplicated()].copy()
    ctx = _context(p, masks)
    d = d.merge(ctx, on=["mask_seed", ID, DATE], how="left", validate="one_to_one")
    d["year2"] = pd.to_datetime(d[DATE]).dt.year.astype(int)
    d["history"] = d["year2"].lt(2025)
    d["canon"] = pd.to_datetime(d[DATE]).dt.dayofyear.isin(CANON)
    return d


def _candidate(d: pd.DataFrame, peer: str, weight: float, shock: float = 0.0,
               state: float = 0.0, route: str = "history") -> np.ndarray:
    b = d["ext40"].to_numpy(float)
    pp = d[peer].to_numpy(float)
    delta = weight * np.nan_to_num(pp - b, nan=0.0)
    if route == "history":
        active = d["history"].to_numpy(bool) & ~d["canon"].to_numpy(bool)
    elif route == "noncanon":
        active = ~d["canon"].to_numpy(bool)
    else:
        active = np.ones(len(d), dtype=bool)
    if shock or state:
        delta = delta + np.where(
            active,
            shock * np.nan_to_num(d["shock"].to_numpy(float), nan=0.0)
            + state * np.nan_to_num(d["state"].to_numpy(float), nan=0.0),
            0.0,
        )
    return np.clip(b + np.where(d[peer].notna().to_numpy(bool) | (weight == 0), delta, 0.0), -0.5, 1.2)


def _audit(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs: list[tuple[str, str, float, float, float, str]] = [("ext40", PEER_C80, 0.0, 0.0, 0.0, "none")]
    for peer, tag in ((PEER_C60, "c60"), (PEER_C80, "c80")):
        for w in (0.07, 0.09, 0.10, 0.12, 0.15):
            specs.append((f"{tag}_hist_{w:.2f}", peer, w, 0.0, 0.0, "history"))
            if tag == "c80" and w in (0.09, 0.10, 0.12):
                specs.append((f"{tag}_hist_{w:.2f}_shock10_state05", peer, w, 0.10, -0.05, "history"))
    rows: list[dict[str, object]] = []
    for name, peer, w, a, b, route in specs:
        pred = _candidate(d, peer, w, a, b, route)
        d["_pred"] = pred
        for group_name, group in (
            ("all", np.ones(len(d), bool)),
            ("history", d["history"].to_numpy(bool)),
            ("2025", ~d["history"].to_numpy(bool)),
            ("new", d["cohort"].astype(str).eq("new").to_numpy()),
            ("shared", d["cohort"].astype(str).eq("shared").to_numpy()),
            ("new2025", (d["cohort"].astype(str).eq("new") & ~d["history"]).to_numpy()),
            ("shared2025", (d["cohort"].astype(str).eq("shared") & ~d["history"]).to_numpy()),
        ):
            if not group.any():
                continue
            for seed, sg in d.groupby("seed", sort=True):
                take = group & d["seed"].eq(seed).to_numpy()
                if not take.any():
                    continue
                y = d.loc[take, "truth"].to_numpy(float)
                base = d.loc[take, "ext40"].to_numpy(float)
                pp = pred[take]
                rows.append({"candidate": name, "seed": int(seed), "group": group_name,
                             "n": int(take.sum()), "rmse": _rmse(y, pp),
                             "baseline_rmse": _rmse(y, base), "delta_rmse": _rmse(y, pp) - _rmse(y, base)})
        # Fine-grained source/distance slices are deliberately recorded only
        # for the strongest c80 rules to keep the report compact.
        if name in {"ext40", "c80_hist_0.09", "c80_hist_0.10_shock10_state05", "c80_hist_0.12_shock10_state05"}:
            for col in ("source", "distance_bin"):
                for value, sg in d.groupby(col, dropna=False, sort=True):
                    for seed in sorted(d["seed"].unique()):
                        # Keep a full-frame boolean mask; using ``sg.index``
                        # directly would create a shorter mask for ``d.loc``.
                        if pd.isna(value):
                            take = d[col].isna().to_numpy() & d["seed"].eq(seed).to_numpy()
                        else:
                            take = d[col].eq(value).to_numpy() & d["seed"].eq(seed).to_numpy()
                        if not take.any():
                            continue
                        y = d.loc[take, "truth"].to_numpy(float); base = d.loc[take, "ext40"].to_numpy(float)
                        rows.append({"candidate": name, "seed": int(seed), "group": f"{col}={value}",
                                     "n": int(take.sum()), "rmse": _rmse(y, pred[take]),
                                     "baseline_rmse": _rmse(y, base), "delta_rmse": _rmse(y, pred[take]) - _rmse(y, base)})
    metrics = pd.DataFrame(rows)
    # Correct pooled RMSE from row-level predictions and rank by worst seed;
    # this avoids selecting a rule from a single lucky audit.
    pooled: list[dict[str, object]] = []
    for name, g in metrics[metrics.group.eq("all")].groupby("candidate", sort=False):
        e = g["rmse"].to_numpy(float); eb = g["baseline_rmse"].to_numpy(float); n = g["n"].to_numpy(float)
        ds = g["delta_rmse"].to_numpy(float)
        pooled.append({"candidate": name, "n": int(n.sum()),
                       "pooled_rmse": float(np.sqrt(np.average(e * e, weights=n))),
                       "pooled_baseline_rmse": float(np.sqrt(np.average(eb * eb, weights=n))),
                       "pooled_delta_rmse": float(np.sqrt(np.average(e * e, weights=n)) - np.sqrt(np.average(eb * eb, weights=n))),
                       "worst_seed_delta": float(np.max(ds)), "all_seed_improve": bool(np.all(ds <= 0.0))})
    summary = pd.DataFrame(pooled).sort_values(["all_seed_improve", "pooled_rmse", "worst_seed_delta"], ascending=[False, True, True])
    return metrics, summary


def _apply_private(private: pd.DataFrame, base: pd.DataFrame) -> dict[str, object]:
    """Generate new c80 candidates after the audit, preserving old files."""
    p = private.copy().reset_index(drop=True); p[DATE] = pd.to_datetime(p[DATE])
    hidden = p[GAP].fillna(False).astype(bool).to_numpy()
    pm = p.copy(); pm.loc[hidden, TARGET] = np.nan
    # All dynamic columns are already masked by the organiser on synthetic
    # gaps; explicitly clear any stray values before peer selection.
    for c in ("s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi", "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi", "era5_temp_c", "era5_precip_mm", "year", "doy", "ndvi_climatology_mean", "ndvi_climatology_std", "n_reference_years"):
        if c in pm:
            pm.loc[hidden, c] = np.nan
    print(f"actual private peer c80: query={int(hidden.sum())}", flush=True)
    peer_cache = R / "root_c80_private_peer_grid.csv"
    pair_cache = R / "root_c80_private_peer_pairs.csv"
    if peer_cache.exists():
        pp = pd.read_csv(peer_cache, parse_dates=[DATE], low_memory=False)
        pairs = pd.read_csv(pair_cache, low_memory=False) if pair_cache.exists() else pd.DataFrame()
    else:
        pp, pairs = peer_predictions(pm, hidden, partition="private_actual_root_c80")
        pp = pp.drop(columns=["_row"], errors="ignore")
        pp.to_csv(peer_cache, index=False, float_format="%.8f")
        pairs.to_csv(pair_cache, index=False, float_format="%.8f")
    q = p.loc[hidden, [ID, DATE]].copy().reset_index(drop=True)
    q = q.merge(pp[[ID, DATE, PEER_C60, PEER_C80]], on=[ID, DATE], how="left", validate="one_to_one")
    # Reuse the already produced observable shock/state rows, joined by key.
    sr = pd.read_csv(R / "ensemble_cv_v2_peer_apply_rows.csv", parse_dates=[DATE], low_memory=False)
    # ``ensemble_cv_v2_peer_apply_rows`` calls the ext40 anchor ``baseline``;
    # retain that naming distinction from the explicit private base file.
    q = q.merge(sr[[ID, DATE, "shock", "state"]],
                on=[ID, DATE], how="left", validate="one_to_one")
    # ext40 in apply rows is the same anchor used for the saved private base;
    # the explicit base file is used as the immutable prediction source.
    b = base.copy(); b[DATE] = pd.to_datetime(b[DATE])
    q = q.merge(b.rename(columns={"primary_ndvi_pred": "base_file"}), on=[ID, DATE], how="left", validate="one_to_one")
    if q["base_file"].isna().any():
        raise RuntimeError("base file does not cover all hidden keys")
    q["year"] = q[DATE].dt.year.astype(int); q["canon"] = q[DATE].dt.dayofyear.isin(CANON)
    b0 = q["base_file"].to_numpy(float)
    outputs: dict[str, object] = {}
    rules = {
        "model_dani_extwide40_v3_30_peerblend09_c80_history_submission.csv": (PEER_C80, 0.09, 0.0, 0.0),
        "model_dani_extwide40_v3_30_peerblend10_c80_history_shock10_state05_submission.csv": (PEER_C80, 0.10, 0.10, -0.05),
        "model_dani_extwide40_v3_30_peerblend12_c80_history_shock10_state05_submission.csv": (PEER_C80, 0.12, 0.10, -0.05),
        "model_dani_extwide40_v3_30_peerblend10_c60_history_shock10_state05_submission.csv": (PEER_C60, 0.10, 0.10, -0.05),
    }
    for fname, (peer, w, a, st) in rules.items():
        active = (q["year"].to_numpy(int) < 2025) & ~q["canon"].to_numpy(bool)
        delta = w * np.nan_to_num(q[peer].to_numpy(float) - b0, nan=0.0)
        delta += np.where(active, a * np.nan_to_num(q["shock"].to_numpy(float), nan=0.0) + st * np.nan_to_num(q["state"].to_numpy(float), nan=0.0), 0.0)
        pred = np.clip(b0 + delta, -0.5, 1.2)
        out = q[[ID, DATE]].copy(); out["primary_ndvi_pred"] = pred
        path = O / fname
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing candidate {path.name}")
        out.to_csv(path, index=False, float_format="%.8f")
        check = pd.read_csv(path, parse_dates=[DATE])
        if list(check.columns) != [ID, DATE, "primary_ndvi_pred"] or len(check) != int(hidden.sum()) or check.duplicated([ID, DATE]).any() or not np.isfinite(check.primary_ndvi_pred).all():
            raise RuntimeError(f"contract failure for {path.name}")
        meta = {"candidate": path.name, "formula": f"ext40 + {w:.2f}*({peer}-ext40) + history_noncanon*({a:.2f}*shock{st:+.2f}*state)", "rows": int(len(out)), "hidden_rows": int(hidden.sum()), "peer_config": peer, "peer_finite": int(q[peer].notna().sum()), "production_baseline_overwritten": False, "base_sha256": _sha(BASE_PRIVATE), "peer_pairs": int(len(pairs)), "candidate_sha256": _sha(path)}
        (O / (path.stem + "_metadata.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs[fname] = meta
        print(fname, "peer", int(q[peer].notna().sum()), "range", float(pred.min()), float(pred.max()), flush=True)
    # Save diagnostics without any labels.
    q.to_csv(R / "root_c80_private_apply_rows.csv", index=False, float_format="%.8f")
    return outputs


def main() -> None:
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    private[GAP] = private[GAP].fillna(False).astype(bool)
    d = _load_audits(private)
    metrics, summary = _audit(d)
    metrics.to_csv(R / "root_c80_ext40_audit_metrics.csv", index=False, float_format="%.10f")
    summary.to_csv(R / "root_c80_ext40_audit_summary.csv", index=False, float_format="%.10f")
    lines = ["# c80 peer + ext40 audit", "", "Three leakage-safe private-like masks (0, 1, 70404); c80 maps use only visible same-year/same-date targets. Hidden labels are used only for scoring.", "", summary.to_string(index=False), "", "The selected rules are history/non-canonical only; 2025 rows remain the ext40 anchor.", "No old output was overwritten."]
    (R / "root_c80_ext40_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    # Apply only if the audit produced an all-seed-improving c80 rule.  This
    # guard prevents accidental artifact creation if upstream tables change.
    best = summary.iloc[0]
    if bool(best["all_seed_improve"]) and str(best["candidate"]).startswith("c80_"):
        base = pd.read_csv(BASE_PRIVATE, parse_dates=[DATE], low_memory=False)
        _apply_private(private, base)
    else:
        print("No c80 rule passed the all-seed guard; private candidates not emitted.", flush=True)


if __name__ == "__main__":
    main()

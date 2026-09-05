"""Leakage-safe same-date peer-AOI transfer experiment.

For each (target AOI, year), affine maps from other AOIs are fitted using only
currently visible same-date observations from that same year.  Candidate
quality is estimated by interleaved out-of-fold predictions on those visible
overlaps.  Hidden target values are side-car labels used only after all
predictions have been made.

This is a research-only script.  It consumes the already saved HGB/lag OOF
predictions and writes only ``research/paired_aoi_v2*`` artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402


ID = "anon_polygon_id"
DATE = "date"
TARGET = "primary_ndvi"
GAP = "is_synthetic_gap"

DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi",
    "era5_temp_c", "era5_precip_mm", "year", TARGET, "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


@dataclass(frozen=True)
class PairModel:
    intercept: float
    slope: float
    n: int
    corr: float
    cv_rmse: float
    cv_mae: float


def _robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Small deterministic Huber-IRLS affine fit with conservative bounds."""
    good = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[good], float)
    y = np.asarray(y[good], float)
    if len(x) < 4 or np.nanstd(x) < 1e-8:
        return float(np.nanmedian(y) if len(y) else 0.35), 0.0
    design = np.c_[np.ones(len(x)), x]
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(6):
        resid = y - design @ coef
        center = float(np.median(resid))
        scale = 1.4826 * float(np.median(np.abs(resid - center))) + 1e-5
        u = np.abs(resid - center) / (1.5 * scale)
        weights = np.ones_like(u)
        far = u > 1.0
        weights[far] = 1.0 / u[far]
        sw = np.sqrt(weights)
        coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
    intercept = float(np.clip(coef[0], -0.40, 0.40))
    slope = float(np.clip(coef[1], 0.0, 2.50))
    return intercept, slope


def _pair_model(x: np.ndarray, y: np.ndarray, min_fit: int = 6) -> PairModel | None:
    good = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[good], float)
    y = np.asarray(y[good], float)
    n = len(x)
    if n < min_fit or np.std(x) < 1e-7 or np.std(y) < 1e-7:
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return None

    # Interleaved dates put all seasons into each training split and avoid
    # evaluating the affine map on the same rows used to fit it.
    n_splits = min(5, max(2, n // 5))
    fold_id = np.arange(n) % n_splits
    oof = np.full(n, np.nan, dtype=float)
    for fold in range(n_splits):
        valid = fold_id == fold
        train = ~valid
        if train.sum() < 4:
            continue
        a, b = _robust_affine(x[train], y[train])
        oof[valid] = a + b * x[valid]
    ok = np.isfinite(oof)
    if ok.sum() < max(4, n // 2):
        return None
    err = oof[ok] - y[ok]
    a, b = _robust_affine(x, y)
    return PairModel(
        intercept=a,
        slope=b,
        n=n,
        corr=corr,
        cv_rmse=float(np.sqrt(np.mean(err * err))),
        cv_mae=float(np.mean(np.abs(err))),
    )


def _config_name(min_n: int, min_corr: float, max_rmse: float, topk: int) -> str:
    return f"n{min_n}_c{int(round(min_corr * 100)):02d}_r{int(round(max_rmse * 1000)):03d}_k{topk}"


CONFIGS = [
    (min_n, min_corr, max_rmse, topk)
    for min_n in (8, 12, 16)
    for min_corr in (0.40, 0.60, 0.80)
    for max_rmse in (0.060, 0.080, 0.100, 0.125)
    for topk in (1, 2, 3)
]


def peer_predictions(
    frame: pd.DataFrame,
    query_mask: np.ndarray,
    *,
    partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return grid predictions and pair diagnostics without reading truth."""
    d = frame.copy()
    d[DATE] = pd.to_datetime(d[DATE])
    d["_calc_year"] = d[DATE].dt.year.astype(int)
    query_mask = np.asarray(query_mask, bool)
    # Strict guardrail: anything scored as hidden must be absent from all
    # values visible to pair selection/calibration.
    if d.loc[query_mask, TARGET].notna().any():
        raise AssertionError("query target leaked into peer frame")
    known = d[TARGET].notna().to_numpy(bool) & ~query_mask
    if GAP in d:
        known &= ~d[GAP].fillna(False).astype(bool).to_numpy()

    q = d.loc[query_mask, [ID, DATE]].copy().reset_index().rename(columns={"index": "_row"})
    config_names = [_config_name(*c) for c in CONFIGS]
    pred_matrix = np.full((len(q), len(CONFIGS)), np.nan, dtype=float)
    pair_rows: list[dict[str, object]] = []

    q["_qpos"] = np.arange(len(q), dtype=int)
    q["_calc_year"] = q[DATE].dt.year.astype(int)
    for year, qy in q.groupby("_calc_year", sort=False):
        yr = d["_calc_year"].eq(int(year)).to_numpy()
        obs = d.loc[known & yr, [DATE, ID, TARGET]]
        if obs.empty:
            continue
        pivot = obs.pivot_table(index=DATE, columns=ID, values=TARGET, aggfunc="first")
        ids = list(pivot.columns)
        for target_id, qt in qy.groupby(ID, sort=False):
            if target_id not in pivot.columns:
                continue
            y_target = pivot[target_id]
            models: dict[str, PairModel] = {}
            for peer_id in ids:
                if peer_id == target_id:
                    continue
                xy = pd.concat([pivot[peer_id], y_target], axis=1, keys=["x", "y"]).dropna()
                model = _pair_model(xy["x"].to_numpy(float), xy["y"].to_numpy(float))
                if model is None:
                    continue
                models[str(peer_id)] = model
                pair_rows.append({
                    "partition": partition,
                    "year": int(year),
                    "target_aoi": str(target_id),
                    "peer_aoi": str(peer_id),
                    "n": model.n,
                    "corr": model.corr,
                    "cv_rmse": model.cv_rmse,
                    "cv_mae": model.cv_mae,
                    "intercept": model.intercept,
                    "slope": model.slope,
                })
            if not models:
                continue

            # ``iterrows`` is deliberate here: pandas may rename leading-
            # underscore fields in ``itertuples``, which could silently break
            # alignment of peer predictions with query rows.
            for _, qrow in qt.iterrows():
                date = qrow[DATE]
                qpos = int(qrow["_qpos"])
                if date not in pivot.index:
                    continue
                row = pivot.loc[date]
                available: list[tuple[str, float, PairModel]] = []
                for peer_id, model in models.items():
                    if peer_id not in row.index:
                        continue
                    value = row[peer_id]
                    if np.isfinite(value):
                        available.append((peer_id, float(value), model))
                if not available:
                    continue
                for j, (min_n, min_corr, max_rmse, topk) in enumerate(CONFIGS):
                    candidates = [
                        (m.cv_rmse, -m.n, m.intercept + m.slope * value, m)
                        for _, value, m in available
                        if m.n >= min_n and m.corr >= min_corr and m.cv_rmse <= max_rmse
                    ]
                    if not candidates:
                        continue
                    candidates.sort(key=lambda z: (z[0], z[1]))
                    candidates = candidates[:topk]
                    vals = np.asarray([z[2] for z in candidates], float)
                    weights = np.asarray([
                        min(1.0, z[3].n / 24.0) / (z[3].cv_rmse ** 2 + 0.0009)
                        for z in candidates
                    ], float)
                    pred_matrix[qpos, j] = float(np.clip(np.average(vals, weights=weights), -0.2, 1.1))

    out = q[["_row", ID, DATE]].copy()
    for j, name in enumerate(config_names):
        out[name] = pred_matrix[:, j]
    pairs = pd.DataFrame(pair_rows).drop_duplicates(
        ["partition", "year", "target_aoi", "peer_aoi"], keep="last"
    ) if pair_rows else pd.DataFrame()
    return out, pairs


def _align_saved(keys: pd.DataFrame, path: Path, pred_col: str) -> np.ndarray:
    p = pd.read_csv(path)
    p[DATE] = pd.to_datetime(p[DATE])
    p = p[[ID, DATE, pred_col]].drop_duplicates([ID, DATE], keep="last")
    z = keys[[ID, DATE]].merge(p, on=[ID, DATE], how="left", validate="one_to_one")
    if z[pred_col].isna().any():
        raise RuntimeError(f"missing saved predictions in {path.name}: {z[pred_col].isna().sum()}")
    return z[pred_col].to_numpy(float)


def _random_mask(private: pd.DataFrame, seed: int, frac: float = 0.15) -> tuple[pd.DataFrame, np.ndarray]:
    """Byte-for-byte protocol equivalent to research/hgb_cv.py."""
    d = private.copy().sort_values([ID, DATE]).reset_index(drop=True)
    d[DATE] = pd.to_datetime(d[DATE])
    d["_truth"] = d[TARGET].astype(float)
    d[GAP] = False
    rng = np.random.default_rng(int(seed))
    mask = np.zeros(len(d), dtype=bool)
    pool = d[TARGET].notna()
    years = d[DATE].dt.year
    for _, ix in d.loc[pool].groupby([ID, years], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        n = max(1, int(round(float(frac) * len(ii))))
        mask[rng.choice(ii, size=min(n, len(ii)), replace=False)] = True
    for col in DYNAMIC:
        if col in d:
            d.loc[mask, col] = np.nan
    d.loc[mask, GAP] = True
    return d, mask


def _scenario_frames(train: pd.DataFrame, private: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame, np.ndarray, pd.DataFrame]]:
    exact_saved = pd.read_csv(RESEARCH / "exact_compare_preds.csv", parse_dates=[DATE])
    for year in (2019, 2020, 2021, 2022, 2023, 2024):
        frame, _ = make_fold(train, private, year)
        mask = frame[GAP].fillna(False).astype(bool).to_numpy()
        side = frame.loc[mask, [ID, DATE, "_truth"]].copy().reset_index(drop=True)
        saved = exact_saved.loc[exact_saved["year"].eq(year)].copy()
        side["hgb"] = _align_saved(side, RESEARCH / "exact_compare_preds.csv", "hgb")
        # The exact saved file has repeated AOI/date only across no partitions;
        # align through its year slice explicitly to retain one-to-one keys.
        for col in ("hgb", "lag_k16_d3"):
            pp = saved[[ID, DATE, col]].copy()
            z = side[[ID, DATE]].merge(pp, on=[ID, DATE], how="left", validate="one_to_one")
            if z[col].isna().any():
                raise RuntimeError(f"missing exact {col} for {year}")
            side[col if col != "lag_k16_d3" else "lag"] = z[col].to_numpy(float)
        yield f"exact_{year}", frame, mask, side

    for seed in (0, 1, 2):
        frame, mask = _random_mask(private, seed)
        side = frame.loc[mask, [ID, DATE, "_truth"]].copy().reset_index(drop=True)
        side["hgb"] = _align_saved(side, RESEARCH / f"hgb_cv_pred_seed{seed}.csv", "primary_ndvi_pred")
        side["lag"] = _align_saved(side, RESEARCH / f"teammate_sweep_postcorr_lag_random{seed}.csv", "primary_ndvi_pred")
        yield f"random_{seed}", frame, mask, side


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan


def main() -> None:
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    metric_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []

    for partition, frame, mask, side in _scenario_frames(train, private):
        print(partition, "n", int(mask.sum()), flush=True)
        peer, pairs = peer_predictions(frame, mask, partition=partition)
        pair_parts.append(pairs)
        z = side.merge(peer.drop(columns=["_row"]), on=[ID, DATE], how="left", validate="one_to_one")
        z["partition"] = partition
        z["family"] = "exact" if partition.startswith("exact") else "random"
        z["year"] = pd.to_datetime(z[DATE]).dt.year.astype(int)
        y = z["_truth"].to_numpy(float)
        bases = {
            "hgb": z["hgb"].to_numpy(float),
            "hgb_lag20": 0.8 * z["hgb"].to_numpy(float) + 0.2 * z["lag"].to_numpy(float),
            "hgb_lag30": 0.7 * z["hgb"].to_numpy(float) + 0.3 * z["lag"].to_numpy(float),
        }
        cohorts = {"all": np.ones(len(z), dtype=bool)}
        if partition.startswith("random"):
            cohorts["year_2025"] = z["year"].eq(2025).to_numpy()
            cohorts["history"] = z["year"].lt(2025).to_numpy()
        for cohort, take in cohorts.items():
            if not take.any():
                continue
            for base_name, base in bases.items():
                base_rmse = _rmse(y[take], base[take])
                metric_rows.append({
                    "partition": partition, "family": z["family"].iat[0], "cohort": cohort,
                    "base": base_name, "peer_config": "none", "peer_weight": 0.0,
                    "n": int(take.sum()), "peer_n": 0, "coverage": 0.0,
                    "rmse": base_rmse, "delta_rmse": 0.0,
                })
                for config in [_config_name(*c) for c in CONFIGS]:
                    pp = z[config].to_numpy(float)
                    covered = take & np.isfinite(pp)
                    for weight in (0.02, 0.05, 0.08, 0.10, 0.15):
                        pred = base.copy()
                        pred[covered] = (1.0 - weight) * base[covered] + weight * pp[covered]
                        score = _rmse(y[take], pred[take])
                        metric_rows.append({
                            "partition": partition, "family": z["family"].iat[0], "cohort": cohort,
                            "base": base_name, "peer_config": config, "peer_weight": weight,
                            "n": int(take.sum()), "peer_n": int(covered.sum()),
                            "coverage": float(covered.sum() / take.sum()),
                            "rmse": score, "delta_rmse": score - base_rmse,
                        })
        # Compact row-level artifact: baselines plus all peer configs.
        prediction_parts.append(z)
        print("  pairs", len(pairs), "peer coverage max", z[[c for c in z if c.startswith("n")]].notna().mean().max(), flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESEARCH / "paired_aoi_v2_metrics.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(RESEARCH / "paired_aoi_v2_predictions.csv", index=False)
    if pair_parts:
        pd.concat(pair_parts, ignore_index=True).to_csv(RESEARCH / "paired_aoi_v2_pairs.csv", index=False)

    # Pooled RMSE by family/cohort: aggregate squared errors correctly through
    # n-weighted RMSE^2 rather than averaging per-partition RMSE.
    agg = metrics.groupby(["family", "cohort", "base", "peer_config", "peer_weight"], as_index=False).apply(
        lambda g: pd.Series({
            "runs": len(g),
            "n": int(g["n"].sum()),
            "peer_n": int(g["peer_n"].sum()),
            "coverage": float(g["peer_n"].sum() / g["n"].sum()),
            "rmse": float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"]))),
            "baseline_rmse": float(np.sqrt(np.average((g["rmse"] - g["delta_rmse"]) ** 2, weights=g["n"]))),
            "mean_delta": float(np.average(g["delta_rmse"], weights=g["n"])),
            "improved_runs": int((g["delta_rmse"] < 0).sum()),
        }), include_groups=False
    ).reset_index(drop=True)
    agg["delta_rmse"] = agg["rmse"] - agg["baseline_rmse"]
    agg.to_csv(RESEARCH / "paired_aoi_v2_aggregate.csv", index=False)

    # Robust shortlist: rank candidates by their worst standardized delta over
    # exact/all, random/all, and random/2025.  A useful candidate should not
    # merely exploit one proxy family.
    scored = agg[agg["peer_config"].ne("none")].copy()
    key = ["base", "peer_config", "peer_weight"]
    views = []
    for family, cohort, label in (("exact", "all", "exact"), ("random", "all", "random"), ("random", "year_2025", "random2025")):
        part = scored[(scored.family == family) & (scored.cohort == cohort)].copy()
        part = part[key + ["rmse", "baseline_rmse", "delta_rmse", "coverage", "improved_runs", "runs"]]
        part = part.rename(columns={c: f"{label}_{c}" for c in part.columns if c not in key})
        views.append(part)
    shortlist = views[0]
    for view in views[1:]:
        shortlist = shortlist.merge(view, on=key, how="inner", validate="one_to_one")
    shortlist["worst_delta"] = shortlist[["exact_delta_rmse", "random_delta_rmse", "random2025_delta_rmse"]].max(axis=1)
    shortlist["mean_delta"] = shortlist[["exact_delta_rmse", "random_delta_rmse", "random2025_delta_rmse"]].mean(axis=1)
    shortlist["all_three_improve"] = shortlist["worst_delta"] < 0
    shortlist = shortlist.sort_values(["all_three_improve", "worst_delta", "mean_delta"], ascending=[False, True, True])
    shortlist.to_csv(RESEARCH / "paired_aoi_v2_shortlist.csv", index=False)

    best = shortlist.iloc[0]
    report = [
        "# Paired AOI v2 — leakage-safe evaluation",
        "",
        "Affine peer maps and peer ranking use only visible same-year, same-date target overlaps. "
        "Each pair's ranking error is interleaved out-of-fold; hidden labels are used only for final scoring.",
        "",
        f"Best robust row: base `{best['base']}`, peer `{best['peer_config']}`, weight {best['peer_weight']:.2f}.",
        "",
        f"- exact hidden-DOY: {best['exact_baseline_rmse']:.6f} -> {best['exact_rmse']:.6f} "
        f"(delta {best['exact_delta_rmse']:+.6f}, coverage {best['exact_coverage']:.1%})",
        f"- random private-like: {best['random_baseline_rmse']:.6f} -> {best['random_rmse']:.6f} "
        f"(delta {best['random_delta_rmse']:+.6f}, coverage {best['random_coverage']:.1%})",
        f"- random private-like 2025: {best['random2025_baseline_rmse']:.6f} -> {best['random2025_rmse']:.6f} "
        f"(delta {best['random2025_delta_rmse']:+.6f}, coverage {best['random2025_coverage']:.1%})",
        "",
        f"All three pooled proxies improve: `{bool(best['all_three_improve'])}`.",
        "No production file was changed.",
    ]
    (RESEARCH / "paired_aoi_v2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()

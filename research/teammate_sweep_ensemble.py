"""Private-like ensemble sweep for the Agropulse NDVI imputer.

This experiment is deliberately isolated under ``research/``.  It masks
observed rows in ``private_features.csv`` before building any features, then
compares the archive HistGradientBoosting model (including a small
random-state ensemble) with the source-aware local and lag-aware local
estimators.  Two masks are evaluated: a broad AOI/year mask and a 2025-only
mask, because the latter has the same year/shape as the real hidden shared-ID
rows.  No production ``outputs/model_dani_tuned*`` file is written.

Run from the repository root::

    .\\.venv\\Scripts\\python.exe research\\teammate_sweep_ensemble.py

The script writes only ``research/teammate_sweep_ensemble_*`` artefacts.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
ARCHIVE_SRC = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
sys.path.insert(0, str(ARCHIVE_SRC))
from agropulse.pipeline import (  # type: ignore  # noqa: E402
    FULL_FEATURES,
    build_features,
    create_model,
)

sys.path.insert(0, str(ROOT / "src"))
from infer import predict_private  # type: ignore  # noqa: E402
from infer_lag import predict_private_lag  # type: ignore  # noqa: E402


ID = "anon_polygon_id"
DATE = "date"
TARGET = "primary_ndvi"
GAP = "is_synthetic_gap"

# All fields that are hidden by the organizer on a synthetic row.  Keeping
# this list explicit prevents accidental target/sensor leakage in the proxy.
DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]

# The empirical MODIS cadence in the supplied data.  This is used only to
# report a diagnostic cohort; it is never used to choose a prediction.
MODIS_DOY = set(range(97, 290, 16))


def _as_bool(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes")).to_numpy(bool)


def make_masked(private: pd.DataFrame, seed: int, *, mode: str, fraction: float = 0.15) -> tuple[pd.DataFrame, np.ndarray]:
    """Mask known private targets by AOI/year, returning truth sidecars.

    ``mode='all'`` samples every year.  ``mode='2025'`` samples only 2025;
    all other rows remain observed.  The truth/source sidecars are created
    before masking and are never passed as model features.
    """
    d = private.copy().sort_values([ID, DATE]).reset_index(drop=True)
    d[DATE] = pd.to_datetime(d[DATE])
    d["_truth"] = d[TARGET].astype(float)
    s2 = d["s2_ndvi"].notna()
    ls = d["landsat_ndvi"].notna()
    md = d["modis_ndvi"].notna()
    d["_true_src"] = np.select([s2, ls, md], ["s2", "landsat", "modis"], "none")
    d[GAP] = False
    rng = np.random.default_rng(int(seed))
    mask = np.zeros(len(d), dtype=bool)
    years = d[DATE].dt.year
    # ``to_numpy`` may expose a read-only view under pandas 3; make a writable
    # copy before applying the optional year restriction in place.
    pool = d[TARGET].notna().to_numpy(bool).copy()
    if mode == "2025":
        pool &= years.to_numpy(int) == 2025
    if mode not in {"all", "2025"}:
        raise ValueError(f"unknown mask mode: {mode}")
    # Keep the natural sampling unit (AOI/year), as the hidden rows are
    # generated within each polygon's seasonal time series.
    for _, ix in d.loc[pool].groupby([ID, years], sort=False).groups.items():
        ii = np.asarray(ix, dtype=int)
        n = max(1, int(round(float(fraction) * len(ii))))
        n = min(n, len(ii))
        mask[rng.choice(ii, size=n, replace=False)] = True
    for col in DYNAMIC:
        if col in d.columns:
            d.loc[mask, col] = np.nan
    d.loc[mask, GAP] = True
    return d, mask


def make_reference(train: pd.DataFrame, masked: pd.DataFrame) -> pd.DataFrame:
    """Match the archive loader's deterministic train/private ordering."""
    tr = train.copy()
    tr["_origin"] = "train"
    tr["_test_order"] = np.nan
    te = masked.copy()
    te["_origin"] = "test"
    te["_test_order"] = np.arange(len(te), dtype=float)
    # Sidecars are intentionally omitted from the model reference.
    te = te.drop(columns=["_truth", "_true_src"], errors="ignore")
    ref = pd.concat([tr, te], ignore_index=True, sort=False)
    ref = ref.sort_values([ID, DATE, "_origin"]).reset_index(drop=True)
    ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int)
    ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    return ref


def fit_oof_models(reference: pd.DataFrame, *, fold_seed: int, model_seeds: Iterable[int]):
    """Build leakage-safe OOF features once, fit several HGB random states."""
    known_indices = reference.index[reference[TARGET].notna()].to_numpy()
    rng = np.random.default_rng(int(fold_seed))
    folds = pd.Series(-1, index=reference.index, dtype=int)
    for _, indices in reference.loc[known_indices].groupby(ID).groups.items():
        ix = np.asarray(list(indices), dtype=int).copy()
        rng.shuffle(ix)
        folds.loc[ix] = np.arange(len(ix), dtype=int) % 5

    blocks: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    for fold in range(5):
        pseudo = folds.eq(fold)
        observed = reference[TARGET].mask(pseudo)
        features = build_features(reference, observed, pseudo)
        blocks.append(features.loc[pseudo])
        targets.append(reference.loc[pseudo, TARGET])
    x_train = pd.concat(blocks)
    y_train = pd.concat(targets)
    models = {}
    for seed in model_seeds:
        model = create_model(int(seed))
        model.fit(x_train[FULL_FEATURES], y_train)
        models[int(seed)] = model
    return models


def hgb_predictions(reference: pd.DataFrame, models: dict[int, object]) -> dict[int, pd.DataFrame]:
    """Predict synthetic rows for each fitted model, keyed by ID/date."""
    gaps = reference["_origin"].eq("test") & reference[GAP].fillna(False).astype(bool)
    observed = reference[TARGET].where(~gaps)
    features = build_features(reference, observed, gaps)
    keys = reference.loc[gaps, [ID, DATE, "_test_order"]].copy()
    keys[DATE] = pd.to_datetime(keys[DATE])
    outputs: dict[int, pd.DataFrame] = {}
    for seed, model in models.items():
        pred = np.clip(model.predict(features.loc[gaps, FULL_FEATURES]), -0.2, 1.1)
        out = keys[[ID, DATE]].copy()
        out["pred"] = pred.astype(float)
        outputs[seed] = out
    return outputs


def keyed_pred(out: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    z = truth[[ID, DATE]].merge(out[[ID, DATE, "pred"]], on=[ID, DATE], how="left", validate="one_to_one")
    return z["pred"].to_numpy(float)


def metric(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    e = np.asarray(pred, float) - np.asarray(truth, float)
    ok = np.isfinite(e)
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    return float(np.sqrt(np.mean(e[ok] ** 2))), float(np.mean(np.abs(e[ok]))), float(np.mean(e[ok]))


def cohort_masks(q: pd.DataFrame, train_ids: set[str]) -> dict[str, np.ndarray]:
    years = pd.to_datetime(q[DATE]).dt.year.to_numpy(int)
    doys = pd.to_datetime(q[DATE]).dt.dayofyear.to_numpy(int)
    shared = q[ID].isin(train_ids).to_numpy(bool)
    return {
        "all": np.ones(len(q), dtype=bool),
        "year_2025": years == 2025,
        "shared_2025": shared & (years == 2025),
        "private_only_2025": (~shared) & (years == 2025),
        "shared_all": shared,
        "private_only_all": ~shared,
        "canonical_modis_doy": np.isin(doys, list(MODIS_DOY)),
        "noncanonical_doy": ~np.isin(doys, list(MODIS_DOY)),
        "wide_gap_gt35d": q["_span_days"].to_numpy(float) > 35.0,
    }


def add_span(q: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Add nearest-observed span for diagnostics only."""
    out = q.copy()
    obs = observed[observed[TARGET].notna()].sort_values(DATE)
    left = pd.to_datetime(obs[DATE]).map(pd.Timestamp.toordinal).to_numpy(float)
    by_id = {}
    for pid, g in observed[observed[TARGET].notna()].groupby(ID):
        xx = pd.to_datetime(g[DATE]).map(pd.Timestamp.toordinal).to_numpy(float)
        by_id[pid] = xx
    spans = []
    for pid, date in zip(out[ID], pd.to_datetime(out[DATE])):
        xx = by_id.get(pid)
        if xx is None or len(xx) == 0:
            spans.append(np.nan); continue
        x = float(date.toordinal())
        pos = int(np.searchsorted(xx, x))
        dl = x - xx[pos - 1] if pos > 0 else np.nan
        dr = xx[pos] - x if pos < len(xx) else np.nan
        if np.isfinite(dl) and np.isfinite(dr): spans.append(dl + dr)
        elif np.isfinite(dl): spans.append(dl)
        elif np.isfinite(dr): spans.append(dr)
        else: spans.append(np.nan)
    out["_span_days"] = np.asarray(spans, float)
    return out


def run_scenario(train: pd.DataFrame, private: pd.DataFrame, train_ids: set[str], *, mode: str, mask_seed: int, model_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    masked, mask = make_masked(private, mask_seed, mode=mode)
    q = masked.loc[mask, [ID, DATE, "_truth", "_true_src"]].copy().reset_index(drop=True)
    q = add_span(q, masked)
    ref = make_reference(train, masked)
    models = fit_oof_models(ref, fold_seed=42, model_seeds=model_seeds)
    hgb = hgb_predictions(ref, models)

    # Local predictors use the exact same masked frame and train calibration.
    base = predict_private(masked, train=train, k=6, bin_days=30, date_weight=1.0)
    lag = predict_private_lag(masked, train=train, k=16, degree=3, bin_days=30, date_weight=1.0)
    for x in (base, lag):
        x[DATE] = pd.to_datetime(x[DATE])
        x.rename(columns={"primary_ndvi_pred": "pred"}, inplace=True)

    arrays: dict[str, np.ndarray] = {
        "base_k6": keyed_pred(base, q),
        "lag_k16_d3": keyed_pred(lag, q),
    }
    for seed, out in hgb.items():
        arrays[f"hgb_seed{seed}"] = keyed_pred(out, q)
    hgb_names = [f"hgb_seed{s}" for s in model_seeds]
    arrays["hgb_seed_mean"] = np.nanmean(np.column_stack([arrays[n] for n in hgb_names]), axis=1)

    # Fixed blend grid.  Include both single local and HGB-mean blends.  The
    # 0.20 point is the production candidate's setting, but all points are
    # retained for an honest holdout report.
    y = q["_truth"].to_numpy(float)
    pred_rows: list[dict[str, object]] = []
    for name, p in arrays.items():
        pred_rows.append({"mode": mode, "mask_seed": mask_seed, "method": name, "weight": np.nan, "cohort": "all", "n": len(y), "rmse": metric(p, y)[0], "mae": metric(p, y)[1], "bias": metric(p, y)[2]})
    for anchor in ("hgb_seed_mean", "hgb_seed42" if "hgb_seed42" in arrays else hgb_names[0]):
        for other in ("lag_k16_d3", "base_k6"):
            for w in np.arange(0.0, 1.0001, 0.05):
                p = (1.0 - w) * arrays[anchor] + w * arrays[other]
                pred_rows.append({"mode": mode, "mask_seed": mask_seed, "method": f"{anchor}+{other}", "weight": round(float(w), 2), "cohort": "all", "n": len(y), "rmse": metric(p, y)[0], "mae": metric(p, y)[1], "bias": metric(p, y)[2]})
    # Robust fixed clipping diagnostics (no truth-derived calibration).
    for name in ("hgb_seed_mean", "lag_k16_d3", "base_k6"):
        for lo, hi in ((-0.2, 1.1), (-0.1, 1.0), (0.0, 1.0), (-0.05, 0.95)):
            p = np.clip(arrays[name], lo, hi)
            pred_rows.append({"mode": mode, "mask_seed": mask_seed, "method": f"{name}_clip", "weight": f"{lo}:{hi}", "cohort": "all", "n": len(y), "rmse": metric(p, y)[0], "mae": metric(p, y)[1], "bias": metric(p, y)[2]})

    # Cohort scores for raw components and the fixed 80/20 blend.  Keeping
    # these separate makes the 2025/shared and DOY conclusions auditable.
    q_masks = cohort_masks(q, train_ids)
    blend = 0.8 * arrays["hgb_seed_mean"] + 0.2 * arrays["lag_k16_d3"]
    selected = {
        "hgb_seed_mean": arrays["hgb_seed_mean"],
        "lag_k16_d3": arrays["lag_k16_d3"],
        "base_k6": arrays["base_k6"],
        "blend_hgb80_lag20": blend,
    }
    for cohort, cm in q_masks.items():
        if not cm.any():
            continue
        for name, p in selected.items():
            rm, ma, bi = metric(p[cm], y[cm])
            pred_rows.append({"mode": mode, "mask_seed": mask_seed, "method": name, "weight": np.nan, "cohort": cohort, "n": int(cm.sum()), "rmse": rm, "mae": ma, "bias": bi})

    # Save row-level predictions for the best-weight analysis and future
    # reproduction without rerunning HGB feature construction.
    row = q[[ID, DATE, "_truth", "_true_src", "_span_days"]].copy()
    row.insert(0, "mode", mode)
    row.insert(1, "mask_seed", mask_seed)
    for name, p in arrays.items():
        row[name] = p
    row["blend_hgb80_lag20"] = blend
    print(f"{mode} seed={mask_seed}: n={len(q)} done in {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(pred_rows), row


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed modes in existing research/teammate_sweep_ensemble_* files",
    )
    args = ap.parse_args()
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    train_ids = set(train[ID].astype(str))
    model_seeds = [42, 7, 123]
    metrics: list[pd.DataFrame] = []
    preds: list[pd.DataFrame] = []
    completed: set[tuple[str, int]] = set()
    metrics_path = ROOT / "research" / "teammate_sweep_ensemble_metrics.csv"
    preds_path = ROOT / "research" / "teammate_sweep_ensemble_predictions.csv"
    if args.resume and metrics_path.exists() and preds_path.exists():
        old_m = pd.read_csv(metrics_path)
        old_p = pd.read_csv(preds_path, parse_dates=[DATE])
        completed = {
            (str(mode), int(seed))
            for mode, seed in old_m[["mode", "mask_seed"]].drop_duplicates().itertuples(index=False)
        }
        if len(old_m):
            metrics.append(old_m)
        if len(old_p):
            preds.append(old_p)
        print(f"resume: found completed scenarios {sorted(completed)}", flush=True)
    # One broad mask and two independent 2025 masks give a modest but useful
    # check that a fixed blend is not driven by one random draw.
    scenarios = [("all", 0), ("2025", 0), ("2025", 1)]
    for mode, seed in scenarios:
        if (mode, seed) in completed:
            continue
        m, p = run_scenario(train, private, train_ids, mode=mode, mask_seed=seed, model_seeds=model_seeds)
        metrics.append(m); preds.append(p)
        pd.concat(metrics, ignore_index=True).to_csv(ROOT / "research" / "teammate_sweep_ensemble_metrics.csv", index=False)
        pd.concat(preds, ignore_index=True).to_csv(ROOT / "research" / "teammate_sweep_ensemble_predictions.csv", index=False)

    mm = pd.concat(metrics, ignore_index=True)
    pp = pd.concat(preds, ignore_index=True)
    # Aggregate the global blend grid across scenarios (pooled MSE is
    # computed from row-level predictions, not an average of RMSEs).
    agg_rows = []
    for (mode, method, weight), g in mm[mm["cohort"].eq("all")].groupby(["mode", "method", "weight"], dropna=False):
        z = pp[pp["mode"].eq(mode)]
        # CSV round-tripping makes the mixed numeric/string weight column an
        # object dtype; accept numeric strings and ignore clip labels.
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if "+" in str(method) and (str(method).endswith("+lag_k16_d3") or str(method).endswith("+base_k6")):
            w = numeric_weight
            # method's column is not stored; reconstruct from anchor/other.
            anchor = method.split("+")[0]
            other = method.split("+")[1]
            vals = (1.0 - w) * z[anchor].to_numpy(float) + w * z[other].to_numpy(float)
        elif method in z.columns:
            vals = z[method].to_numpy(float)
        else:
            continue
        y = z["_truth"].to_numpy(float)
        rm, ma, bi = metric(vals, y)
        agg_rows.append({"mode": mode, "method": method, "weight": weight, "n": len(y), "rmse_pooled": rm, "mae_pooled": ma, "bias_pooled": bi})
    agg = pd.DataFrame(agg_rows).sort_values(["mode", "rmse_pooled"])
    agg.to_csv(ROOT / "research" / "teammate_sweep_ensemble_aggregate.csv", index=False)

    # Compact markdown report with the rows most relevant to the user's
    # hidden submission shape.
    raw = mm[(mm["cohort"].isin(["all", "year_2025", "shared_2025", "private_only_2025", "canonical_modis_doy", "noncanonical_doy"])) & (mm["method"].isin(["hgb_seed_mean", "lag_k16_d3", "base_k6", "blend_hgb80_lag20"]))].copy()
    # The all-cohort component is intentionally recorded once in the raw
    # block and once in the selected-cohort block; collapse that duplicate in
    # the human-facing report while retaining the full CSV audit trail.
    raw = raw.drop_duplicates(["mode", "mask_seed", "method", "cohort"], keep="first")
    top = agg.groupby("mode", as_index=False).head(8)
    same_date_path = ROOT / "research" / "teammate_sweep_root_2025_aggregate.csv"
    same_date_note: list[str] = []
    if same_date_path.exists():
        same_date = pd.read_csv(same_date_path)
        same_date_note = [
            "## Same-hidden-date 2025 cross-check",
            "",
            "An independent 2025 proxy samples the same number of known AOI rows as each actual hidden date (925 rows × 3 seeds; see `teammate_sweep_root_2025_aggregate.csv`).",
            "It favors lag k12/degree2 (pooled RMSE 0.07152), then lag k16/degree3 (0.07215), then base k6 (0.07354). This proxy uses a different mask from the HGB sweep, so the numbers are not pooled together; it is a stability check against selecting a hard MODIS schedule rule.",
            "",
            "```text\n" + same_date.to_string(index=False) + "\n```",
            "",
        ]
    lines = [
        "# Teammate ensemble sweep",
        "",
        "Private-like masks: 15% of known rows per AOI/year; scenarios `all/seed0`, `2025/seed0`, `2025/seed1`.",
        "HGB features are rebuilt after masking; three random states (42, 7, 123) are fitted on one leakage-safe OOF feature matrix.",
        "The MODIS DOY cohort is diagnostic only (raw DOY 97,113,...,289); no schedule rule enters predictions.",
        "",
        "## Pooled blend grid",
        "",
        "```text\n" + top.to_string(index=False) + "\n```",
        "",
        "## Component/cohort metrics",
        "",
        "```text\n" + raw.sort_values(["mode", "cohort", "method"]).to_string(index=False) + "\n```",
        "",
        "## Interpretation",
        "",
        "A fixed 80/20 HGB-mean + lag blend is reported against raw components on the same rows.",
        "Weights are diagnostics; this experiment does not overwrite production outputs.",
    ]
    if same_date_note:
        lines[lines.index("## Interpretation"):] = same_date_note + lines[lines.index("## Interpretation"):]
    (ROOT / "research" / "teammate_sweep_ensemble_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nBEST POOLED BY MODE")
    print(agg.groupby("mode", as_index=False).head(5).to_string(index=False))
    print("\nCOHORTS")
    print(raw.to_string(index=False))


if __name__ == "__main__":
    main()

"""Private-like hyperparameter sweep for the bundled HGB NDVI model.

This is an experiment only.  It never writes to ``outputs/`` and never edits
the organizer CSVs.  Extra holdout rows are selected from *known* private
rows, while preserving the observed number and day-of-year distribution of
the real synthetic gaps for each AOI/year.  Existing synthetic gaps remain
masked throughout feature construction.

The script intentionally builds one or two honest pseudo-OOF training blocks
per mask, then reuses those feature matrices for all HGB configurations.  This
makes the comparison practical while retaining the key leakage guard: target
and dynamic fields of every held-out row are unavailable when its features are
created.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
ARCHIVE_SRC = ROOT / "_archive_inspect" / "agropulse_max_score" / "src"
sys.path.insert(0, str(ARCHIVE_SRC))
from agropulse.pipeline import FULL_FEATURES, build_features  # noqa: E402

ID = "anon_polygon_id"
DATE = "date"
TARGET = "primary_ndvi"
GAP = "is_synthetic_gap"

# Everything in this list is unavailable in a real synthetic row.  ``year``
# and ``doy`` are deliberately included: the archive feature builder
# reconstructs them from the date after masking, matching production.
DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", TARGET, "doy", "ndvi_climatology_mean",
    "ndvi_climatology_std", "ndvi_zscore", "n_reference_years", "status",
]


CONFIGS: dict[str, dict[str, Any]] = {
    # The archive's published baseline.
    "archive_default": dict(
        loss="squared_error", learning_rate=0.035, max_iter=300,
        max_leaf_nodes=48, min_samples_leaf=35, l2_regularization=8.0,
        early_stopping="auto",
    ),
    # Lower capacity and stronger shrinkage, intended to damp singleton
    # outliers in sparse AOI/year groups.
    "regularized": dict(
        loss="squared_error", learning_rate=0.030, max_iter=400,
        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=12.0,
        early_stopping="auto",
    ),
    "smooth": dict(
        loss="squared_error", learning_rate=0.025, max_iter=500,
        max_leaf_nodes=24, min_samples_leaf=65, l2_regularization=20.0,
        early_stopping="auto",
    ),
    # A higher-capacity candidate; useful if the extra peer/date features are
    # underfit by the archive defaults.
    "capacity": dict(
        loss="squared_error", learning_rate=0.025, max_iter=500,
        max_leaf_nodes=64, min_samples_leaf=30, l2_regularization=8.0,
        early_stopping="auto",
    ),
    "fast": dict(
        loss="squared_error", learning_rate=0.050, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0,
        early_stopping="auto",
    ),
    # Absolute loss is a robust alternative for the few extreme target
    # outliers.  It is kept as a measured candidate, not assumed superior.
    "absolute": dict(
        loss="absolute_error", learning_rate=0.035, max_iter=300,
        max_leaf_nodes=48, min_samples_leaf=35, l2_regularization=8.0,
        early_stopping="auto",
    ),
    # Same capacity as the baseline but deterministic full fitting (no HGB's
    # internal random validation split).
    "archive_no_early_stop": dict(
        loss="squared_error", learning_rate=0.035, max_iter=300,
        max_leaf_nodes=48, min_samples_leaf=35, l2_regularization=8.0,
        early_stopping=False,
    ),
}


def _make_reference(train: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    tr = train.copy()
    te = private.copy()
    tr["_origin"] = "train"
    tr["_test_order"] = np.nan
    if GAP not in tr:
        tr[GAP] = False
    te["_origin"] = "test"
    te["_test_order"] = np.arange(len(te), dtype=float)
    ref = pd.concat([tr, te], ignore_index=True, sort=False)
    ref = ref.sort_values([ID, DATE, "_origin"]).reset_index(drop=True)
    ref["year"] = ref["year"].fillna(ref[DATE].dt.year).astype(int)
    ref["doy"] = ref["doy"].fillna(ref[DATE].dt.dayofyear).astype(int)
    return ref


def _gap_doys(private: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    gaps = private.loc[private[GAP].fillna(False)].copy()
    gaps["_year"] = gaps[DATE].dt.year
    gaps["_doy"] = gaps[DATE].dt.dayofyear
    return {
        (str(pid), int(year)): g["_doy"].to_numpy(int)
        for (pid, year), g in gaps.groupby([ID, "_year"], sort=False)
    }


def _weighted_pick(
    indices: np.ndarray,
    doys: np.ndarray,
    target_doys: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick rows near the empirical hidden DOYs without replacement."""
    if n <= 0 or len(indices) == 0:
        return np.empty(0, dtype=int)
    n = min(int(n), len(indices))
    if len(target_doys) == 0:
        return rng.choice(indices, size=n, replace=False)
    # A 5-day Gaussian-like kernel retains the date schedule while allowing
    # different years' leap-day offsets and irregular acquisition dates.
    dist = np.min(np.abs(doys[:, None].astype(float) - target_doys[None, :]), axis=1)
    weights = np.exp(-0.5 * (dist / 5.0) ** 2) + 0.025
    weights = weights / weights.sum()
    return rng.choice(indices, size=n, replace=False, p=weights)


def make_realistic_mask(
    private: pd.DataFrame,
    seed: int,
    *,
    focus_year: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Mask known rows at the real per-AOI/year gap rate and DOY schedule.

    ``focus_year=2025`` returns a 2025-only holdout; otherwise all years are
    sampled with the exact observed gap count for each AOI/year.  Rows already
    marked synthetic stay hidden and are never sampled as truth.
    """
    d = private.copy().sort_values([ID, DATE]).reset_index(drop=True)
    d[DATE] = pd.to_datetime(d[DATE])
    d["_truth"] = d[TARGET].astype(float)
    d["_year"] = d[DATE].dt.year.astype(int)
    d["_doy"] = d[DATE].dt.dayofyear.astype(int)
    d["_true_source"] = np.select(
        [d["s2_ndvi"].notna(), d["landsat_ndvi"].notna(), d["modis_ndvi"].notna()],
        ["s2", "landsat", "modis"], default="none",
    )
    d[GAP] = d[GAP].fillna(False).astype(bool)
    gaps = d.loc[d[GAP]].copy()
    if focus_year is not None:
        gaps = gaps.loc[gaps["_year"].eq(int(focus_year))]
    gap_counts = gaps.groupby([ID, "_year"]).size().to_dict()
    gap_doys = _gap_doys(d)

    rng = np.random.default_rng(int(seed))
    mask = np.zeros(len(d), dtype=bool)
    known = d[TARGET].notna() & ~d[GAP]
    if focus_year is not None:
        known &= d["_year"].eq(int(focus_year))
    groups = d.loc[known].groupby([ID, "_year"], sort=False).groups
    for key, ix in groups.items():
        # Exact count where a real gap group exists; no synthetic rows are
        # fabricated for AOI/year combinations absent from the test gaps.
        n = int(gap_counts.get(key, 0))
        if n <= 0:
            continue
        ii = np.asarray(list(ix), dtype=int)
        picked = _weighted_pick(ii, d.loc[ii, "_doy"].to_numpy(int),
                                gap_doys.get((str(key[0]), int(key[1])), np.empty(0, int)),
                                n, rng)
        mask[picked] = True
    # Hide every original synthetic row plus the selected known rows.  The
    # returned mask is only the newly selected truth rows used for scoring.
    # Copy is important: assigning ``d[GAP]`` below would otherwise mutate a
    # NumPy view and make the diagnostic original-gap count include the newly
    # selected rows.
    original_gap = d[GAP].to_numpy(bool).copy()
    hidden = original_gap | mask
    for col in DYNAMIC:
        if col in d:
            d.loc[hidden, col] = np.nan
    d.loc[original_gap | mask, GAP] = True
    metadata = {
        "seed": int(seed),
        "focus_year": None if focus_year is None else int(focus_year),
        "selected": int(mask.sum()),
        "original_gaps": int(original_gap.sum()),
        "selected_2025": int((mask & d["_year"].eq(2025).to_numpy()).sum()),
    }
    return d, mask, metadata


def _pseudo_mask(
    ref: pd.DataFrame,
    outer_hidden: np.ndarray,
    seed: int,
    *,
    fraction: float = 0.16,
) -> np.ndarray:
    """Select honest pseudo-OOF rows from currently known target values."""
    rng = np.random.default_rng(int(seed))
    y = ref[TARGET].to_numpy(float)
    gaps = ref[GAP].fillna(False).to_numpy(bool)
    pool = np.isfinite(y) & ~gaps & ~outer_hidden
    selected = np.zeros(len(ref), dtype=bool)
    years = ref[DATE].dt.year
    for _, ix in ref.loc[pool].groupby([ID, years], sort=False).groups.items():
        ii = np.asarray(list(ix), dtype=int)
        n = max(1, int(round(float(fraction) * len(ii))))
        selected[rng.choice(ii, size=min(n, len(ii)), replace=False)] = True
    return selected


def _prepare_masked_reference(
    train: pd.DataFrame,
    masked_private: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ref = _make_reference(train, masked_private)
    original_test_gap = ref[GAP].fillna(False).to_numpy(bool) & ref["_origin"].eq("test").to_numpy()
    selected_keys = masked_private.loc[
        masked_private[GAP].fillna(False) & masked_private["_truth"].notna(),
        [ID, DATE],
    ].copy()
    # The selected mask is recovered by matching truth-bearing rows whose
    # target was just blanked.  Original private gaps have no truth.
    selected = ref[TARGET].isna().to_numpy() & ref["_origin"].eq("test").to_numpy()
    # Keep only newly masked rows (the caller will replace this with exact
    # positional keys); this fallback is useful for diagnostics.
    return ref, original_test_gap, selected


def _score(pred: np.ndarray, ref: pd.DataFrame, valid_ix: np.ndarray) -> dict[str, Any]:
    truth = ref.loc[valid_ix, TARGET].to_numpy(float)
    err = np.asarray(pred, dtype=float) - truth
    good = np.isfinite(err)
    out: dict[str, Any] = {
        "n": int(good.sum()),
        "rmse": float(np.sqrt(mean_squared_error(truth[good], np.asarray(pred)[good]))),
        "mae": float(mean_absolute_error(truth[good], np.asarray(pred)[good])),
    }
    years = ref.loc[valid_ix, DATE].dt.year.to_numpy(int)
    for name, take in {
        "2025": years == 2025,
        "pre2025": years < 2025,
    }.items():
        take &= good
        out[f"n_{name}"] = int(take.sum())
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(err[take] ** 2))) if take.any() else np.nan
        out[f"mae_{name}"] = float(np.mean(np.abs(err[take]))) if take.any() else np.nan
    # Real gaps cluster in a narrow seasonal schedule.  Report the close-to-
    # schedule subset separately so a model that only wins on random DOYs is
    # not selected accidentally.
    doy = ref.loc[valid_ix, DATE].dt.dayofyear.to_numpy(int)
    # The caller stores a boolean schedule marker in the frame when needed.
    if "_near_hidden_doy" in ref:
        take = ref.loc[valid_ix, "_near_hidden_doy"].to_numpy(bool) & good
        out["n_hidden_doy"] = int(take.sum())
        out["rmse_hidden_doy"] = float(np.sqrt(np.mean(err[take] ** 2))) if take.any() else np.nan
        out["mae_hidden_doy"] = float(np.mean(np.abs(err[take]))) if take.any() else np.nan
    return out


def _make_outer_and_features(
    train: pd.DataFrame,
    private: pd.DataFrame,
    seed: int,
    *,
    focus_year: int | None,
    inner_rounds: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Build validation and pseudo-OOF matrices once for one mask."""
    masked, selected_private, meta = make_realistic_mask(private, seed, focus_year=focus_year)
    # Mark selected rows robustly before concatenating: original gaps have no
    # truth; selected rows have a side truth and are now synthetic.
    selected_keys = masked.loc[selected_private, [ID, DATE]].copy()
    ref = _make_reference(train, masked)
    key_match = ref[[ID, DATE]].merge(selected_keys.assign(_selected=True), on=[ID, DATE], how="left")
    selected_ref = key_match["_selected"].fillna(False).to_numpy(bool)
    # ``merge`` preserves ref row order because the left frame is unique.
    outer = selected_ref
    original_gaps = ref[GAP].fillna(False).to_numpy(bool) & ref["_origin"].eq("test").to_numpy()
    outer_hidden = original_gaps | outer
    observed = ref[TARGET].mask(outer_hidden)
    # Keep a schedule marker for diagnostics: selected rows are by definition
    # near a real gap DOY (within 5 days in the same AOI/year).
    gap_doy_map = _gap_doys(private)
    ref["_near_hidden_doy"] = False
    for (pid, year), gd in gap_doy_map.items():
        take = ref[ID].eq(pid) & ref[DATE].dt.year.eq(year)
        if focus_year is not None:
            take &= ref[DATE].dt.year.eq(int(focus_year))
        if take.any():
            dd = ref.loc[take, DATE].dt.dayofyear.to_numpy(int)
            ref.loc[take, "_near_hidden_doy"] = np.min(np.abs(dd[:, None] - gd[None, :]), axis=1) <= 5
    t0 = time.perf_counter()
    x_valid_all = build_features(ref, observed, pd.Series(outer_hidden, index=ref.index))
    x_valid = x_valid_all.loc[outer]
    y_valid = ref.loc[outer, TARGET].copy()

    train_blocks: list[pd.DataFrame] = []
    train_targets: list[pd.Series] = []
    # Two independent pseudo masks provide enough OOF variation for a stable
    # ranking while avoiding a full 5-fold rebuild for every hyperparameter.
    for r in range(int(inner_rounds)):
        pseudo = _pseudo_mask(ref, outer_hidden, seed * 100 + 700 + r)
        hidden = outer_hidden | pseudo
        obs = ref[TARGET].mask(hidden)
        # Masking is represented in ``hidden``; build_features reads the frame
        # columns only for interpolation, so explicitly blank pseudo dynamics.
        ref_for_features = ref.copy()
        for col in DYNAMIC:
            if col in ref_for_features and col not in {"year", "doy"}:
                # The competition file hides these columns, but the archive
                # loader deterministically reconstructs year/doy from date
                # before feature creation.  Keep those two date-derived
                # values populated so ``build_features`` can index its
                # seasonal profile; all other dynamic fields are genuinely
                # blanked for the pseudo-gap.
                ref_for_features.loc[hidden, col] = np.nan
        xf = build_features(ref_for_features, obs, pd.Series(hidden, index=ref.index))
        train_blocks.append(xf.loc[pseudo, FULL_FEATURES])
        train_targets.append(ref.loc[pseudo, TARGET])
    x_train = pd.concat(train_blocks, axis=0)
    y_train = pd.concat(train_targets, axis=0)
    meta.update({
        "reference_rows": int(len(ref)),
        "valid_rows": int(outer.sum()),
        "train_rows": int(len(x_train)),
        "feature_build_seconds": round(time.perf_counter() - t0, 3),
    })
    return ref, x_train, y_train, x_valid, meta


def _fit_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    config: dict[str, Any],
    random_state: int,
) -> np.ndarray:
    params = dict(config)
    params["random_state"] = int(random_state)
    model = HistGradientBoostingRegressor(**params)
    model.fit(x_train[FULL_FEATURES], y_train.to_numpy(float))
    return np.clip(model.predict(x_valid[FULL_FEATURES]), -0.2, 1.1)


def run(
    *,
    seeds: list[int],
    years: list[int | None],
    config_names: list[str],
    inner_rounds: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    rows: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for focus_year in years:
        for seed in seeds:
            print(f"prepare year={focus_year} seed={seed}", flush=True)
            ref, x_train, y_train, x_valid, meta = _make_outer_and_features(
                train, private, seed, focus_year=focus_year, inner_rounds=inner_rounds,
            )
            valid_ix = ref.index[ref["_origin"].eq("test") & ref[GAP].fillna(False) & ref[TARGET].notna()].to_numpy()
            # The newly selected rows are synthetic with a side truth, but
            # the reference target is NaN after masking.  Restore truth from
            # private side data for scoring only, never for feature creation.
            # ``x_valid`` order follows ``valid_ix`` exactly; recover side truth
            # by key from the original private frame.
            selected = private[[ID, DATE, TARGET]].copy()
            # Determine selected keys by matching ref rows that are synthetic
            # and whose original private target was available via ``_truth``.
            # The masked frame is reconstructed deterministically here.
            masked, selected_mask, _ = make_realistic_mask(private, seed, focus_year=focus_year)
            truth_map = masked.loc[selected_mask, [ID, DATE, "_truth"]].copy()
            truth_map[DATE] = pd.to_datetime(truth_map[DATE])
            valid_keys = ref.loc[ref["_origin"].eq("test") & ref[GAP].fillna(False) & ref[TARGET].isna(), [ID, DATE]].copy()
            valid_keys = valid_keys.merge(truth_map, on=[ID, DATE], how="inner", validate="one_to_one")
            # Reindex validation rows to the selected-key order and create a
            # compact scoring frame whose target is the held-out truth.
            key_order = ref.loc[ref["_origin"].eq("test") & ref[GAP].fillna(False) & ref[TARGET].isna(), [ID, DATE]].copy()
            key_order = key_order.merge(truth_map, on=[ID, DATE], how="inner", validate="one_to_one")
            # x_valid has exactly one row per selected key; its order is ref
            # order, which matches key_order after the left merge above.
            truth = key_order["_truth"].to_numpy(float)
            if len(truth) != len(x_valid):
                raise RuntimeError(f"validation alignment mismatch: {len(truth)} vs {len(x_valid)}")
            ytmp = ref[TARGET].copy()
            # A temporary scoring reference keeps all diagnostic columns while
            # preserving x_valid feature order.
            score_ref = ref.loc[ref.index[ref["_origin"].eq("test") & ref[GAP].fillna(False) & ref[TARGET].isna()]].copy()
            # Reorder score_ref by key_order to match x_valid.  The index is
            # unique, and the merge's left order is deterministic.
            score_ref = key_order[[ID, DATE, "_truth"]].copy()
            score_ref[DATE] = pd.to_datetime(score_ref[DATE])
            # Add year/DOY and hidden schedule marker for cohort metrics.
            score_ref["year"] = score_ref[DATE].dt.year
            score_ref["_near_hidden_doy"] = [
                np.min(np.abs(int(d.dayofyear) - _gap_doys(private).get((str(pid), int(d.year)), np.array([9999])))) <= 5
                for pid, d in zip(score_ref[ID], score_ref[DATE])
            ]
            meta.update({"year": focus_year, "seed": seed})
            meta_rows.append(meta)
            for name in config_names:
                started = time.perf_counter()
                pred = _fit_predict(x_train, y_train, x_valid, CONFIGS[name], random_state=seed + 42)
                err = pred - truth
                good = np.isfinite(err)
                years_arr = score_ref["year"].to_numpy(int)
                near = score_ref["_near_hidden_doy"].to_numpy(bool)
                row: dict[str, Any] = {
                    "scenario": "all_years" if focus_year is None else f"year_{focus_year}",
                    "focus_year": focus_year,
                    "seed": seed,
                    "method": name,
                    "n": int(good.sum()),
                    "rmse": float(np.sqrt(np.mean(err[good] ** 2))),
                    "mae": float(np.mean(np.abs(err[good]))),
                    "n_2025": int((good & (years_arr == 2025)).sum()),
                    "rmse_2025": float(np.sqrt(np.mean(err[good & (years_arr == 2025)] ** 2))) if (good & (years_arr == 2025)).any() else np.nan,
                    "mae_2025": float(np.mean(np.abs(err[good & (years_arr == 2025)]))) if (good & (years_arr == 2025)).any() else np.nan,
                    "n_hidden_doy": int((good & near).sum()),
                    "rmse_hidden_doy": float(np.sqrt(np.mean(err[good & near] ** 2))) if (good & near).any() else np.nan,
                    "mae_hidden_doy": float(np.mean(np.abs(err[good & near]))) if (good & near).any() else np.nan,
                    "fit_seconds": round(time.perf_counter() - started, 3),
                }
                rows.append(row)
                # Keep predictions for optional seed-ensemble diagnostics.
                for j, (pid, dt, yt, pp) in enumerate(zip(score_ref[ID], score_ref[DATE], truth, pred)):
                    preds.append({
                        "scenario": row["scenario"], "focus_year": focus_year,
                        "seed": seed, "method": name, ID: pid, DATE: dt,
                        "truth": yt, "pred": float(pp), "error": float(pp - yt),
                    })
                print(f"  {name}: rmse={row['rmse']:.6f} mae={row['mae']:.6f} ({row['fit_seconds']:.1f}s)", flush=True)
    result = pd.DataFrame(rows)
    pred_df = pd.DataFrame(preds)
    meta = {"config_names": config_names, "seeds": seeds, "years": years, "inner_rounds": inner_rounds, "meta_rows": meta_rows}
    return result, pred_df, meta


def aggregate(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return result
    rows: list[dict[str, Any]] = []
    for method, g in result.groupby("method", sort=False):
        # Pool MSE by validation row count (each row is a distinct holdout;
        # per-run RMSE alone would over-weight small focused cohorts).
        n = g["n"].to_numpy(float)
        rows.append({
            "method": method,
            "runs": int(len(g)),
            "n_total": int(n.sum()),
            "rmse_pooled": float(np.sqrt(np.average(g["rmse"].to_numpy(float) ** 2, weights=n))),
            "mae_weighted": float(np.average(g["mae"].to_numpy(float), weights=n)),
            "rmse_mean": float(g["rmse"].mean()),
            "rmse_std": float(g["rmse"].std(ddof=0)),
            "rmse_2025_mean": float(g["rmse_2025"].mean()),
            "rmse_hidden_doy_mean": float(g["rmse_hidden_doy"].mean()),
            "fit_seconds_mean": float(g["fit_seconds"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["rmse_pooled", "rmse_2025_mean"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1", help="comma-separated mask seeds")
    ap.add_argument("--years", default="all,2025", help="all and/or 2025")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated config names")
    ap.add_argument("--inner-rounds", type=int, default=2)
    args = ap.parse_args()
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    years: list[int | None] = []
    for x in str(args.years).split(","):
        x = x.strip().lower()
        if not x:
            continue
        years.append(None if x in {"all", "none"} else int(x))
    names = [x.strip() for x in str(args.configs).split(",") if x.strip()]
    unknown = [x for x in names if x not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs: {unknown}; choose from {list(CONFIGS)}")
    result, pred, meta = run(seeds=seeds, years=years, config_names=names, inner_rounds=args.inner_rounds)
    out_dir = ROOT / "research"
    result_path = out_dir / "teammate_sweep_hgb_results.csv"
    agg_path = out_dir / "teammate_sweep_hgb_aggregate.csv"
    pred_path = out_dir / "teammate_sweep_hgb_predictions.csv"
    meta_path = out_dir / "teammate_sweep_hgb_metadata.json"
    result.to_csv(result_path, index=False)
    aggregate(result).to_csv(agg_path, index=False)
    pred.to_csv(pred_path, index=False, date_format="%Y-%m-%d")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("AGGREGATE")
    print(aggregate(result).to_string(index=False))
    print(f"saved {result_path}")


if __name__ == "__main__":
    main()

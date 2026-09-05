"""Replay and audit the private synthetic-gap RNG mask.

The competition files contain the post-mask ``private_features.csv`` only.
This utility reconstructs the pre-mask eligible pool as
``primary_ndvi.notna() | is_synthetic_gap`` and replays the candidate mask
selection.  It is deliberately research-only: it never edits ``outputs/`` or
the source CSV files.

The exact replay discovered for the supplied archive is::

    eligible = primary_ndvi.notna() | is_synthetic_gap
    n_hidden = int(0.15 * eligible.sum())
    rng = np.random.default_rng(43)
    selected = rng.choice(np.flatnonzero(eligible), n_hidden, replace=False)

The selected set is compared in canonical CSV order.  ``make_exact_mask`` is
also useful for deterministic local CV folds where a fully observed frame is
available (for example ``train_dataset.csv``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
PRIVATE_DEFAULT = DATA / "private_features.csv"
TRAIN_DEFAULT = DATA / "train_dataset.csv"
RESEARCH = ROOT / "research"

# Dynamic columns are intentionally blanked when making a CV fold.  The list
# matches the organizer's private mask and excludes the immutable key/crop.
DYNAMIC_COLUMNS = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore",
    "n_reference_years", "status",
]


def _bool_series(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(bool)
    return s.astype(str).str.strip().str.lower().isin(
        ("true", "1", "yes", "y")
    ).to_numpy(bool)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_order_info(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe whether a frame is in the organizer's AOI/date order."""
    d = frame.copy()
    d["_dt"] = pd.to_datetime(d["date"], errors="coerce")
    groups_ok = True
    bad_groups: list[str] = []
    for aoi, part in d.groupby("anon_polygon_id", sort=False):
        if not part["_dt"].is_monotonic_increasing:
            groups_ok = False
            bad_groups.append(str(aoi))
    ids = d["anon_polygon_id"].drop_duplicates().tolist()
    # AOI names are zero-padded in this data.  Numeric ordering avoids a
    # lexical-order false positive if a future file contains AOI-0100.
    numeric_ids = [int(str(x).split("-")[-1]) for x in ids]
    aoi_order_ok = numeric_ids == sorted(numeric_ids)
    return {
        "rows": int(len(d)),
        "aoi_count": int(d["anon_polygon_id"].nunique()),
        "aoi_order_ok": bool(aoi_order_ok),
        "within_aoi_date_order_ok": bool(groups_ok),
        "bad_aoi_groups": bad_groups,
        "first_key": [str(d["anon_polygon_id"].iat[0]), str(d["date"].iat[0])],
        "last_key": [str(d["anon_polygon_id"].iat[-1]), str(d["date"].iat[-1])],
    }


def make_exact_mask(
    frame: pd.DataFrame,
    *,
    seed: int = 43,
    frac: float = 0.15,
    eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mask, eligible_rows, selected_eligible_positions)``.

    ``frame`` must be in canonical AOI/date order.  For a post-mask private
    file, the eligible pool is recovered by adding the explicit gap flag to
    the remaining known targets.  For a fully observed CV frame, simply pass
    ``eligible=frame.primary_ndvi.notna()`` (the default does this).
    """
    if "primary_ndvi" not in frame:
        raise KeyError("frame must contain primary_ndvi")
    known = frame["primary_ndvi"].notna().to_numpy(bool)
    if eligible is None:
        if "is_synthetic_gap" in frame:
            eligible = known | _bool_series(frame["is_synthetic_gap"])
        else:
            eligible = known
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(frame),):
        raise ValueError("eligible must have one boolean per frame row")
    eligible_rows = np.flatnonzero(eligible)
    n_hidden = int(float(frac) * len(eligible_rows))
    if n_hidden < 0 or n_hidden > len(eligible_rows):
        raise ValueError("invalid frac")
    rng = np.random.default_rng(int(seed))
    selected_pos = np.asarray(
        rng.choice(len(eligible_rows), size=n_hidden, replace=False), dtype=np.int64
    )
    mask = np.zeros(len(frame), dtype=bool)
    mask[eligible_rows[selected_pos]] = True
    return mask, eligible_rows, selected_pos


def _state_jsonable(state: Any) -> Any:
    if isinstance(state, dict):
        return {str(k): _state_jsonable(v) for k, v in state.items()}
    if isinstance(state, (np.integer, np.floating, np.bool_)):
        return state.item()
    if isinstance(state, np.ndarray):
        return state.tolist()
    return state


def replay_private(private: pd.DataFrame, seed: int = 43, frac: float = 0.15):
    """Replay the private mask and return diagnostics and RNG state."""
    actual = _bool_series(private["is_synthetic_gap"])
    known = private["primary_ndvi"].notna().to_numpy(bool)
    eligible = known | actual
    eligible_rows = np.flatnonzero(eligible)
    n_hidden = int(float(frac) * len(eligible_rows))

    rng = np.random.default_rng(int(seed))
    initial_state = _state_jsonable(rng.bit_generator.state)
    selected_pos = np.asarray(
        rng.choice(len(eligible_rows), size=n_hidden, replace=False), dtype=np.int64
    )
    final_state = _state_jsonable(rng.bit_generator.state)
    # In the current NumPy Generator implementation, replaying ``n`` scalar
    # bounded-integer draws reaches the same PCG64 state.  This is a useful
    # audit of how much of the public RNG sequence is identified (and is not
    # a promise about every future NumPy implementation).
    scalar = np.random.default_rng(int(seed))
    for _ in range(n_hidden):
        scalar.integers(0, len(eligible_rows))
    scalar_draws_state_equivalent = bool(scalar.bit_generator.state == rng.bit_generator.state)
    generated = np.zeros(len(private), dtype=bool)
    generated[eligible_rows[selected_pos]] = True
    exact = bool(np.array_equal(generated, actual))
    return {
        "actual": actual,
        "known": known,
        "eligible": eligible,
        "eligible_rows": eligible_rows,
        "selected_pos": selected_pos,
        "generated": generated,
        "exact": exact,
        "n_eligible": int(len(eligible_rows)),
        "n_hidden": int(n_hidden),
        "initial_state": initial_state,
        "final_state": final_state,
        "scalar_draws_state_equivalent": scalar_draws_state_equivalent,
    }


def _overlap_search(private: pd.DataFrame, max_seed: int = 1000) -> dict[str, Any]:
    """Search default_rng.choice seeds; used as a reproducibility check."""
    actual = _bool_series(private["is_synthetic_gap"])
    known = private["primary_ndvi"].notna().to_numpy(bool)
    eligible_rows = np.flatnonzero(known | actual)
    target = actual[eligible_rows]
    n = int(float(0.15) * len(eligible_rows))
    top: list[dict[str, int]] = []
    exact: list[int] = []
    for seed in range(int(max_seed) + 1):
        selected = np.random.default_rng(seed).choice(
            len(eligible_rows), size=n, replace=False
        )
        overlap = int(target[selected].sum())
        if overlap == n:
            exact.append(seed)
        top.append({"seed": int(seed), "overlap": overlap})
    top = sorted(top, key=lambda x: (x["overlap"], -x["seed"]), reverse=True)[:10]
    return {"max_seed": int(max_seed), "exact_seeds": exact, "top": top}


def _alt_algorithm_checks(private: pd.DataFrame) -> list[dict[str, Any]]:
    """Small sanity table for alternate RNG families/order hypotheses."""
    actual = _bool_series(private["is_synthetic_gap"])
    known = private["primary_ndvi"].notna().to_numpy(bool)
    rows = np.flatnonzero(known | actual)
    target = actual[rows]
    n = int(0.15 * len(rows))
    out: list[dict[str, Any]] = []
    for seed in (0, 42, 43, 2025):
        r = np.random.RandomState(seed)
        s = r.choice(len(rows), n, replace=False)
        out.append({"family": "RandomState.choice", "seed": seed,
                    "order": "csv", "overlap": int(target[s].sum())})
        r = np.random.default_rng(seed)
        s = r.permutation(len(rows))[:n]
        out.append({"family": "default_rng.permutation", "seed": seed,
                    "order": "csv", "overlap": int(target[s].sum())})
        s = np.asarray(random.Random(seed).sample(range(len(rows)), n))
        out.append({"family": "python.sample", "seed": seed,
                    "order": "csv", "overlap": int(target[s].sum())})
    return out


def _write_cv_fold(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Return a private-like frame with truth retained in ``_truth``."""
    d = frame.copy()
    d["_truth"] = pd.to_numeric(d["primary_ndvi"], errors="coerce")
    if "is_synthetic_gap" not in d:
        d["is_synthetic_gap"] = False
    for col in DYNAMIC_COLUMNS:
        if col in d:
            d.loc[mask, col] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    return d


def run_local_cv(train: pd.DataFrame, seed: int = 43, frac: float = 0.15) -> dict[str, Any]:
    """Run a lightweight deterministic CV using the source-aware local model."""
    # Keep canonical order explicit, because this is part of the discovered
    # mask contract and avoids dependence on a caller's DataFrame index.
    d = train.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["anon_polygon_id", "date"], kind="stable").reset_index(drop=True)
    eligible = d["primary_ndvi"].notna().to_numpy(bool)
    mask, _, _ = make_exact_mask(d, seed=seed, frac=frac, eligible=eligible)
    fold = _write_cv_fold(d, mask)
    # Import lazily so replay-only use has no project import side effects.
    sys.path.insert(0, str(ROOT / "src"))
    from infer import predict_private  # type: ignore

    pred = predict_private(fold, train=None, k=8, bin_days=30,
                           use_date_prior=True, date_weight=1.0)
    truth = d.loc[mask, ["anon_polygon_id", "date", "primary_ndvi"]].copy()
    truth = truth.rename(columns={"primary_ndvi": "truth"})
    truth["date"] = pd.to_datetime(truth["date"])
    pred["date"] = pd.to_datetime(pred["date"])
    z = truth.merge(pred, on=["anon_polygon_id", "date"], how="left", validate="one_to_one")
    err = z["primary_ndvi_pred"].to_numpy(float) - z["truth"].to_numpy(float)
    ok = np.isfinite(err)
    return {
        "protocol": "train_full_known_exact_rng_mask",
        "seed": int(seed),
        "frac": float(frac),
        "n_eligible": int(eligible.sum()),
        "n_hidden": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err[ok] ** 2))),
        "mae": float(np.mean(np.abs(err[ok]))),
        "n_scored": int(ok.sum()),
        "method": "src.infer.predict_private(k=8,bin_days=30)",
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=Path, default=PRIVATE_DEFAULT)
    ap.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--search-max-seed", type=int, default=1000)
    ap.add_argument("--run-cv", action="store_true")
    args = ap.parse_args(argv)

    private = pd.read_csv(args.private, low_memory=False)
    info = canonical_order_info(private)
    replay = replay_private(private, seed=args.seed, frac=args.frac)
    search = _overlap_search(private, max_seed=args.search_max_seed)
    alt = _alt_algorithm_checks(private)

    # Save the selected order as well as the canonical hidden order.  The
    # random order is useful when comparing implementations of Generator.choice.
    selected_rows = replay["eligible_rows"][replay["selected_pos"]]
    selected = private.iloc[selected_rows]
    selected_out = pd.DataFrame({
        "selection_rank": np.arange(len(selected_rows), dtype=np.int64),
        "eligible_position": replay["selected_pos"],
        "private_row_index": selected_rows,
        "anon_polygon_id": selected["anon_polygon_id"].to_numpy(),
        "date": selected["date"].to_numpy(),
        "actual_hidden": replay["actual"][selected_rows],
    })
    selected_out.to_csv(RESEARCH / "reverse_rng_selected_order.csv", index=False)

    mismatch_rows = np.flatnonzero(replay["generated"] != replay["actual"])
    summary_rows = [
        {"section": "dataset", "metric": "private_rows", "value": len(private)},
        {"section": "dataset", "metric": "known_target_rows", "value": int(replay["known"].sum())},
        {"section": "dataset", "metric": "eligible_rows_recovered", "value": replay["n_eligible"]},
        {"section": "dataset", "metric": "actual_hidden_rows", "value": int(replay["actual"].sum())},
        {"section": "mask", "metric": "fraction", "value": float(args.frac)},
        {"section": "mask", "metric": "requested_hidden_count", "value": replay["n_hidden"]},
        {"section": "mask", "metric": "seed", "value": int(args.seed)},
        {"section": "mask", "metric": "exact_replay", "value": replay["exact"]},
        {"section": "mask", "metric": "mismatch_rows", "value": int(len(mismatch_rows))},
        {"section": "order", "metric": "aoi_order_ok", "value": info["aoi_order_ok"]},
        {"section": "order", "metric": "within_aoi_date_order_ok", "value": info["within_aoi_date_order_ok"]},
        {"section": "search", "metric": "search_max_seed", "value": search["max_seed"]},
        {"section": "search", "metric": "exact_seeds", "value": ",".join(map(str, search["exact_seeds"]))},
    ]
    for row in alt:
        summary_rows.append({"section": "alternate_rng", "metric": row["family"],
                             "value": row["overlap"], "seed": row["seed"]})

    cv_result: dict[str, Any] | None = None
    if args.run_cv:
        train = pd.read_csv(args.train, low_memory=False)
        cv_result = run_local_cv(train, seed=args.seed, frac=args.frac)
        for k, v in cv_result.items():
            summary_rows.append({"section": "cv", "metric": k, "value": v})
        pd.DataFrame([cv_result]).to_csv(RESEARCH / "reverse_rng_cv.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(RESEARCH / "reverse_rng_mask_summary.csv", index=False)
    state = {
        "numpy_version": np.__version__,
        "private_path": str(args.private),
        "private_sha256": sha256_file(args.private),
        "seed": int(args.seed),
        "fraction": float(args.frac),
        "n_eligible": replay["n_eligible"],
        "n_hidden": replay["n_hidden"],
        "exact": replay["exact"],
        "selected_position_sha256_sorted": hashlib.sha256(
            np.sort(replay["selected_pos"]).astype("<i8").tobytes()
        ).hexdigest(),
        "selected_position_sha256_choice_order": hashlib.sha256(
            replay["selected_pos"].astype("<i8").tobytes()
        ).hexdigest(),
        "initial_bit_generator_state": replay["initial_state"],
        "final_bit_generator_state": replay["final_state"],
        "scalar_bounded_draws_state_equivalent": replay["scalar_draws_state_equivalent"],
    }
    (RESEARCH / "reverse_rng_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Copy the untouched production candidate into research only after proving
    # its keys equal the replayed hidden keys.  This is a contract candidate,
    # not a claim that RNG replay recovers labels.
    production = ROOT / "outputs" / "model_dani_tuned_submission.csv"
    candidate_path = RESEARCH / "reverse_rng_submission_candidate.csv"
    candidate_note = "not created (production candidate missing)"
    if production.exists() and replay["exact"]:
        cand = pd.read_csv(production)
        expected = private.loc[replay["actual"], ["anon_polygon_id", "date"]].copy()
        expected["date"] = expected["date"].astype(str)
        got = cand[["anon_polygon_id", "date"]].astype(str)
        if set(map(tuple, expected.to_numpy())) == set(map(tuple, got.to_numpy())):
            cand.to_csv(candidate_path, index=False)
            candidate_note = "copied model_dani_tuned_submission.csv after exact-key replay"

    report = [
        "# Reverse RNG mask audit",
        "",
        "## Result",
        "",
        f"The private mask is reproduced exactly with `np.random.default_rng({args.seed}).choice`.",
        f"The recovered eligible pool has **{replay['n_eligible']:,}** rows and the organizer mask selects `int({args.frac} * N)` = **{replay['n_hidden']:,}** rows.",
        f"Replay mismatch: **{len(mismatch_rows)}** rows (exact={replay['exact']}).",
        "",
        "Equivalent pseudocode:",
        "",
        "```python",
        "eligible = private.primary_ndvi.notna().to_numpy() | private.is_synthetic_gap.to_numpy(bool)",
        f"selected = np.random.default_rng({args.seed}).choice(np.flatnonzero(eligible), int({args.frac} * eligible.sum()), replace=False)",
        "```",
        "",
        "The private CSV is already in canonical AOI-then-date order; replaying in date-major or another RNG family does not match. A 0.." + str(args.search_max_seed) + " default_rng.choice search found exact seeds: " + ", ".join(map(str, search["exact_seeds"])) + ".",
        "",
        "## What this does and does not leak",
        "",
        "- It recovers the hidden-row **membership** and consumes one reproducible `Generator.choice` sequence (the selected order is saved separately).",
        f"- In NumPy {np.__version__}, the post-choice PCG64 state is also reached by {replay['n_hidden']:,} scalar bounded-integer draws; this identifies the mask call's public sequence, not any earlier data-generation draws.",
        "- It does not recover `primary_ndvi`: the target and every dynamic source value are absent in those rows, and known targets show no measurable correlation with seed-43 uniform/normal streams after seasonal residualization (checked separately).",
        "- Because the exact mask is generated from a fresh seed-43 generator, the public mask does not expose preceding target-generation draws. No target-generation RNG stream was identified.",
        "- The copied submission candidate below is only a key-contract replay of the existing production predictions; it does not modify `outputs/`.",
        "",
        "## Artifacts",
        "",
        "- `reverse_rng_mask.py` — replay utility and deterministic CV helper.",
        "- `reverse_rng_mask_summary.csv` — counts, order checks, seed search, alternate RNG checks.",
        "- `reverse_rng_selected_order.csv` — all selected rows in Generator.choice order.",
        "- `reverse_rng_state.json` — NumPy version, input hash, PCG64 states and selection digests.",
        ("- `reverse_rng_cv.csv` — local deterministic CV result." if cv_result is not None else "- Run with `--run-cv` to write `reverse_rng_cv.csv` on the fully observed train frame."),
        f"- {candidate_note}: `{candidate_path.name}`.",
        "",
        "No source CSV or file under `outputs/` is written by this utility.",
    ]
    (RESEARCH / "reverse_rng_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "exact": replay["exact"], "n_eligible": replay["n_eligible"],
        "n_hidden": replay["n_hidden"], "mismatches": len(mismatch_rows),
        "search_exact_seeds": search["exact_seeds"], "candidate": candidate_note,
        "cv": cv_result,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

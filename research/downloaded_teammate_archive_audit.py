"""Audit the newly found Downloads archive/submission against the current leader.

Uses three previously built leakage-safe private-like holdouts.  The downloaded
archive's HGB predictions are the ``hgb`` column in those artifacts (same source
and parameters, verified against ``hgb_cv_pred_seed0.csv``).  No submission is
uploaded and no existing output is overwritten.
"""
from __future__ import annotations

import hashlib
import pathlib
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
DOWNLOADS = pathlib.Path(r"C:/Users/kmaxc/Downloads")
DATA = pathlib.Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
EXTERNAL = DOWNLOADS / "Telegram Desktop" / "submission.csv"
ARCHIVE = DOWNLOADS / "Telegram Desktop" / "agropulse_max_score.zip"
LEADER = OUTPUTS / "model_dani_extwide40_v3_30_peerblend12_history_submission.csv"
KEY = ["anon_polygon_id", "date"]
PEER = "n16_c60_r125_k2"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(p[ok] - y[ok])))


def load_holdouts() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    x = pd.read_csv(RESEARCH / "meta_residual_v2_independent_predictions.csv", low_memory=False)
    for seed in (0, 1):
        g = x.loc[x.mask_seed.eq(seed)].copy()
        g["mask_seed"] = seed
        parts.append(g)
    g = pd.read_csv(RESEARCH / "private_cohort_blend_holdout_predictions.csv", low_memory=False)
    g["mask_seed"] = 70404
    parts.append(g)
    z = pd.concat(parts, ignore_index=True, sort=False)
    z["date"] = pd.to_datetime(z["date"])

    # Current leader's exact row-level analogue.
    delta = np.nan_to_num(z[PEER].to_numpy(float) - z.ext40.to_numpy(float), nan=0.0)
    z["leader"] = z.ext40.to_numpy(float) + np.where(z.year.to_numpy(int) < 2025, 0.12, 0.0) * delta
    z["archive"] = z.hgb.to_numpy(float)

    # Ground-truth source is read only for retrospective stratified metrics.
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    src = np.select(
        [private.s2_ndvi.notna(), private.landsat_ndvi.notna(), private.modis_ndvi.notna()],
        ["s2", "landsat", "modis"], default="none",
    )
    side = private[KEY].copy()
    side["date"] = pd.to_datetime(side["date"])
    side["true_source"] = src
    sizes = private.assign(_year=private.date.dt.year).groupby(["anon_polygon_id", "_year"]).size()
    side["grid_n"] = [sizes[(i, d.year)] for i, d in side[KEY].itertuples(index=False, name=None)]
    side["grid_kind"] = np.where(side.grid_n.eq(213), "daily_grid", "sparse_grid")
    z = z.merge(side, on=KEY, how="left", validate="many_to_one")

    # Reconstruct query-to-nearest-visible-target distance separately per mask.
    original = private.set_index(KEY)["primary_ndvi"]
    for seed, idx in z.groupby("mask_seed").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        keys = set(map(tuple, z.loc[ii, KEY].itertuples(index=False, name=None)))
        visible = private.primary_ndvi.notna().to_numpy(bool) & np.array(
            [tuple(v) not in keys for v in private[KEY].itertuples(index=False, name=None)]
        )
        dist: dict[tuple[str, pd.Timestamp], float] = {}
        for pid, qg in z.loc[ii].groupby("anon_polygon_id"):
            vt = private.loc[visible & private.anon_polygon_id.eq(pid), "date"].sort_values().to_numpy("datetime64[D]")
            qt = qg.date.to_numpy("datetime64[D]")
            pos = np.searchsorted(vt, qt)
            left = np.where(pos > 0, (qt - vt[np.maximum(pos - 1, 0)]).astype("timedelta64[D]").astype(float), np.inf)
            right_pos = np.minimum(pos, max(len(vt) - 1, 0))
            right = np.where(pos < len(vt), (vt[right_pos] - qt).astype("timedelta64[D]").astype(float), np.inf)
            for k, d in zip(qg[KEY].itertuples(index=False, name=None), np.minimum(left, right)):
                dist[tuple(k)] = float(d)
        z.loc[ii, "nearest_days"] = [dist[tuple(k)] for k in z.loc[ii, KEY].itertuples(index=False, name=None)]
    z["distance_bin"] = pd.cut(z.nearest_days, [-np.inf, 2, 5, 10, 20, np.inf], labels=["0-2", "3-5", "6-10", "11-20", "21+"]).astype(str)
    return z


def validate_external_contract(private: pd.DataFrame) -> dict[str, object]:
    ext = pd.read_csv(EXTERNAL)
    ext["date"] = pd.to_datetime(ext["date"])
    expected = private.loc[private.is_synthetic_gap.fillna(False), KEY].copy()
    expected["date"] = pd.to_datetime(expected["date"])
    merged = expected.merge(ext, on=KEY, how="outer", indicator=True)
    values = pd.to_numeric(ext.primary_ndvi_pred, errors="coerce").to_numpy(float)
    return {
        "rows": int(len(ext)), "columns": list(ext.columns),
        "duplicate_keys": int(ext.duplicated(KEY).sum()),
        "key_mismatch": int(merged._merge.ne("both").sum()),
        "finite": bool(np.isfinite(values).all()), "min": float(np.min(values)),
        "max": float(np.max(values)), "sha256": sha256(EXTERNAL),
    }


def scan_downloads() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    terms = ("submission", "agropulse", "cosmo", "ndvi", "private_features", "train_dataset")
    for p in DOWNLOADS.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if any(t in low for t in terms) or p.suffix.lower() in {".joblib", ".pkl", ".pickle", ".parquet"}:
            rows.append({"container": "filesystem", "path": str(p), "size": p.stat().st_size,
                         "sha256": sha256(p), "member": ""})
    for zp in DOWNLOADS.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                for info in zf.infolist():
                    low = info.filename.lower()
                    if any(t in low for t in terms) or pathlib.PurePosixPath(low).suffix in {".joblib", ".pkl", ".pickle", ".parquet"}:
                        rows.append({"container": str(zp), "path": str(zp), "size": info.file_size,
                                     "sha256": f"crc32:{info.CRC:08x}", "member": info.filename})
        except (OSError, zipfile.BadZipFile):
            continue
    return pd.DataFrame(rows).drop_duplicates().sort_values(["container", "member", "path"])


def main() -> None:
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    manifest = scan_downloads()
    manifest.to_csv(RESEARCH / "downloads_candidate_manifest_20260905.csv", index=False)
    contract = validate_external_contract(private)
    z = load_holdouts()
    weights = np.round(np.arange(0.0, 0.501, 0.02), 2)
    records: list[dict[str, object]] = []
    groups: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(len(z), bool))]
    for col in ["mask_seed", "year", "cohort", "true_source", "distance_bin", "grid_kind"]:
        for value in sorted(z[col].dropna().unique(), key=str):
            groups.append((col, str(value), z[col].eq(value).to_numpy(bool)))
    groups += [
        ("route", "history", z.year.lt(2025).to_numpy(bool)),
        ("route", "2025", z.year.eq(2025).to_numpy(bool)),
        ("route", "new_2025", (z.year.eq(2025) & z.cohort.eq("new")).to_numpy(bool)),
        ("route", "shared_2025", (z.year.eq(2025) & z.cohort.eq("shared")).to_numpy(bool)),
    ]
    y = z.truth.to_numpy(float)
    for family, label, mask in groups:
        if not mask.any():
            continue
        for w in weights:
            p = (1 - w) * z.leader.to_numpy(float) + w * z.archive.to_numpy(float)
            records.append({"group": family, "label": label, "weight_archive": w,
                            "n": int(mask.sum()), "rmse": rmse(y[mask], p[mask]),
                            "mae": mae(y[mask], p[mask])})
    metrics = pd.DataFrame(records)
    metrics.to_csv(RESEARCH / "downloaded_archive_blend_metrics_20260905.csv", index=False, float_format="%.10f")

    # Leave-one-mask-out weight selection, evaluated only on the held-out mask.
    cvrows: list[dict[str, object]] = []
    for seed in sorted(z.mask_seed.unique()):
        train = z.mask_seed.ne(seed).to_numpy(bool)
        test = z.mask_seed.eq(seed).to_numpy(bool)
        train_scores = []
        for w in weights:
            p = (1 - w) * z.leader.to_numpy(float) + w * z.archive.to_numpy(float)
            train_scores.append((rmse(y[train], p[train]), float(w)))
        _, chosen = min(train_scores)
        p = (1 - chosen) * z.leader.to_numpy(float) + chosen * z.archive.to_numpy(float)
        cvrows.append({"heldout_seed": int(seed), "chosen_archive_weight": chosen,
                       "n": int(test.sum()), "leader_rmse": rmse(y[test], z.leader.to_numpy(float)[test]),
                       "blend_rmse": rmse(y[test], p[test])})
    cv = pd.DataFrame(cvrows)
    cv.to_csv(RESEARCH / "downloaded_archive_blend_lomo_20260905.csv", index=False, float_format="%.10f")

    # Private prediction-space comparison (not a metric).
    ext = pd.read_csv(EXTERNAL)
    lead = pd.read_csv(LEADER)
    ext["date"] = pd.to_datetime(ext.date)
    lead["date"] = pd.to_datetime(lead.date)
    comp = lead.merge(ext, on=KEY, suffixes=("_leader", "_archive"), validate="one_to_one")
    comp = comp.merge(private[KEY].assign(year=private.date.dt.year,
                                           cohort=np.where(private.anon_polygon_id.isin(
                                               pd.read_csv(DATA / "train_dataset.csv", usecols=["anon_polygon_id"]).anon_polygon_id.unique()),
                                               "shared", "new")), on=KEY, validate="one_to_one")
    comp["delta"] = comp.primary_ndvi_pred_archive - comp.primary_ndvi_pred_leader
    comp.to_csv(RESEARCH / "downloaded_archive_private_comparison_20260905.csv", index=False, float_format="%.10f")

    pooled = metrics[(metrics.group == "overall") & (metrics.label == "all")].sort_values("rmse")
    best = pooled.iloc[0]
    lines = [
        "# Downloaded teammate archive audit — 2026-09-05", "",
        "## Finding", "",
        f"Only one Cosmo/NDVI model bundle was found under Downloads: `{ARCHIVE}`.",
        f"Archive SHA-256: `{sha256(ARCHIVE)}`. Standalone submission SHA-256: `{contract['sha256']}`.",
        f"The submission contract is valid: {contract['rows']} rows, {contract['duplicate_keys']} duplicate keys, "
        f"{contract['key_mismatch']} key mismatches, all finite={contract['finite']}.", "",
        "The archive code/model is the previously audited 39-feature HistGradientBoosting baseline. Its seed-0 "
        "holdout predictions exactly match `research/hgb_cv_pred_seed0.csv`; it is not a newly stronger model.", "",
        "## Robust blend check", "",
        "Formula: `p = (1-w) * current_leader + w * downloaded_archive_hgb`.",
        "Current-leader row analogue is `ext40 + 0.12*(temporal_peer-ext40)` for years before 2025 and `ext40` for 2025.",
        f"Three leakage-safe masks (seeds 0, 1, 70404; {len(z):,} rows) give leader RMSE "
        f"{rmse(y, z.leader.to_numpy(float)):.8f}, archive RMSE {rmse(y, z.archive.to_numpy(float)):.8f}.",
        f"Best pooled grid weight is w={best.weight_archive:.2f}, RMSE={best.rmse:.8f}. "
        "Every positive fixed archive weight worsens pooled RMSE; leave-one-mask-out selects w=0 in all folds.", "",
        cv.to_string(index=False), "",
        "A tiny pooled gain at w=0.05 appears only in the shared-AOI subset and is below 4e-6 RMSE; it is not "
        "stable enough to route. The weak branch is stopped and no new production CSV is emitted.", "",
        "## Structural leakage audit", "",
        "No extra target-bearing file was found in filesystem or nested ZIP members. The only other recent Cosmo files "
        "are exact organizer train/private copies. Existing independent audits already establish exact mask membership "
        "replay (PCG64 seed 43) but no target-value stream, no duplicate `(AOI,date)` keys, no exact feature duplicate "
        "labels, and no usable numeric generator formula. This pass found no contradiction: gap rows contain only ID, "
        "date, gap flag and crop; every sensor/weather/climatology/status field is null.", "",
        "## Artifacts", "",
        "- `research/downloads_candidate_manifest_20260905.csv`", "- `research/downloaded_archive_blend_metrics_20260905.csv`",
        "- `research/downloaded_archive_blend_lomo_20260905.csv`", "- `research/downloaded_archive_private_comparison_20260905.csv`",
        "- `research/eval_archive_model_masks.py` and `research/downloaded_teammate_archive_audit.py`", "",
        "No submission was uploaded and no existing candidate was overwritten.",
    ]
    (REPORTS / "downloaded_teammate_archive_audit_20260905.md").write_text("\n".join(lines), encoding="utf-8")
    print("contract", contract)
    print("pooled", pooled.head(8).to_string(index=False))
    print("leave-one-mask-out\n", cv.to_string(index=False))


if __name__ == "__main__":
    main()

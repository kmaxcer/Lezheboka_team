"""Однокомандный аудит воспроизводимости и контракта submission.

Команда работает с входами только для чтения и отказывается перезаписывать
манифест. Она фиксирует хеши, размеры наборов, ключи синтетических пропусков
и при наличии независимый RMSE/GapScore по эталону. Этот артефакт позволяет
эксперту повторить проверенный пакетный запуск без состояния ноутбука
и скрытых глобальных переменных.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_csv_auto  # noqa: E402
from run_batch_inference import validate_candidate  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gap_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="JSON path; existing files are never overwritten")
    args = ap.parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.manifest}")
    for p in (args.train, args.private, args.candidate):
        if not p.exists():
            raise FileNotFoundError(p)

    train = read_csv_auto(args.train, low_memory=False)
    private = read_csv_auto(args.private, low_memory=False)
    candidate = read_csv_auto(args.candidate, low_memory=False)
    hidden = gap_mask(private["is_synthetic_gap"])
    checked = validate_candidate(private, candidate, expected_rows=int(hidden.sum()))

    result: dict[str, object] = {
        "schema_version": 1,
        "command": "scripts/reproducibility_audit.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "inputs": {
            "train": {"path": str(args.train.resolve()), "sha256": sha256(args.train), "rows": int(len(train)), "columns": list(train.columns)},
            "private": {"path": str(args.private.resolve()), "sha256": sha256(args.private), "rows": int(len(private)), "columns": list(private.columns)},
        },
        "synthetic_gap_rows": int(hidden.sum()),
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256(args.candidate),
            "rows": int(len(checked)),
            "columns": list(checked.columns),
            "unique_keys": int(checked[["anon_polygon_id", "date"]].drop_duplicates().shape[0]),
            "finite": bool(np.isfinite(checked["primary_ndvi_pred"].to_numpy(float)).all()),
            "prediction_min": float(checked.primary_ndvi_pred.min()),
            "prediction_max": float(checked.primary_ndvi_pred.max()),
        },
        "ground_truth": None,
    }
    if args.ground_truth:
        gt = read_csv_auto(args.ground_truth, low_memory=False)
        target_col = "primary_ndvi" if "primary_ndvi" in gt.columns else "primary_ndvi_true"
        if target_col not in gt.columns:
            raise ValueError("ground truth must contain primary_ndvi or primary_ndvi_true")
        merged = checked.merge(gt[["anon_polygon_id", "date", target_col]], on=["anon_polygon_id", "date"], how="inner", validate="one_to_one")
        y = merged[target_col].to_numpy(float)
        pred = merged.primary_ndvi_pred.to_numpy(float)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        result["ground_truth"] = {
            "path": str(args.ground_truth.resolve()), "sha256": sha256(args.ground_truth),
            "rows_joined": int(len(merged)), "rmse": rmse,
            "gap_score": round(30.0 * max(0.0, 1.0 - rmse / 0.10), 2),
        }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

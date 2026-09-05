"""Проверяет готовый tuned-кандидат Dani относительно входного private-файла."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _flag(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=Path, default=ROOT / "_archive_inspect" / "agropulse_max_score" / "data" / "private_features.csv")
    ap.add_argument("--submission", type=Path, default=ROOT / "outputs" / "model_dani_tuned_submission.csv")
    ap.add_argument("--metadata", type=Path, default=ROOT / "outputs" / "model_dani_tuned_metadata.json")
    args = ap.parse_args()
    private = pd.read_csv(args.private, low_memory=False)
    out = pd.read_csv(args.submission, low_memory=False)
    key = ["anon_polygon_id", "date"]
    assert list(out.columns) == key + ["primary_ndvi_pred"]
    assert not out[key].duplicated().any()
    assert np.isfinite(out["primary_ndvi_pred"].to_numpy(float)).all()
    hidden = private.loc[_flag(private["is_synthetic_gap"]), key].reset_index(drop=True)
    assert out[key].reset_index(drop=True).equals(hidden)
    if args.metadata.exists():
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
        assert int(meta["rows_submission"]) == len(out)
        assert int(meta["unique_keys"]) == len(out)
    print(f"OK: {len(out)} rows, 3 columns, unique finite hidden keys")


if __name__ == "__main__":
    main()

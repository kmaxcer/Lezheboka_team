"""Safe experimental predictors and submission generation.

This module does not modify :mod:`src.infer`.  It exposes a history-augmented
wrapper that concatenates train rows with private rows before local neighbour
search, then writes a candidate submission under ``research/``.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import predict_private  # noqa: E402


def predict_history_augmented(
    private: pd.DataFrame,
    train: pd.DataFrame | None = None,
    *,
    k: int = 6,
    bin_days: int = 30,
    date_weight: float = 1.0,
) -> pd.DataFrame:
    """Predict hidden private rows using train history as local neighbours.

    The caller's row order is retained in the returned three-column frame by
    merging on the required key.  Train rows are marked non-synthetic and are
    only used as observed history; no hidden/private target is filled in.
    """
    pr = private.copy()
    if train is None:
        return predict_private(pr, k=k, bin_days=bin_days,
                               use_date_prior=True, date_weight=date_weight)
    tr = train.copy()
    tr["is_synthetic_gap"] = False
    # Keep the private schema and append only matching columns.  This avoids
    # accidentally introducing columns that could alter masked-row handling.
    cols = [c for c in pr.columns if c in tr.columns]
    frame = pd.concat([tr[cols], pr[cols]], ignore_index=True, sort=False)
    out = predict_private(frame, train=None, k=k, bin_days=bin_days,
                          use_date_prior=True, date_weight=date_weight)
    # ``predict_private`` emits synthetic rows in frame order.  The train part
    # has no synthetic flags, so every emitted key belongs to private.
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--bin-days", type=int, default=30)
    ap.add_argument("--date-weight", type=float, default=1.0)
    args = ap.parse_args()
    tr = pd.read_csv(args.train, parse_dates=["date"])
    pr = pd.read_csv(args.private, parse_dates=["date"])
    out = predict_history_augmented(pr, tr, k=max(3, args.k),
                                    bin_days=max(0, args.bin_days),
                                    date_weight=args.date_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.8f")
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()

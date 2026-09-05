"""Таксономия ошибок для заполнения скрытых пропусков.

Метки private недоступны, поэтому команда переносит шаблон синтетических дней года
на известные годы train и оценивает тот же протокол маскирования, что и настоящая
задача. Она записывает построчные диагностики и сгруппированный RMSE.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from infer import predict_private  # noqa: E402
from validate import make_fold  # noqa: E402


def _source(df: pd.DataFrame) -> np.ndarray:
    s2 = np.isfinite(df["s2_ndvi"].to_numpy(float))
    ls = np.isfinite(df["landsat_ndvi"].to_numpy(float))
    md = np.isfinite(df["modis_ndvi"].to_numpy(float))
    out = np.full(len(df), "none", dtype=object)
    out[s2] = "s2"
    out[~s2 & ls] = "landsat"
    out[~s2 & ~ls & md] = "modis"
    return out


def _row_meta(fold: pd.DataFrame, original: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Добавляет геометрию пропуска, истинный источник и ошибку предсказания."""
    hidden = fold["is_synthetic_gap"].astype(bool).to_numpy()
    idx = np.flatnonzero(hidden)
    out = fold.iloc[idx][["anon_polygon_id", "date", "crop_type"]].copy().reset_index(drop=True)
    out["pred"] = pred
    out["truth"] = fold.iloc[idx]["_truth"].to_numpy(float)
    out["err"] = out["pred"] - out["truth"]
    out["abs_err"] = out["err"].abs()
    # Источник определяется до маскирования по исходному индексу строки train.
    src = _source(original)
    out["source"] = src[idx]
    out["year"] = pd.to_datetime(out["date"]).dt.year.astype(int)
    out["gap_days"] = np.nan
    out["bracket_days"] = np.nan
    out["edge"] = False
    out["run_len"] = 1
    local_pos = {int(global_i): int(pos) for pos, global_i in enumerate(idx)}

    # В исходных файлах каждый фолд отсортирован по AOI и дате, но внутри группы
    # сортировка выполняется явно для корректной обработки разреженных AOI и пропусков.
    for (pid, year), g in fold.groupby(["anon_polygon_id", "_year"], sort=False):
        qidx = g.index[g["is_synthetic_gap"].astype(bool)].to_numpy()
        if len(qidx) == 0:
            continue
        known = g.index[g["primary_ndvi"].notna()]
        xq = g.loc[qidx, "date"].values.astype("datetime64[D]").astype(np.int64)
        if len(known):
            xo = g.loc[known, "date"].values.astype("datetime64[D]").astype(np.int64)
            dist = np.abs(xq[:, None] - xo[None, :])
            left = np.where(xo[None, :] < xq[:, None], xq[:, None] - xo[None, :], np.inf).min(axis=1)
            right = np.where(xo[None, :] > xq[:, None], xo[None, :] - xq[:, None], np.inf).min(axis=1)
            gap = np.minimum(left, right)
            bracket = left + right
            edge = ~np.isfinite(left) | ~np.isfinite(right)
            loc = [local_pos[int(v)] for v in qidx]
            out.loc[loc, "gap_days"] = gap
            out.loc[loc, "bracket_days"] = bracket
            out.loc[loc, "edge"] = edge
        # Длина серии основана на соседних скрытых датах, а не на соседстве строк.
        qsorted = g.loc[qidx].sort_values("date")
        qdates = pd.Series(pd.to_datetime(qsorted["date"]).to_numpy())
        lengths = []
        start = 0
        for j in range(1, len(qdates) + 1):
            if j == len(qdates) or (qdates.iloc[j] - qdates.iloc[j - 1]).days != 1:
                lengths.extend([j - start] * (j - start))
                start = j
        out.loc[[local_pos[int(v)] for v in qsorted.index], "run_len"] = lengths
    return out


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        ("all", rows),
        ("year", rows), ("source", rows), ("crop_type", rows),
        ("run_len", rows), ("edge", rows),
    ]
    records = []
    for dim, group in dimensions:
        groups = [("all", group)] if dim == "all" else group.groupby(dim, dropna=False)
        for key, g in groups:
            e = g["err"].to_numpy(float)
            records.append({
                "dimension": dim, "group": str(key), "n": len(g),
                "rmse": float(np.sqrt(np.mean(e * e))),
                "mae": float(np.mean(np.abs(e))),
            })
    # Добавляет полезное непрерывное расстояние до скобки без раздувания результата.
    b = rows.copy()
    b["bracket_bin"] = pd.cut(
        b["bracket_days"], bins=[-np.inf, 7, 15, 30, np.inf],
        labels=["<=7", "8-15", "16-30", ">30"],
    )
    for key, g in b.groupby("bracket_bin", observed=False, dropna=False):
        e = g["err"].to_numpy(float)
        records.append({"dimension": "bracket_bin", "group": str(key), "n": len(g),
                        "rmse": float(np.sqrt(np.mean(e * e))),
                        "mae": float(np.mean(np.abs(e)))})
    return pd.DataFrame(records)


def run(train_path: Path, private_path: Path, output_dir: Path,
        years: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path, parse_dates=["date"])
    private = pd.read_csv(private_path, parse_dates=["date"])
    years = years or [2019, 2020, 2021, 2022, 2023, 2024]
    rows = []
    for year in years:
        fold, truth = make_fold(train, private, year)
        if len(truth) == 0:
            continue
        out = predict_private(fold, k=8)
        rows.append(_row_meta(fold, train, out.primary_ndvi_pred.to_numpy(float)))
    if not rows:
        raise ValueError("no evaluable rows; check paths and years")
    detail = pd.concat(rows, ignore_index=True)
    summary = _summary(detail)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "error_rows.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "error_summary.csv", index=False, encoding="utf-8")
    return detail, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--years", nargs="*", type=int)
    args = ap.parse_args()
    detail, summary = run(args.train, args.private, args.output_dir, args.years)
    print(f"rows={len(detail)}")
    print(summary.sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()

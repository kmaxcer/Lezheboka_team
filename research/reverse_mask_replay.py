"""Replay the private synthetic-gap mask without touching model outputs.

The challenge file keeps ``primary_ndvi`` null after masking, but the eligible
population can be recovered as ``primary_ndvi.notna() | is_synthetic_gap``.
This utility verifies the exact NumPy RNG rule found during the reverse audit
and writes diagnostics under ``research/reverse_mask_*`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE = ROOT / "_archive_inspect" / "agropulse_max_score" / "data" / "private_features.csv"
OUT_DIR = ROOT / "research"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy(copy=False)
    return s.fillna(False).astype(bool).to_numpy()


def replay(private: pd.DataFrame, seed: int = 43, fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray, dict]:
    observed = private["primary_ndvi"].notna().to_numpy()
    declared = as_bool(private["is_synthetic_gap"])
    eligible = observed | declared
    eligible_rows = np.flatnonzero(eligible)
    n_hide = int(fraction * len(eligible_rows))
    # The generator samples ordinals in the eligible population, then maps
    # those ordinals back to CSV rows.
    selected_ord = np.random.default_rng(seed).choice(len(eligible_rows), n_hide, replace=False)
    predicted_hidden = np.zeros(len(private), dtype=bool)
    predicted_hidden[eligible_rows[selected_ord]] = True
    exact = bool(np.array_equal(predicted_hidden, declared))
    info = {
        "seed": int(seed),
        "fraction": float(fraction),
        "eligible_rows": int(len(eligible_rows)),
        "hidden_rows_declared": int(declared.sum()),
        "hidden_rows_replayed": int(predicted_hidden.sum()),
        "exact_match": exact,
        "mismatch_rows": int(np.count_nonzero(predicted_hidden != declared)),
    }
    return declared, predicted_hidden, info


def seed_scan(private: pd.DataFrame, seeds: range, fraction: float = 0.15) -> pd.DataFrame:
    observed = private["primary_ndvi"].notna().to_numpy()
    declared = as_bool(private["is_synthetic_gap"])
    eligible_rows = np.flatnonzero(observed | declared)
    n_hide = int(fraction * len(eligible_rows))
    rows = []
    for seed in seeds:
        selected = np.random.default_rng(seed).choice(len(eligible_rows), n_hide, replace=False)
        pred = np.zeros(len(private), dtype=bool)
        pred[eligible_rows[selected]] = True
        overlap = int(np.count_nonzero(pred & declared))
        rows.append(
            {
                "seed": int(seed),
                "overlap": overlap,
                "mismatch": int(np.count_nonzero(pred != declared)),
                "exact": bool(np.array_equal(pred, declared)),
            }
        )
    return pd.DataFrame(rows).sort_values(["exact", "overlap"], ascending=[False, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--seed-scan-max", type=int, default=200)
    args = ap.parse_args()

    private = pd.read_csv(args.private, low_memory=False)
    declared, replayed, info = replay(private, args.seed, args.fraction)
    if not info["exact_match"]:
        raise SystemExit(f"mask replay failed: {info}")

    dt = pd.to_datetime(private["date"], errors="coerce")
    eligible = private["primary_ndvi"].notna().to_numpy() | declared
    eligible_ord = np.full(len(private), -1, dtype=np.int64)
    eligible_ord[eligible] = np.arange(int(eligible.sum()), dtype=np.int64)
    audit = private[["anon_polygon_id", "date", "is_synthetic_gap", "primary_ndvi", "crop_type"]].copy()
    audit.insert(0, "csv_row", np.arange(len(private), dtype=np.int64))
    audit["date"] = dt.dt.strftime("%Y-%m-%d")
    audit["eligible"] = eligible
    audit["eligible_ordinal"] = eligible_ord
    audit["replayed_hidden"] = replayed
    audit["replay_match"] = replayed == declared
    audit.to_csv(OUT_DIR / "reverse_mask_indices.csv", index=False)

    scan = seed_scan(private, range(max(0, int(args.seed_scan_max)) + 1), args.fraction)
    scan.to_csv(OUT_DIR / "reverse_mask_seed_scan.csv", index=False)

    hidden = private.loc[declared].copy()
    hidden["_year_from_date"] = dt.loc[declared].dt.year.to_numpy()
    known = private.loc[~declared].copy()
    by_year = (
        pd.DataFrame({"year": dt.dt.year, "hidden": declared, "eligible": eligible})
        .groupby("year")
        .agg(rows=("hidden", "size"), hidden=("hidden", "sum"), eligible=("eligible", "sum"))
        .reset_index()
    )
    by_year["hidden_rate_of_eligible"] = by_year["hidden"] / by_year["eligible"]
    by_year.to_csv(OUT_DIR / "reverse_mask_by_year.csv", index=False)

    dynamic = [
        "s2_ndvi",
        "s2_evi",
        "s2_ndwi",
        "landsat_ndvi",
        "landsat_evi",
        "landsat_ndwi",
        "modis_ndvi",
        "modis_evi",
        "era5_temp_c",
        "era5_precip_mm",
        "year",
        "primary_ndvi",
        "doy",
        "ndvi_climatology_mean",
        "ndvi_climatology_std",
        "n_reference_years",
    ]
    dynamic_present = [c for c in dynamic if c in private.columns]
    info.update(
        {
            "private_rows": int(len(private)),
            "private_sha256": sha256(args.private),
            "hidden_dynamic_all_null": bool(private.loc[declared, dynamic_present].isna().all().all()),
            "hidden_dynamic_missing_share": float(private.loc[declared, dynamic_present].isna().mean().mean()),
            "hidden_dates": int(private.loc[declared, "date"].nunique()),
            "hidden_aoi": int(private.loc[declared, "anon_polygon_id"].nunique()),
            "known_target_rows": int(private["primary_ndvi"].notna().sum()),
            "declared_eligible_rows": int(eligible.sum()),
            "seed_scan_best_non43": scan.loc[scan.seed != args.seed].iloc[0].to_dict() if len(scan) > 1 else None,
        }
    )
    (OUT_DIR / "reverse_mask_summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    top = scan.head(8)
    report = f"""# Reverse audit: exact private mask replay

Дата запуска: 2026-09-05.

## Результат

В private восстановлено точное правило формирования synthetic gaps:

1. eligible-популяция — строки, где исходный target был доступен:
   `primary_ndvi.notna() OR is_synthetic_gap`;
2. размер eligible-популяции: **{info['eligible_rows']}**;
3. скрывается `int(0.15 * N) = {info['hidden_rows_declared']}` строк;
4. выбираются ordinal-индексы через `np.random.default_rng(43).choice(N, 3112,
   replace=False)`.

Replay совпал с каждым из **{info['hidden_rows_declared']} / {info['hidden_rows_declared']}**
флагов (`mismatch_rows = 0`). Это reverse-engineering маски, а не восстановление
самих target-значений: скрытые динамические поля остаются пустыми.

## Проверка уникальности seed

Сканирование seed `0..{args.seed_scan_max}` выполнено для той же eligible-популяции.
Лучшие строки:

```text
{top.to_string(index=False)}
```

Seed 43 даёт точное совпадение; альтернативы дают лишь случайное частичное
пересечение. Повторяемость подтверждает, что порядок CSV и ordinal-популяция
сохранены.

## Что это даёт модели

Знание seed не раскрывает значения `primary_ndvi`, потому что target и все
сенсорные/погодные поля в gap-строках замаскированы. Практическая польза —
строить **точные synthetic folds** на известных target-строках и отдельно
проверять гипотезы для private-паттерна без подгонки по приблизительной маске.
Отдельного submission-кандидата только на replay маски не создавалось.

## Артефакты

- `reverse_mask_replay.py` — воспроизводимый replay;
- `reverse_mask_indices.csv` — строковые индексы, eligible-ordinal и проверка;
- `reverse_mask_seed_scan.csv` — сканирование seed;
- `reverse_mask_by_year.csv` — контрольные количества;
- `reverse_mask_summary.json` — машиночитаемая сводка.

Входной private SHA256: `{info['private_sha256']}`. Файлы `outputs/model_dani_tuned*`
не читаются для построения replay и не изменялись.
"""
    (OUT_DIR / "reverse_mask_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


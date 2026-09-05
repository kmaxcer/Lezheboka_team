"""Пакетный анализ аномалий для любого совместимого с организатором CSV региона.

Команда принимает новый файл AOI/теста без изменения кода, при необходимости
подтягивает артефакт предсказаний для синтетических строк и записывает
обогащённые строки, непрерывные периоды и сводку качества по AOI.
Существующие результаты не перезаписываются.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_csv_auto
from anomaly import enrich_regions

def _mask(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))

def main(argv=None):
    ap = argparse.ArgumentParser(description="Запускает leakage-safe анализ аномалий по нескольким регионам")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference", type=Path, help="наблюдаемый CSV train/эталона")
    ap.add_argument("--predictions", type=Path, help="кандидат с primary_ndvi_pred")
    ap.add_argument("--periods", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--seasonal-window", type=int, default=15)
    args = ap.parse_args(argv)
    targets = [args.output, args.periods, args.summary]
    existing = [str(p) for p in targets if p is not None and p.exists()]
    if existing: raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    frame = read_csv_auto(args.input, low_memory=False)
    values = pd.to_numeric(frame.get("primary_ndvi", pd.Series(float("nan"), index=frame.index)), errors="coerce")
    if args.predictions:
        pred = read_csv_auto(args.predictions, low_memory=False).copy()
        for d in (frame, pred):
            d["anon_polygon_id"] = d["anon_polygon_id"].astype(str)
            d["date"] = pd.to_datetime(d["date"], errors="raise").dt.strftime("%Y-%m-%d")
        pi = pd.MultiIndex.from_frame(pred[["anon_polygon_id", "date"]])
        lookup = pd.Series(pd.to_numeric(pred["primary_ndvi_pred"], errors="coerce").to_numpy(), index=pi)
        idx = pd.MultiIndex.from_frame(frame[["anon_polygon_id", "date"]])
        use = _mask(frame["is_synthetic_gap"]) if "is_synthetic_gap" in frame else values.isna()
        fill = lookup.reindex(idx).to_numpy()
        values.loc[use] = fill[use.to_numpy()]
    reference = read_csv_auto(args.reference, low_memory=False) if args.reference else None
    enriched, periods, summary = enrich_regions(frame, values=values, reference_frame=reference,
                                                  seasonal_window=args.seasonal_window)
    args.output.parent.mkdir(parents=True, exist_ok=True); enriched.to_csv(args.output, index=False, encoding="utf-8")
    if args.periods: args.periods.parent.mkdir(parents=True, exist_ok=True); periods.to_csv(args.periods, index=False, encoding="utf-8")
    if args.summary: args.summary.parent.mkdir(parents=True, exist_ok=True); summary.to_csv(args.summary, index=False, encoding="utf-8")
    print(f"rows={len(enriched)} regions={len(summary)} anomalies={int(summary.anomaly_n.sum())} output={args.output}")

if __name__ == "__main__": main()

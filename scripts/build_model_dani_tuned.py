"""Строит tuned-модель NDVI Dani и готовый CSV-кандидат.

Сборщик объединяет leakage-safe модель HistGradientBoosting из архива
с независимо проверенным локальным интерполятором с лагами из
``src/infer_lag.py``. Входные CSV не изменяются. По умолчанию данные
читаются из ``_archive_inspect/agropulse_max_score/data``; для другой копии
можно указать пути ``--train`` и ``--private``.

Результаты (в ``--output-dir``):

* ``model_dani_tuned_submission.csv`` — итоговый трёхколоночный кандидат;
* ``model_dani_tuned_hgb.csv`` и ``model_dani_tuned_lag.csv`` — компоненты;
* ``model_dani_tuned.joblib`` — обученный оцениватель HGB и конфигурация;
* ``model_dani_tuned_metadata.json`` — хеши, параметры и размеры файлов.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "_archive_inspect" / "agropulse_max_score"


def _default_path(name: str) -> Path:
    candidates = (
        ARCHIVE_ROOT / "data" / name,
        ROOT / "data" / name,
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_archive_pipeline():
    source_root = ARCHIVE_ROOT / "src"
    if not source_root.exists():
        raise FileNotFoundError(
            f"Не найден bundled pipeline: {source_root}. "
            "Оставьте распакованный архив в _archive_inspect или передайте его обратно."
        )
    sys.path.insert(0, str(source_root))
    from agropulse.pipeline import (  # type: ignore
        FULL_FEATURES,
        fit_final_model,
        load_competition_data,
        predict_submission,
    )
    return FULL_FEATURES, fit_final_model, load_competition_data, predict_submission


def _read_private(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    if "is_synthetic_gap" not in frame:
        frame["is_synthetic_gap"] = False
    return frame


def _as_bool(values: pd.Series) -> pd.Series:
    """Безопасно разбирает флаг, даже если CSV хранит его как текст."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _merge_predictions(
    private: pd.DataFrame,
    hgb: pd.DataFrame,
    lag: pd.DataFrame,
    lag_weight: float,
) -> pd.DataFrame:
    key = ["anon_polygon_id", "date"]
    hidden = private.loc[_as_bool(private["is_synthetic_gap"]), key].copy()
    hidden["date"] = pd.to_datetime(hidden["date"])
    h = hgb.copy(); h["date"] = pd.to_datetime(h["date"])
    l = lag.copy(); l["date"] = pd.to_datetime(l["date"])
    if h[key].duplicated().any() or l[key].duplicated().any():
        raise ValueError("Компонентная submission содержит дубликаты ключа")
    joined = h.merge(
        l,
        on=key,
        how="outer",
        validate="one_to_one",
        suffixes=("_hgb", "_lag"),
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("HGB и lag компоненты имеют разные synthetic-ключи")
    w = float(np.clip(lag_weight, 0.0, 1.0))
    joined["primary_ndvi_pred"] = (
        (1.0 - w) * joined["primary_ndvi_pred_hgb"].to_numpy(float)
        + w * joined["primary_ndvi_pred_lag"].to_numpy(float)
    )
    # Сохраняет широкое ограничение архивной модели. Оно влияет только на
    # невозможную экстраполяцию и оставляет входы скорера физически правдоподобными.
    joined["primary_ndvi_pred"] = np.clip(
        joined["primary_ndvi_pred"].to_numpy(float), -0.2, 1.1
    )
    final = hidden.merge(
        joined[key + ["primary_ndvi_pred"]],
        on=key,
        how="left",
        validate="one_to_one",
    )
    if len(final) != len(hidden) or not np.isfinite(final["primary_ndvi_pred"]).all():
        raise ValueError("Итоговая submission неполна или содержит NaN/Inf")
    return final


def build(
    train_path: Path,
    private_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    lag_weight: float = 0.20,
) -> dict[str, Any]:
    if not train_path.exists() or not private_path.exists():
        raise FileNotFoundError(f"Нет входного файла: {train_path} / {private_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_features, fit_final_model, load_competition_data, predict_submission = _load_archive_pipeline()

    # Загрузчик архива выполняет каноническую сортировку train+private и
    # заполняет year/doy из даты. Модель обучается заново; готовые submission
    # и модель из ZIP молча не переиспользуются.
    train, private_from_loader, reference = load_competition_data(train_path, private_path)
    hgb_model, _ = fit_final_model(reference, seed=int(seed))
    hgb_path = output_dir / "model_dani_tuned_hgb.csv"
    hgb = predict_submission(reference, hgb_model, hgb_path)

    private = _read_private(private_path)
    sys.path.insert(0, str(ROOT / "src"))
    from infer_lag import predict_private_lag  # type: ignore

    lag = predict_private_lag(
        private,
        train=train,
        k=16,
        degree=3,
        bin_days=30,
        use_date_prior=True,
        date_weight=1.0,
    )
    lag_path = output_dir / "model_dani_tuned_lag.csv"
    lag.to_csv(lag_path, index=False, encoding="utf-8", float_format="%.8f")

    final = _merge_predictions(private, hgb, lag, lag_weight)
    final_path = output_dir / "model_dani_tuned_submission.csv"
    final.to_csv(final_path, index=False, encoding="utf-8", float_format="%.8f")

    artifact = {
        "name": "model_dani_tuned",
        "kind": "hgb_histgradientboosting_plus_lag_local_cubic",
        "hgb_model": hgb_model,
        "hgb_features": list(full_features),
        "hgb_seed": int(seed),
        "lag": {
            "k": 16,
            "degree": 3,
            "bin_days": 30,
            "date_weight": 1.0,
        },
        "lag_weight": float(np.clip(lag_weight, 0.0, 1.0)),
        "submission_columns": ["anon_polygon_id", "date", "primary_ndvi_pred"],
    }
    model_path = output_dir / "model_dani_tuned.joblib"
    joblib.dump(artifact, model_path)

    metadata = {
        "name": "model_dani_tuned",
        "kind": artifact["kind"],
        "train_path": str(train_path.resolve()),
        "private_path": str(private_path.resolve()),
        "train_sha256": _sha256(train_path),
        "private_sha256": _sha256(private_path),
        "rows_train": int(len(train)),
        "rows_private": int(len(private)),
        "rows_submission": int(len(final)),
        "synthetic_rows": int(_as_bool(private["is_synthetic_gap"]).sum()),
        "hgb_seed": int(seed),
        "lag_weight": float(np.clip(lag_weight, 0.0, 1.0)),
        "lag_parameters": artifact["lag"],
        "prediction_min": float(final["primary_ndvi_pred"].min()),
        "prediction_max": float(final["primary_ndvi_pred"].max()),
        "prediction_mean": float(final["primary_ndvi_pred"].mean()),
        "prediction_std": float(final["primary_ndvi_pred"].std()),
        "columns": list(final.columns),
        "unique_keys": int(final[["anon_polygon_id", "date"]].drop_duplicates().shape[0]),
    }
    metadata_path = output_dir / "model_dani_tuned_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "final": final_path,
        "hgb": hgb_path,
        "lag": lag_path,
        "model": model_path,
        "metadata": metadata_path,
        "metadata_obj": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=_default_path("train_dataset.csv"))
    parser.add_argument("--private", type=Path, default=_default_path("private_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lag-weight", type=float, default=0.20)
    args = parser.parse_args()
    result = build(
        args.train,
        args.private,
        args.output_dir,
        seed=args.seed,
        lag_weight=args.lag_weight,
    )
    meta = result["metadata_obj"]
    print(f"Готово: {result['final']}")
    print(f"Строк: {meta['rows_submission']}; диапазон: {meta['prediction_min']:.6f}..{meta['prediction_max']:.6f}")
    print(f"HGB/lag blend: {1-meta['lag_weight']:.0%}/{meta['lag_weight']:.0%}")


if __name__ == "__main__":
    main()

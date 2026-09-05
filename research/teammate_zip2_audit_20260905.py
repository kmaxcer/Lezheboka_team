"""Проверяет второй архив сокомандника без изменения исходных артефактов."""
from pathlib import Path
import sys
import json
import hashlib

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tmp/teammate_zip2_audit_20260905_2152/MonitoringOfVegetationDynamics"
sys.path.insert(0, str(ARCHIVE / "src"))
from vegetation_monitoring.data import read_dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(frame):
    return {
        "rows": len(frame), "columns": frame.columns.tolist(),
        "duplicates": int(frame.duplicated(["anon_polygon_id", "date"]).sum()),
        "finite": bool(np.isfinite(frame.primary_ndvi_pred).all()),
        "min": float(frame.primary_ndvi_pred.min()), "max": float(frame.primary_ndvi_pred.max()),
    }


def save_new(frame, path):
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def main():
    result = {"archive_sha256": digest(ROOT / "MonitoringOfVegetationDynamics (2).zip")}
    submission_path = ARCHIVE / "outputs/submission.csv"
    submission = pd.read_csv(submission_path)
    result["archived_submission"] = contract(submission)
    result["archived_submission"]["sha256"] = digest(submission_path)
    truth = pd.read_csv(ROOT / "research/data_update_20260905_1350/private_test_ground_truth.csv")
    keys = ["anon_polygon_id", "date"]
    result["archived_submission"]["released_gt_overlap"] = len(submission.merge(truth, on=keys))
    private_path = Path("C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904/private_features.csv")
    frame = read_dataset(private_path)
    model = joblib.load(ARCHIVE / "models/best_model.joblib")
    mask = set(frame.index[frame.is_synthetic_gap.fillna(False).astype(bool)])
    feature_frame = model.builder.transform(frame, masked_index=mask)
    raw_members = np.column_stack([estimator.predict(feature_frame) for estimator in model.estimators])
    predictions = model.predict(frame, mask)
    prediction_frame = frame.loc[sorted(mask), keys].copy()
    prediction_frame["date"] = prediction_frame.date.dt.strftime("%Y-%m-%d")
    prediction_frame["primary_ndvi_pred"] = predictions.loc[prediction_frame.index]
    for i, estimator in enumerate(model.estimators):
        prediction_frame[f"member_{i}_{type(estimator).__name__}"] = raw_members[prediction_frame.index, i]
    output_path = ROOT / "research/teammate_zip2_released_predictions_20260905.csv"
    save_new(prediction_frame, output_path)
    joined = prediction_frame.merge(truth, on=keys, validate="one_to_one")
    result["released_inference"] = {}
    for name in prediction_frame.columns[2:]:
        rmse = float(np.sqrt(np.mean((joined[name] - joined.primary_ndvi_true) ** 2)))
        result["released_inference"][name] = {"rmse": rmse, "gap_score": round(30 * max(0, 1-rmse/.1), 2)}
    result["released_inference"]["rows"] = len(joined)
    result["released_inference"]["file"] = str(output_path)
    result["released_inference"]["sha256"] = digest(output_path)
    result["model"] = {"types": [type(e).__name__ for e in model.estimators], "weights": model.weights.tolist(), "augmentation": model.augmentation}
    result["dataset_hashes"] = {
        name: digest(ARCHIVE / "data" / name)
        for name in ["train_dataset.csv", "test_features.csv", "private_test_ground_truth.csv"]
    }
    new_test = read_dataset(ARCHIVE / "data/test_features.csv")
    new_mask = set(new_test.index[new_test.is_synthetic_gap.fillna(False).astype(bool)])
    new_predictions = model.predict(new_test, new_mask)
    new_frame = new_test.loc[sorted(new_mask), keys].copy()
    new_frame["date"] = new_frame.date.dt.strftime("%Y-%m-%d")
    new_frame["primary_ndvi_pred"] = new_predictions.loc[new_frame.index]
    consistent = submission.merge(new_frame, on=keys, suffixes=("_archive", "_reproduced"), validate="one_to_one")
    result["submission_reproduction"] = {"rows": len(consistent), "max_abs_delta": float(np.max(np.abs(consistent.primary_ndvi_pred_archive-consistent.primary_ndvi_pred_reproduced)))}
    result_path = ROOT / "research/teammate_zip2_audit_20260905.json"
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

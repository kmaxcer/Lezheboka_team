"""Leakage-safe anomaly QA on released private ground truth."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from io_utils import read_csv_auto
from anomaly import add_anomaly_columns

ROOT = Path(__file__).parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
private = read_csv_auto(DATA / "private_features.csv", low_memory=False)
train = read_csv_auto(DATA / "train_dataset.csv", low_memory=False)
gt = read_csv_auto(ROOT / "research/data_update_20260905_1350/private_test_ground_truth.csv")
pred = read_csv_auto(ROOT / "outputs/model_dani_oldgt_robust_blend_localgamma006_jointdiag_w040_20260905_submission.csv")
keys = ["anon_polygon_id", "date"]
for d in (private, train, gt, pred):
    d["anon_polygon_id"] = d["anon_polygon_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"], errors="raise").dt.strftime("%Y-%m-%d")
idx = pd.MultiIndex.from_frame(private[keys])
true_map = pd.Series(gt["primary_ndvi_true"].to_numpy(float), index=pd.MultiIndex.from_frame(gt[keys]))
pred_map = pd.Series(pred["primary_ndvi_pred"].to_numpy(float), index=pd.MultiIndex.from_frame(pred[keys]))
raw = pd.to_numeric(private["primary_ndvi"], errors="coerce")
true_values, pred_values = raw.copy(), raw.copy()
gaps = private["is_synthetic_gap"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
true_values.loc[gaps] = true_map.reindex(idx).to_numpy()[gaps.to_numpy()]
pred_values.loc[gaps] = pred_map.reindex(idx).to_numpy()[gaps.to_numpy()]
ref = train[train["primary_ndvi"].notna()].copy()
t = add_anomaly_columns(private, values=true_values, reference_frame=ref, min_samples=3)
p = add_anomaly_columns(private, values=pred_values, reference_frame=ref, min_samples=3)
mask = gaps.to_numpy() & np.isfinite(true_values.to_numpy(float)) & np.isfinite(pred_values.to_numpy(float))
y_true = t.loc[mask, "status"].isin(["suppression", "critical"]).to_numpy()
y_pred = p.loc[mask, "status"].isin(["suppression", "critical"]).to_numpy()
pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[False, True])
row = {"n_gap": int(mask.sum()), "true_anomaly_n": int(y_true.sum()), "pred_anomaly_n": int(y_pred.sum()),
       "precision": float(pr), "recall": float(rc), "f1": float(f1), "tn": int(cm[0,0]),
       "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
       "true_critical_n": int((t.loc[mask,"status"] == "critical").sum()),
       "pred_critical_n": int((p.loc[mask,"status"] == "critical").sum())}
metrics_path = ROOT / "research/anomaly_quality_released_gt_20260905.csv"
pd.DataFrame([row]).to_csv(metrics_path, index=False)
slice_rows = []
selectors = [("all", np.ones(len(private), dtype=bool)), ("year_2025", pd.to_datetime(private.date).dt.year.eq(2025).to_numpy())]
for name, sel in selectors:
    m = mask & sel
    if not m.any(): continue
    yt = t.loc[m,"status"].isin(["suppression","critical"]).to_numpy(); yp = p.loc[m,"status"].isin(["suppression","critical"]).to_numpy()
    q = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    slice_rows.append({"slice": name, "n": int(m.sum()), "true_anomaly_n": int(yt.sum()), "pred_anomaly_n": int(yp.sum()), "precision": float(q[0]), "recall": float(q[1]), "f1": float(q[2])})
slice_path = ROOT / "research/anomaly_quality_released_gt_20260905_slices.csv"
pd.DataFrame(slice_rows).to_csv(slice_path, index=False)
report = ROOT / "research/anomaly_quality_released_gt_20260905.md"
report.write_text("""# Anomaly quality audit — released ground truth\n\nThe audit joins released `primary_ndvi_true` to the 3,112 old-private synthetic gaps. A leakage-safe climatology is fit from train observations only (current-year exclusion, circular 15-day window, AOI→crop→global fallback). Labels use the documented thresholds: z < -1 suppression and z < -2 critical. The prediction side uses the reviewed candidate. Metrics compare predicted negative-anomaly flags with labels derived from released true NDVI; they are a reproducible QA, not a claim about future hidden labels.\n\nMetrics: `anomaly_quality_released_gt_20260905.csv`; slices: `anomaly_quality_released_gt_20260905_slices.csv`.\n\n""" + pd.DataFrame([row]).to_string(index=False), encoding="utf-8")
print(pd.DataFrame([row]).to_string(index=False))

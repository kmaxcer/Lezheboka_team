"""Контроль эффекта 54 HGB-конфигураций на безопасно скрытых строках.

Это собственная сетка: параметры сокомандника пока неизвестны. Released GT
используется только после predict. Файлы результатов создаются эксклюзивно.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904")
SRC = Path(r"C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\_downloads_inspect\agro\agropulse_max_score\src")
sys.path.insert(0, str(SRC))
from agropulse.pipeline import build_features, load_competition_data, FULL_FEATURES

ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"
PREFIX = ROOT / "research" / "hgb_54_safe_probe_20260905_1810"
DYNAMIC = ["s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi", "landsat_ndwi", "modis_ndvi", "modis_evi", "modis_ndwi", "era5_temp_c", "era5_precip_mm", TARGET, "ndvi_climatology_mean", "ndvi_climatology_std", "ndvi_zscore", "n_reference_years", "status"]


def safe_features(ref, mask):
    frame = ref.copy()
    for col in DYNAMIC:
        if col in frame:
            frame.loc[mask, col] = np.nan
    frame[GAP] = mask
    observed = frame[TARGET].mask(mask)
    return build_features(frame, observed, pd.Series(mask, index=frame.index)).replace([np.inf, -np.inf], np.nan)


def stratified_mask(ref, pool, seed, fraction=.16):
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(ref), bool)
    for _, ix0 in ref.loc[pool].groupby([ID, "year"], sort=False).groups.items():
        ix = np.asarray(ix0, int)
        mask[rng.choice(ix, size=max(1, round(fraction * len(ix))), replace=False)] = True
    return mask


def rmse(y, p):
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - p))))


def write_new(path, body):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        stream.write(body)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.time()
    result_path = str(PREFIX) + "_results.csv"
    if Path(result_path).exists():
        raise FileExistsError(result_path)
    train, private, ref = load_competition_data(DATA / "train_dataset.csv", DATA / "private_features.csv")
    actual = (ref["_origin"].eq("test") & ref[GAP].fillna(False)).to_numpy(bool)
    known = ref[TARGET].notna().to_numpy(bool) & ~actual
    outer = stratified_mask(ref, known & ref["_origin"].eq("test").to_numpy(), 20260905, .15)
    train_pool = known & ~outer
    hidden = actual | outer
    blocks, ys = [], []
    for seed in (11, 29, 47, 83):
        pseudo = stratified_mask(ref, train_pool, seed)
        print(f"features pseudo seed={seed}, n={int(pseudo.sum())}", flush=True)
        features = safe_features(ref, hidden | pseudo)
        blocks.append(features.loc[pseudo])
        ys.append(ref.loc[pseudo, TARGET])
    X, y = pd.concat(blocks, ignore_index=True), pd.concat(ys, ignore_index=True)
    print(f"features outer and actual; train={X.shape}", flush=True)
    query = safe_features(ref, hidden)
    xo, yo = query.loc[outer], ref.loc[outer, TARGET].to_numpy(float)
    xg = query.loc[actual]
    gt_path = ROOT / "research" / "data_update_20260905_1350" / "private_test_ground_truth.csv"
    # Проверяем полный 1:1 contract, но никогда не присоединяем truth к ref.
    gt = pd.read_csv(gt_path, parse_dates=[DATE])
    actual_keys = ref.loc[actual, [ID, DATE]]
    matched = actual_keys.merge(gt, on=[ID, DATE], how="left", validate="one_to_one")
    assert len(matched) == 3112 and matched.primary_ndvi_true.notna().all()
    yg = matched.primary_ndvi_true.to_numpy(float)
    configs = list(product((.02, .035, .07), (24, 48, 96), (35, 70), (0., 8., 30.)))
    columns = ["config", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_iter", "n_iter", "outer_n", "outer_rmse", "outer_gap_score", "released_gt_n", "released_gt_rmse", "released_gt_gap_score", "fit_seconds"]
    rows, best_outer, best_prediction = [], float("inf"), None
    with open(result_path, "x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        with threadpool_limits(limits=3):
            for number, (lr, leaves, minleaf, l2) in enumerate(configs, 1):
                start = time.time()
                spec = dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=minleaf, l2_regularization=l2, max_iter=300)
                model = HistGradientBoostingRegressor(loss="squared_error", random_state=42, early_stopping="auto", **spec)
                model.fit(X, y)
                po = np.clip(model.predict(xo), -.2, 1.1)
                pg = np.clip(model.predict(xg), -.2, 1.1)
                outer_score, gap_score = rmse(yo, po), rmse(yg, pg)
                row = {"config": number, **spec, "n_iter": int(model.n_iter_), "outer_n": len(yo), "outer_rmse": outer_score, "outer_gap_score": round(30 * max(0, 1 - outer_score / .10), 2), "released_gt_n": len(yg), "released_gt_rmse": gap_score, "released_gt_gap_score": round(30 * max(0, 1 - gap_score / .10), 2), "fit_seconds": round(time.time() - start, 2)}
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if outer_score < best_outer:
                    best_outer, best_prediction = outer_score, (number, po.copy(), pg.copy())
                print(f"{number:02d}/54 outer={outer_score:.6f} released={gap_score:.6f} best_outer={best_outer:.6f} seconds={row['fit_seconds']}", flush=True)
    metrics = pd.DataFrame(rows)
    best = metrics.sort_values("outer_rmse").iloc[0].to_dict()
    diagnostic_best = metrics.sort_values("released_gt_rmse").iloc[0].to_dict()
    slices, prediction_rows = [], []
    shared_ids = set(train[ID])
    for label, mask, truth, pred in (("outer", outer, yo, best_prediction[1]), ("released_gt", actual, yg, best_prediction[2])):
        frame = ref.loc[mask, [ID, DATE, "crop_type"]].reset_index(drop=True)
        frame["split"] = label
        frame["truth"] = truth
        frame["pred"] = pred
        frame["squared_error"] = (truth - pred) ** 2
        frame["cohort"] = np.where(frame[ID].isin(shared_ids), "shared", "new")
        frame["year"] = frame[DATE].dt.year
        distance = query.loc[mask, ["dprev1", "dnext1"]].min(axis=1).to_numpy(float)
        frame["distance"] = pd.cut(distance, [-np.inf, 2, 8, 16, np.inf], labels=["<=2", "3-8", "9-16", ">16"]).astype(str)
        if label == "outer":
            sens = ref.loc[mask]
            frame["source"] = np.select([sens.s2_ndvi.notna(), sens.landsat_ndvi.notna(), sens.modis_ndvi.notna()], ["s2", "landsat", "modis"], default="unknown")
        else:
            frame["source"] = "masked_unknown"
        frame["cohort_period"] = frame.cohort + np.where(frame.year.eq(2025), "2025", "history")
        prediction_rows.append(frame)
        for dimension in ("cohort", "year", "source", "distance", "cohort_period"):
            for group, part in frame.groupby(dimension, observed=True):
                slices.append(dict(split=label, dimension=dimension, group=str(group), n=len(part), rmse=float(np.sqrt(part.squared_error.mean()))))
    write_new(str(PREFIX) + "_slices.csv", pd.DataFrame(slices).to_csv(index=False))
    write_new(str(PREFIX) + "_predictions.csv", pd.concat(prediction_rows).to_csv(index=False))
    metadata = dict(configurations=len(rows), feature_count=len(FULL_FEATURES), train_rows=len(X), pseudo_seeds=[11, 29, 47, 83], outer_seed=20260905, outer_fraction=.15, selected_on_outer=best, diagnostic_best_released=diagnostic_best, elapsed_seconds=round(time.time()-t0, 1), gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(), actual_gt_never_in_training=True, all_outer_and_pseudo_dynamic_fields_masked=True, submission_created=False, upload_performed=False)
    write_new(str(PREFIX) + "_metadata.json", json.dumps(metadata, indent=2))
    report = "# HGB: контроль 54 конфигураций\n\n" + "Сетка: learning_rate={.02,.035,.07}; leaves={24,48,96}; min_leaf={35,70}; L2={0,8,30}; max_iter=300. Это наша контрольная сетка, параметры сокомандника неизвестны.\n\n" + "4 pseudo-mask обучающих блока, outer seed 20260905, 15% известных private по AOI/year. Target и все динамические поля outer/pseudo/real gaps скрыты до вычисления признаков; source counts тоже вычислены после маскирования. Released GT только для аудита после predict.\n\n" + f"Лучший по outer: RMSE={best['outer_rmse']:.9f}; released GT RMSE={best['released_gt_rmse']:.9f}, score={best['released_gt_gap_score']:.2f}.\n\n" + f"Диагностический минимум по released GT: {diagnostic_best['released_gt_rmse']:.9f}, score={diagnostic_best['released_gt_gap_score']:.2f}; он не является независимой оценкой после выбора по GT.\n\n" + "27.2 балла требуют RMSE 0.009333333. Этот контроль не доказывает невозможность сильного HGB с другими признаками, но измеряет эффект обычной настройки на доступном безопасном представлении данных. Один outer seed — предварительный эксперимент, кандидат не продвигается без проверки на нескольких масках.\n\n" + "Новых submission не создано, загрузки не выполнялись.\n\n```json\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n```\n"
    write_new(str(PREFIX) + "_report.md", report)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

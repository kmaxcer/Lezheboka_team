"""Leakage-safe tail/anomaly residual correction experiments.

This module is deliberately research-only.  It treats the saved Dani
``blend_lag_0.20`` predictions as a fixed base and asks whether information
which is visible around a masked row (nearby NDVI/climatology residuals and
same-date peer states) can correct the large residual tails.  No hidden target
or ``status`` value is used to build features.  The exact hidden-DOY protocol
and several private-like random masks are evaluated with leave-partition-out
calibration.

The script writes only ``tail_anomaly_v2*`` artifacts in ``research``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(RESEARCH))
from teammate_sweep_postcorr import _mask_private  # noqa: E402


BASE_METHOD = "blend_lag_0.20"
KEY = ["anon_polygon_id", "date"]
CLIP_LO, CLIP_HI = -0.5, 1.2


def _finite(a: object) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(float)


def _safe_z(y: np.ndarray, cm: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Reconstruct a normalized anomaly from visible columns only.

    ``private_features`` intentionally has no ``ndvi_zscore`` column.  The
    algebraic equivalent is available from visible primary NDVI and the
    supplied climatology, so this remains deployable at inference time.
    """
    out = np.full(len(y), np.nan, float)
    ok = np.isfinite(y) & np.isfinite(cm) & np.isfinite(cs) & (cs > 0.015)
    out[ok] = (y[ok] - cm[ok]) / cs[ok]
    return np.clip(out, -8.0, 8.0)


def _weighted(values: np.ndarray, dist: np.ndarray, tau: float,
              clip: tuple[float, float] | None = None) -> float:
    ok = np.isfinite(values) & np.isfinite(dist)
    if not ok.any():
        return np.nan
    v = values[ok]
    if clip is not None:
        v = np.clip(v, clip[0], clip[1])
    w = np.exp(-dist[ok] / max(float(tau), 1e-6))
    sw = float(w.sum())
    return float(np.dot(v, w) / sw) if sw > 0 else np.nan


def _nearest_pair(ii: np.ndarray, ordinal: np.ndarray, values: np.ndarray,
                  i: int) -> tuple[float, float, float, float]:
    """Return previous value/distance and next value/distance."""
    if len(ii) == 0:
        return np.nan, np.nan, np.nan, np.nan
    before = ii[ordinal[ii] < ordinal[i]]
    after = ii[ordinal[ii] > ordinal[i]]
    if len(before):
        b = before[np.argmax(ordinal[before])]
        vb, db = values[b], ordinal[i] - ordinal[b]
    else:
        vb, db = np.nan, np.nan
    if len(after):
        a = after[np.argmin(ordinal[after])]
        va, da = values[a], ordinal[a] - ordinal[i]
    else:
        va, da = np.nan, np.nan
    return float(vb), float(db), float(va), float(da)


def _aggregate_groups(d: pd.DataFrame, known: np.ndarray,
                      z: np.ndarray, dev: np.ndarray,
                      query_idx: np.ndarray) -> np.ndarray:
    """Construct deployable local and peer anomaly features.

    All aggregations are made after applying ``known``.  In particular, the
    query row's target, sensors, weather and climatology are never read.
    """
    n = len(query_idx)
    # Feature names are kept stable and are emitted in the output metadata.
    names: list[str] = []
    for tau in (7, 14, 30, 60, 120):
        names += [f"z_w{tau}", f"dev_w{tau}", f"z_low{tau}", f"z_high{tau}", f"n{tau}"]
    names += ["z_prev", "d_prev", "z_next", "d_next",
              "dev_prev", "dev_dprev", "dev_next", "dev_dnext",
              "z_interp", "dev_interp", "z_min", "z_max", "z_med",
              "z_neg", "z_pos", "z_low_frac", "z_high_frac"]
    names += ["peer_zmed", "peer_zmean", "peer_devmed", "peer_n",
              "peer_zlow", "peer_zhigh", "peer_zmed_w7", "peer_zmed_w30",
              "crop_zmed", "crop_n", "crop_devmed"]
    names += ["cm_local", "cs_local", "cm_n", "p0_minus_cm"]

    d = d.reset_index(drop=True)
    dt = pd.to_datetime(d["date"])
    ordinal = dt.map(pd.Timestamp.toordinal).to_numpy(float)
    ids = d["anon_polygon_id"].astype(str).to_numpy()
    years = dt.dt.year.to_numpy(int)
    y = _finite(d["primary_ndvi"])
    cm = _finite(d.get("ndvi_climatology_mean", pd.Series(np.nan, index=d.index)))
    cs = _finite(d.get("ndvi_climatology_std", pd.Series(np.nan, index=d.index)))
    # Use only fields available on an unmasked row.  Climatology is masked at
    # the query itself by both validation protocols.
    z = z.copy()
    dev = dev.copy()
    out = np.full((n, len(names)), np.nan, float)

    # Local AOI/year groups.  The grouping key is represented by strings and
    # integer years to avoid accidental cross-year leakage.
    local_groups: dict[tuple[str, int], np.ndarray] = {}
    known_good = known & np.isfinite(z)
    tmp = pd.DataFrame({"id": ids, "yr": years})
    for k, ix in tmp.loc[known_good].groupby(["id", "yr"], sort=False).groups.items():
        local_groups[(str(k[0]), int(k[1]))] = np.asarray(ix, dtype=int)

    # Date and crop/date peer tables.  Values are retained as arrays so the
    # query AOI can be excluded without ever using a label from that AOI.
    date_groups: dict[int, np.ndarray] = {}
    for k, ix in pd.DataFrame({"day": dt.dt.normalize()}).loc[known_good].groupby("day", sort=False).groups.items():
        date_groups[int(pd.Timestamp(k).toordinal())] = np.asarray(ix, dtype=int)
    crop = d.get("crop_type", pd.Series("", index=d.index)).fillna("").astype(str).to_numpy()
    crop_groups: dict[tuple[str, int], np.ndarray] = {}
    for k, ix in pd.DataFrame({"crop": crop, "day": dt.dt.normalize()}).loc[known_good].groupby(["crop", "day"], sort=False).groups.items():
        crop_groups[(str(k[0]), int(pd.Timestamp(k[1]).toordinal()))] = np.asarray(ix, dtype=int)

    # Global seasonal climatology fallback, useful for 2025-only AOIs where a
    # same-year local profile is short.
    cm_good = known & np.isfinite(cm)
    cm_by_doy: dict[int, float] = {}
    cs_by_doy: dict[int, float] = {}
    doys = dt.dt.dayofyear.to_numpy(int)
    for k, ix in pd.DataFrame({"doy": doys}).loc[cm_good].groupby("doy", sort=False).groups.items():
        vv = cm[ix]; ss = cs[ix]
        cm_by_doy[int(k)] = float(np.nanmedian(vv)) if np.isfinite(vv).any() else np.nan
        cs_by_doy[int(k)] = float(np.nanmedian(ss)) if np.isfinite(ss).any() else np.nan

    for row, i in enumerate(query_idx):
        ii = local_groups.get((ids[i], int(years[i])), np.empty(0, dtype=int))
        dist = np.abs(ordinal[ii] - ordinal[i]) if len(ii) else np.empty(0)
        col = 0
        for tau in (7, 14, 30, 60, 120):
            win = max(2 * tau, 14)
            take = dist <= win
            if take.any():
                zz = z[ii[take]]
                dd = dev[ii[take]]
                out[row, col] = _weighted(zz, dist[take], tau, (-6, 6)); col += 1
                out[row, col] = _weighted(dd, dist[take], tau, (-1, 1)); col += 1
                goodz = zz[np.isfinite(zz)]
                out[row, col] = float(np.mean(goodz < -1.0)) if len(goodz) else np.nan; col += 1
                out[row, col] = float(np.mean(goodz > 1.0)) if len(goodz) else np.nan; col += 1
                out[row, col] = float(len(goodz)); col += 1
            else:
                col += 5
        zp, dp, zn, dn = _nearest_pair(ii, ordinal, z, i)
        out[row, col:col + 4] = [zp, dp, zn, dn]; col += 4
        vp, vdp, vn, vdn = _nearest_pair(ii, ordinal, dev, i)
        out[row, col:col + 4] = [vp, vdp, vn, vdn]; col += 4
        if np.isfinite(zp) and np.isfinite(zn) and dp + dn > 0:
            out[row, col] = (zp * dn + zn * dp) / (dp + dn)
        elif np.isfinite(zp):
            out[row, col] = zp
        elif np.isfinite(zn):
            out[row, col] = zn
        col += 1
        if np.isfinite(vp) and np.isfinite(vn) and vdp + vdn > 0:
            out[row, col] = (vp * vdn + vn * vdp) / (vdp + vdn)
        elif np.isfinite(vp):
            out[row, col] = vp
        elif np.isfinite(vn):
            out[row, col] = vn
        col += 1
        if len(ii):
            goodz = z[ii[np.isfinite(z[ii])]]
            out[row, col:col + 3] = [np.min(goodz), np.max(goodz), np.median(goodz)] if len(goodz) else [np.nan] * 3
            col += 3
            out[row, col] = float(np.mean(np.maximum(0.0, -1.0 - goodz))) if len(goodz) else np.nan; col += 1
            out[row, col] = float(np.mean(np.maximum(0.0, goodz - 1.0))) if len(goodz) else np.nan; col += 1
            out[row, col] = float(np.mean(goodz < -1.0)) if len(goodz) else np.nan; col += 1
            out[row, col] = float(np.mean(goodz > 1.0)) if len(goodz) else np.nan; col += 1
        else:
            col += 7

        day = int(ordinal[i])
        jj = date_groups.get(day, np.empty(0, dtype=int))
        jj = jj[ids[jj] != ids[i]] if len(jj) else jj
        if len(jj):
            zz = z[jj]; dd = dev[jj]
            good = np.isfinite(zz)
            if good.any():
                qz = zz[good]; qd = dd[good]
                # Eight peer slots: the last two are exact-date median and a
                # short-window median respectively.  Fill the latter below.
                out[row, col:col + 7] = [float(np.median(qz)), float(np.mean(np.clip(qz, -6, 6))),
                                           float(np.nanmedian(qd)), float(len(qz)),
                                           float(np.mean(qz < -1)), float(np.mean(qz > 1)),
                                           float(np.median(qz))]
            col += 8
        else:
            col += 8
        # The final peer_zmed_w30 slot is filled from a small date window.  It
        # is intentionally based on peer medians, not query labels.
        near_vals: list[tuple[float, float]] = []
        for ddays in range(1, 31):
            for sign in (-1, 1):
                qday = day + sign * ddays
                kk = date_groups.get(qday, np.empty(0, dtype=int))
                kk = kk[ids[kk] != ids[i]] if len(kk) else kk
                vals = z[kk] if len(kk) else np.empty(0)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    near_vals.append((float(ddays), float(np.median(vals))))
        if near_vals:
            da = np.asarray([x[0] for x in near_vals]); va = np.asarray([x[1] for x in near_vals])
            out[row, col - 1] = _weighted(va, da, 7.0, (-6, 6))

        cj = crop_groups.get((crop[i], day), np.empty(0, dtype=int))
        cj = cj[ids[cj] != ids[i]] if len(cj) else cj
        if len(cj):
            zz = z[cj]; dd = dev[cj]; good = np.isfinite(zz)
            if good.any():
                out[row, col:col + 3] = [float(np.median(zz[good])), float(good.sum()), float(np.nanmedian(dd[good]))]
        col += 3

        # Local climatology interpolation, again using visible rows only.
        ci = ii[np.isfinite(cm[ii])] if len(ii) else np.empty(0, dtype=int)
        if len(ci):
            cd = np.abs(ordinal[ci] - ordinal[i])
            take = cd <= 180
            if take.any():
                out[row, col] = _weighted(cm[ci[take]], cd[take], 45.0)
                out[row, col + 1] = _weighted(cs[ci[take]], cd[take], 45.0)
                out[row, col + 2] = float(take.sum())
        if not np.isfinite(out[row, col]):
            out[row, col] = cm_by_doy.get(int(doys[i]), np.nan)
        if not np.isfinite(out[row, col + 1]):
            out[row, col + 1] = cs_by_doy.get(int(doys[i]), np.nan)
        p0 = np.nan  # filled by caller after baseline merge
        col += 4

    # Store the names for caller via an attribute-like side channel is awkward;
    # the module-level constant is generated once below.
    return out


FEATURE_NAMES: list[str] = []
for _tau in (7, 14, 30, 60, 120):
    FEATURE_NAMES += [f"z_w{_tau}", f"dev_w{_tau}", f"z_low{_tau}", f"z_high{_tau}", f"n{_tau}"]
FEATURE_NAMES += ["z_prev", "d_prev", "z_next", "d_next", "dev_prev", "dev_dprev", "dev_next", "dev_dnext",
                  "z_interp", "dev_interp", "z_min", "z_max", "z_med", "z_neg", "z_pos", "z_low_frac", "z_high_frac",
                  "peer_zmed", "peer_zmean", "peer_devmed", "peer_n", "peer_zlow", "peer_zhigh", "peer_zmed_w7", "peer_zmed_w30",
                  "crop_zmed", "crop_n", "crop_devmed", "cm_local", "cs_local", "cm_n", "p0_minus_cm"]


def _attach_derived_features(f: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Add nonlinear tail gates and baseline-relative seasonal features."""
    z = pd.DataFrame(f, columns=FEATURE_NAMES)
    # Correct the duplicated peer-window naming: peer_zmed_w7 and _w30 are
    # both represented by the two final peer columns in the raw builder.
    z["p0_minus_cm"] = baseline - z["cm_local"]
    for c in ["z_w7", "z_w14", "z_w30", "z_w60", "z_interp", "z_med", "peer_zmed", "peer_zmean", "peer_zmed_w7", "peer_zmed_w30"]:
        x = pd.to_numeric(z[c], errors="coerce").to_numpy(float)
        z[c + "_neg"] = np.minimum(x + 1.0, 0.0)
        z[c + "_pos"] = np.maximum(x - 1.0, 0.0)
        z[c + "_crit"] = (x < -2.0).astype(float)
    # Interaction: common state is more reliable when local and peer states
    # agree.  Missing values remain missing and are imputed by the estimator.
    z["local_peer_agree"] = z["z_w30"] * z["peer_zmed"]
    z["local_peer_mean"] = z[["z_w30", "peer_zmed"]].mean(axis=1)
    z["local_conf"] = np.minimum(z["n30"].fillna(0.0) / 8.0, 1.0) * np.minimum(z["peer_n"].fillna(0.0) / 8.0, 1.0)
    return z.to_numpy(float), list(z.columns)


def _prediction_map(preds: pd.DataFrame, dataset: str, partition: str) -> pd.DataFrame:
    z = preds[(preds["dataset"] == dataset) & (preds["partition"] == partition) & (preds["method"] == BASE_METHOD)].copy()
    z["date"] = pd.to_datetime(z["date"])
    return z[KEY + ["_truth", "pred"]].rename(columns={"pred": "baseline"})


def _make_part(frame: pd.DataFrame, mask: np.ndarray, dataset: str,
               partition: str, pmap: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    mask = np.asarray(mask, bool)
    if "_truth" not in d:
        d["_truth"] = _finite(d["primary_ndvi"])
    known = (~mask) & np.isfinite(_finite(d["primary_ndvi"]))
    y = _finite(d["primary_ndvi"])
    cm = _finite(d.get("ndvi_climatology_mean", pd.Series(np.nan, index=d.index)))
    cs = _finite(d.get("ndvi_climatology_std", pd.Series(np.nan, index=d.index)))
    z = _safe_z(y, cm, cs); dev = y - cm
    z[~known] = np.nan; dev[~known] = np.nan
    qi = np.flatnonzero(mask)
    raw = _aggregate_groups(d, known, z, dev, qi)
    # Keep the fold truth under an explicit name before joining the baseline;
    # the prediction table also carries a defensive ``_truth`` column.
    q = d.loc[qi, KEY].copy().reset_index(drop=True)
    q["truth"] = _finite(d.loc[qi, "_truth"])
    q["date"] = pd.to_datetime(q["date"])
    q = q.merge(pmap[KEY + ["baseline"]], on=KEY, how="left", validate="one_to_one")
    if q["baseline"].isna().any():
        raise ValueError(f"baseline missing for {dataset}/{partition}")
    # Baseline-relative feature is computed after merge; all other columns are
    # visibility-safe and were generated before querying truth.
    raw[:, FEATURE_NAMES.index("p0_minus_cm")] = q["baseline"].to_numpy(float) - raw[:, FEATURE_NAMES.index("cm_local")]
    aug, names = _attach_derived_features(raw, q["baseline"].to_numpy(float))
    for j, name in enumerate(names):
        q[name] = aug[:, j]
    q["dataset"] = dataset; q["partition"] = partition
    # Diagnostic only: true z/status are never passed to a correction model.
    q["true_z"] = np.nan
    q["true_status"] = ""
    return q


def _fit_model(train: pd.DataFrame, features: list[str], kind: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, HuberRegressor
    from sklearn.pipeline import make_pipeline
    X = train[features].to_numpy(float)
    y = train["truth"].to_numpy(float) - train["baseline"].to_numpy(float)
    if kind.startswith("ridge"):
        alpha = float(kind.replace("ridge", ""))
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), Ridge(alpha=alpha)).fit(X, y)
    if kind == "huber":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), HuberRegressor(epsilon=1.5, alpha=0.001, max_iter=300)).fit(X, y)
    if kind.startswith("hgb"):
        leaf = int(kind.replace("hgb", ""))
        return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.035, max_leaf_nodes=leaf,
                                             min_samples_leaf=45, l2_regularization=10.0,
                                             loss="squared_error", random_state=42).fit(X, y)
    raise ValueError(kind)


def _predict(model, q: pd.DataFrame, features: list[str], shrink: float = 1.0) -> np.ndarray:
    corr = np.asarray(model.predict(q[features].to_numpy(float)), float)
    corr = np.clip(corr, -0.16, 0.16) * float(shrink)
    return np.clip(q["baseline"].to_numpy(float) + corr, CLIP_LO, CLIP_HI)


def _metric(q: pd.DataFrame, p: np.ndarray) -> tuple[float, float]:
    e = np.asarray(p, float) - q["truth"].to_numpy(float)
    return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))


def _fill_diag(parts: list[pd.DataFrame], source: pd.DataFrame, original: pd.DataFrame) -> None:
    """Attach true z/status for reporting only, keyed by AOI/date."""
    if original is None or original.empty:
        return
    oo = original.copy(); oo["date"] = pd.to_datetime(oo["date"])
    cols = [c for c in ["anon_polygon_id", "date", "ndvi_zscore", "status"] if c in oo.columns]
    if len(cols) < 2:
        return
    mm = oo[cols].drop_duplicates(KEY).rename(columns={"ndvi_zscore": "true_z", "status": "true_status"})
    for p in parts:
        z = p.merge(mm, on=KEY, how="left", suffixes=("", "_orig"))
        if "true_z_orig" in z:
            p["true_z"] = z["true_z_orig"]
        if "true_status_orig" in z:
            p["true_status"] = z["true_status_orig"]


def _evaluate(protocol: str, parts: list[pd.DataFrame], train_parts: list[pd.DataFrame],
              rows: list[dict], pred_rows: list[pd.DataFrame], tag: str) -> None:
    # Keep only finite baseline/truth rows for fitting; all correction features
    # are imputed inside each estimator.
    feature_sets = {
        "ridge10": FEATURE_NAMES,
        "ridge30": FEATURE_NAMES,
        "ridge100": FEATURE_NAMES,
        "huber": FEATURE_NAMES,
        "hgb8": FEATURE_NAMES,
        "hgb16": FEATURE_NAMES,
    }
    for ti, test in enumerate(parts):
        fitset = [p for j, p in enumerate(train_parts) if not (train_parts is parts and j == ti)]
        # For same-protocol leave-one-partition-out, train_parts is parts.  For
        # the cross-domain random evaluation it is a separate exact list.
        if not fitset:
            continue
        fit = pd.concat(fitset, ignore_index=True)
        fit = fit[np.isfinite(fit["truth"]) & np.isfinite(fit["baseline"])].copy()
        base = test["baseline"].to_numpy(float)
        rb, mb = _metric(test, base)
        rows.append({"protocol": protocol, "partition": str(test["partition"].iat[0]), "candidate": "baseline",
                     "n": len(test), "rmse": rb, "mae": mb, "tag": tag,
                     "tail_n": int((np.abs(test["truth"] - test["baseline"]) > 0.12).sum())})
        for cand, fs in feature_sets.items():
            model = _fit_model(fit, fs, cand)
            for shrink in ([0.25, 0.5, 0.75, 1.0] if cand in {"ridge10", "ridge30", "hgb8", "hgb16"} else [1.0]):
                name = cand if shrink == 1.0 else f"{cand}_s{shrink:.2f}"
                pp = _predict(model, test, fs, shrink)
                rm, ma = _metric(test, pp)
                rows.append({"protocol": protocol, "partition": str(test["partition"].iat[0]), "candidate": name,
                             "n": len(test), "rmse": rm, "mae": ma, "tag": tag,
                             "tail_n": int((np.abs(test["truth"] - test["baseline"]) > 0.12).sum())})
                if shrink in (0.5, 1.0) and (cand in {"ridge30", "hgb8", "hgb16"}):
                    zz = test[KEY + ["truth", "baseline", "true_z", "true_status"]].copy()
                    zz["pred"] = pp; zz["candidate"] = name; zz["protocol"] = protocol; zz["partition"] = str(test["partition"].iat[0]); zz["tag"] = tag
                    pred_rows.append(zz)


def main() -> None:
    train = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    private = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    preds = pd.read_csv(RESEARCH / "teammate_sweep_postcorr_preds.csv", parse_dates=["date"], low_memory=False)
    exact: list[pd.DataFrame] = []
    for year in range(2019, 2025):
        fold, _ = make_fold(train.copy(), private.copy(), year)
        m = fold["is_synthetic_gap"].fillna(False).to_numpy(bool)
        part = _make_part(fold, m, "exact_hidden_doy", f"exact{year}", _prediction_map(preds, "exact_hidden_doy", f"exact{year}"))
        _fill_diag([part], fold, train)
        exact.append(part)
    random_parts: list[pd.DataFrame] = []
    for seed in (0, 1, 2):
        frame, m = _mask_private(private.copy(), seed)
        part = _make_part(frame, m, "random_private_like", f"random{seed}", _prediction_map(preds, "random_private_like", f"random{seed}"))
        _fill_diag([part], private, private)
        random_parts.append(part)

    rows: list[dict] = []; pred_rows: list[pd.DataFrame] = []
    # Exact years are the strict cross-fit protocol.
    _evaluate("exact_hidden_doy", exact, exact, rows, pred_rows, "loo_year")
    # Random seeds are evaluated both with seed-LOO (diagnostic) and with the
    # strict train-on-exact calibration that has no private-row label overlap.
    _evaluate("random_private_like", random_parts, random_parts, rows, pred_rows, "loo_seed")
    _evaluate("random_private_like", random_parts, exact, rows, pred_rows, "fit_exact")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESEARCH / "tail_anomaly_v2_metrics.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(RESEARCH / "tail_anomaly_v2_predictions.csv", index=False)
    ag = []
    for (protocol, tag, cand), g in metrics.groupby(["protocol", "tag", "candidate"], sort=False):
        ag.append({"protocol": protocol, "tag": tag, "candidate": cand, "n": int(g.n.sum()),
                   "rmse_pooled": float(np.sqrt(np.average(g.rmse ** 2, weights=g.n))),
                   "mae_pooled": float(np.average(g.mae, weights=g.n)), "partitions": len(g)})
    agg = pd.DataFrame(ag).sort_values(["protocol", "tag", "rmse_pooled"])
    agg.to_csv(RESEARCH / "tail_anomaly_v2_aggregate.csv", index=False)

    lines = ["# Tail/anomaly v2 residual correction", "", "Research-only; production unchanged.", "",
             "Features use visible primary/climatology residuals around each masked row and same-date peers; status/hidden labels are diagnostics only.", "",
             "## Pooled cross-fitted results", "", agg.head(80).to_string(index=False), ""]
    for protocol in ["exact_hidden_doy", "random_private_like"]:
        sub = agg[agg.protocol == protocol]
        lines += [f"## {protocol}", "", sub.head(20).to_string(index=False), ""]
    lines += ["Decision: retain only a correction that improves the pooled exact and strict fit_exact random protocols; no outputs/model_dani_tuned* files are modified."]
    (RESEARCH / "tail_anomaly_v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(agg.head(80).to_string(index=False))


if __name__ == "__main__":
    main()

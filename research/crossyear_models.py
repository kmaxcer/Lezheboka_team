"""Fast experiments for using same-AOI history across calendar years.

The real private file has two regimes: some AOIs expose only 2025 while the
training file contains their 2010--2024 history.  ``src.infer`` currently
uses only same-AOI/same-year neighbours, so this script measures whether an
aligned seasonal template can improve that regime without using a masked
row.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026/09/04/ml/work/cosmo_latest_20260904")
# Correct path is kept configurable at runtime; this fallback is useful when
# the script is copied outside the desktop workspace.
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
sys.path.insert(0, str(ROOT / "src"))
from infer import predict_private, _prepare, _fit_source_maps, _source_labels  # noqa: E402


DYNAMIC = [
    "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi",
    "landsat_ndwi", "modis_ndvi", "modis_evi", "era5_temp_c",
    "era5_precip_mm", "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std", "n_reference_years",
    "status",
]


def _mask_2025(pr: pd.DataFrame, seed: int = 42, frac: float = .18):
    d = pr[pr.date.dt.year == 2025].copy().reset_index(drop=True)
    d["_truth"] = d.primary_ndvi.astype(float)
    d["is_synthetic_gap"] = False
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(d), dtype=bool)
    for _, g in d[d.primary_ndvi.notna()].groupby("anon_polygon_id"):
        ix = g.index.to_numpy()
        n = max(1, int(round(frac * len(ix))))
        mask[rng.choice(ix, n, replace=False)] = True
    for c in DYNAMIC:
        if c in d:
            d.loc[mask, c] = np.nan
    d.loc[mask, "is_synthetic_gap"] = True
    return d, mask


def _source_maps(frame: pd.DataFrame):
    z = _prepare(frame)
    known = z._obs.to_numpy(bool)
    return z, _fit_source_maps(z, known, bin_days=30)


def _robust_affine(x: np.ndarray, y: np.ndarray):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 8 or np.ptp(x) < 1e-7:
        return float(np.nanmedian(y - x)) if len(y) else 0.0, 1.0
    # Trim only obvious outliers; the data contain occasional corrupted
    # auxiliary values but the target itself is mostly in [0, 1].
    lo, hi = np.quantile(x, [.03, .97]); keep = (x >= lo) & (x <= hi)
    lo, hi = np.quantile(y, [.03, .97]); keep &= (y >= lo) & (y <= hi)
    if keep.sum() < 8:
        keep = np.ones(len(x), bool)
    b, a = np.polyfit(x[keep], y[keep], 1)
    if not np.isfinite(a + b) or abs(b) > 3:
        return 0.0, 1.0
    return float(a), float(b)


def _template_predict(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    hidden: np.ndarray,
    *,
    radius: float = 18.0,
    k: int = 24,
    blend: float = .55,
    adapt_radius: float = 45.0,
):
    """Predict hidden rows with an aligned cross-year seasonal template.

    For each AOI and query day, historical observations are selected by DOY
    distance.  A current-year affine correction is estimated from the
    unmasked 2025 points of that AOI, which transfers level/scale while
    avoiding the query itself.  The result is blended with the same-year
    local interpolation from the production estimator.
    """
    # Work on a clean combined frame.  Train rows are never synthetic.
    d = frame.copy().reset_index(drop=True)
    d["is_synthetic_gap"] = d["is_synthetic_gap"].fillna(False).astype(bool)
    h = train.copy()
    h["is_synthetic_gap"] = False
    cols = [c for c in d.columns if c in h.columns or c == "is_synthetic_gap"]
    # Keep only train AOIs that appear in the query frame, avoiding useless
    # groups and making the experiment fast.
    h = h[h.anon_polygon_id.isin(set(d.anon_polygon_id))]
    allf = pd.concat([h[cols], d[cols]], ignore_index=True, sort=False)
    z = _prepare(allf)
    # Synthetic mask is present only on d; map rows back by a private key.
    qmask = z.is_synthetic_gap.to_numpy(bool)
    y = z.primary_ndvi.to_numpy(float)
    known = np.isfinite(y) & ~qmask
    doy = z._doy.to_numpy(int)
    year = z._year.to_numpy(int)
    ids = z.anon_polygon_id.to_numpy(object)
    src = z._src.to_numpy(object)
    maps = _fit_source_maps(z, known, bin_days=30)

    # Production estimate on the private-only frame.  It is a useful local
    # component whenever 2025 has left/right anchors.
    prod = predict_private(d, train=train, k=8, bin_days=30)
    key = d.loc[qmask[-len(d):], ["anon_polygon_id", "date"]].copy()
    key["date"] = pd.to_datetime(key.date).dt.strftime("%Y-%m-%d")
    pp = key.merge(prod, on=["anon_polygon_id", "date"], how="left").primary_ndvi_pred.to_numpy(float)

    # Index of private rows in combined frame (they are the final len(d)).
    final_idx = np.arange(len(z) - len(d), len(z))
    # Output arrays are indexed by hidden private rows only (same order as
    # ``d.loc[d.is_synthetic_gap]``), not by every row in the combined frame.
    hidden_combined = final_idx[qmask[final_idx]]
    hidden_pos = {int(ix): j for j, ix in enumerate(hidden_combined)}
    out = np.full(len(hidden_combined), np.nan)
    # Historical and current observations by AOI.
    for pid in pd.unique(ids[final_idx]):
        all_ix = np.flatnonzero(ids == pid)
        q_ix = final_idx[(ids[final_idx] == pid) & qmask[final_idx]]
        if not len(q_ix):
            continue
        # Current 2025 known values used to estimate an AOI/year correction.
        cur = all_ix[(year[all_ix] == 2025) & known[all_ix]]
        hist = all_ix[(year[all_ix] != 2025) & known[all_ix]]
        if not len(hist):
            hist = all_ix[known[all_ix]]
        # Build a set of source-converted target curves.  We evaluate a
        # target-source mixture only approximately; primary target itself is
        # the safest domain for cross-year transfer.
        for qi, q in enumerate(q_ix):
            if not len(hist):
                continue
            dd = np.abs(doy[hist] - doy[q]).astype(float)
            # Growing-season data do not wrap across New Year; no circular
            # distance here.  Use a compact neighbourhood and weighted cubic
            # fit to retain peak curvature.
            order = np.argsort(dd)
            sel = order[dd[order] <= radius]
            if len(sel) < 4:
                sel = order[:min(k, len(order))]
            else:
                sel = sel[:min(k, len(sel))]
            hi = hist[sel]
            w = 1.0 / (1.0 + dd[sel] / max(1.0, radius))
            xx = (doy[hi] - doy[q]) / max(1.0, radius)
            yy = y[hi]
            deg = min(2, len(yy) - 1)
            try:
                base = float(np.polynomial.polynomial.polyfit(xx, yy, deg, w=w)[0])
            except Exception:
                base = float(np.average(yy, weights=w))
            # Robust current-year correction at nearby DOYs.  Estimate both
            # offset and scale when enough anchors exist; otherwise offset.
            adj = base
            if len(cur) >= 4:
                dc = np.abs(doy[cur] - doy[q]).astype(float)
                cs = cur[np.argsort(dc)[:min(40, len(cur))]]
                # Historical template at each current anchor, excluding the
                # anchor's own year, to obtain comparable pairs.
                bx, by = [], []
                for cj in cs:
                    hd = np.abs(doy[hist] - doy[cj])
                    so = np.argsort(hd)[:min(k, len(hd))]
                    if len(so):
                        ww = 1.0 / (1.0 + hd[so] / max(1.0, radius))
                        xx2 = (doy[hist[so]] - doy[cj]) / max(1.0, radius)
                        try:
                            tv = float(np.polynomial.polynomial.polyfit(xx2, y[hist[so]], min(2, len(so)-1), w=ww)[0])
                        except Exception:
                            tv = float(np.average(y[hist[so]], weights=ww))
                        if np.isfinite(tv) and np.isfinite(y[cj]):
                            bx.append(tv); by.append(y[cj])
                if len(bx) >= 4:
                    a, b = _robust_affine(np.asarray(bx), np.asarray(by))
                    # Shrink affine correction heavily when anchors are
                    # sparse; this avoids copying a single noisy observation.
                    shrink = min(1.0, len(bx) / 12.0)
                    adj = (1-shrink) * base + shrink * (a + b * base)
            # Blend with local current-year prediction.  q_ix order follows
            # the private frame's hidden rows; map by ordinal within qmask.
            # ``q_pos`` is robust even if groups are not contiguous.
            q_pos = hidden_pos.get(int(q), -1)
            local = pp[q_pos] if 0 <= q_pos < len(pp) else np.nan
            if np.isfinite(local):
                # More trust in local anchors when a query is interior; more
                # trust in template at edges or very sparse current year.
                ncur = len(cur)
                lw = min(.75, max(.20, ncur / 90.0))
                adj = lw * local + (1-lw) * adj
            if q_pos >= 0:
                out[q_pos] = np.clip(adj, -0.5, 1.2)
    # Fill any misses with production values.
    out[~np.isfinite(out)] = pp[~np.isfinite(out)]
    return out, pp


def main():
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    # Regime labels from the complete private file.
    fullhist = pr.groupby("anon_polygon_id").size().to_dict()
    rows = []
    for seed in [0, 1, 2]:
        d, mask = _mask_2025(pr, seed)
        q = d.loc[mask].copy(); truth = q._truth.to_numpy(float)
        pred, prod = _template_predict(d, tr, mask)
        ids = q.anon_polygon_id.to_numpy(object)
        labels = np.array([fullhist.get(x, 0) > 300 for x in ids])
        for label, m in [("all", np.ones(len(q), bool)), ("train_history", ~labels), ("private_history", labels)]:
            for name, p in [("prod", prod), ("template", pred)]:
                e = p[m] - truth[m]
                rows.append((seed, label, name, int(m.sum()), float(np.sqrt(np.mean(e*e))), float(np.mean(np.abs(e)))))
    out = pd.DataFrame(rows, columns=["seed", "regime", "method", "n", "rmse", "mae"])
    print(out.groupby(["regime", "method"]).apply(lambda z: pd.Series({"n": z.n.sum(), "rmse": np.sqrt(np.average(z.rmse**2, weights=z.n)), "mae": np.average(z.mae, weights=z.n)})).reset_index().to_string(index=False))
    out.to_csv(ROOT / "research" / "crossyear_models_results.csv", index=False)


if __name__ == "__main__":
    main()

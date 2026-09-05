"""Cross-sectional latent-factor tail correction (research only).

For each masked fold, this experiment estimates a date-level vegetation shock
from visible AOI residuals and fits each AOI's loading on that shock.  The
query prediction is blended conservatively with the fixed HGB+lag baseline.
No query target, status, or hidden row is used in the factor construction.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402
sys.path.insert(0, str(RESEARCH))
from teammate_sweep_postcorr import _mask_private  # noqa: E402


def _source(df: pd.DataFrame) -> np.ndarray:
    return np.select([df.s2_ndvi.notna(), df.landsat_ndvi.notna(), df.modis_ndvi.notna()], ["s2", "ls", "md"], "none")


def _interp_group(x: np.ndarray, v: np.ndarray, q: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(v)
    if ok.sum() == 0: return np.full(len(q), np.nan)
    if ok.sum() == 1: return np.full(len(q), v[ok][0])
    o = np.argsort(x[ok]); return np.interp(q, x[ok][o], v[ok][o])


def factor_predict(frame: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return several factor-model predictions for masked rows."""
    d = frame.copy().reset_index(drop=True); d["date"] = pd.to_datetime(d["date"])
    ids = d.anon_polygon_id.astype(str).to_numpy(); years = d.date.dt.year.to_numpy(int)
    day = d.date.map(pd.Timestamp.toordinal).to_numpy(float); doy = d.date.dt.dayofyear.to_numpy(int)
    y = pd.to_numeric(d.primary_ndvi, errors="coerce").to_numpy(float)
    cm = pd.to_numeric(d.get("ndvi_climatology_mean", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    cs = pd.to_numeric(d.get("ndvi_climatology_std", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    known = (~mask) & np.isfinite(y); qidx = np.flatnonzero(mask)
    # Reconstruct a usable climatology at each query from visible same-AOI
    # rows; this is independent of target labels at hidden rows.
    cmq = np.full(len(qidx), np.nan); csq = np.full(len(qidx), np.nan)
    groups = d.groupby([ids, years], sort=False).groups
    for _, ix0 in groups.items():
        ix = np.asarray(ix0, int); kk = ix[known[ix]]; qq = ix[mask[ix]]
        if len(qq) == 0: continue
        cmq[np.searchsorted(qidx, qq)] = _interp_group(day[kk], cm[kk], day[qq]) if len(kk) else np.nan
        csq[np.searchsorted(qidx, qq)] = _interp_group(day[kk], cs[kk], day[qq]) if len(kk) else np.nan
    # Global seasonal fallback for sparse AOIs.
    for j, qi in enumerate(qidx):
        if not np.isfinite(cmq[j]):
            z = known & (doy == doy[qi]) & np.isfinite(cm)
            cmq[j] = float(np.nanmedian(cm[z])) if z.any() else np.nan
        if not np.isfinite(csq[j]):
            z = known & (doy == doy[qi]) & np.isfinite(cs)
            csq[j] = float(np.nanmedian(cs[z])) if z.any() else 0.14
    # Visible residuals.  Clip extreme sensor artefacts before estimating a
    # cross-sectional factor so one AOI cannot move all predictions.
    res = y - cm; res[~known] = np.nan; res = np.clip(res, -0.45, 0.45)
    zres = res / np.maximum(cs, 0.025); zres[~np.isfinite(zres)] = np.nan; zres = np.clip(zres, -5, 5)
    date_groups = {pd.Timestamp(k).toordinal(): np.asarray(v, int) for k, v in d.loc[known].groupby("date", sort=False).groups.items()}
    f_abs: dict[int, float] = {}; f_z: dict[int, float] = {}; n_date: dict[int, int] = {}
    for k, ii in date_groups.items():
        va = res[ii]; vz = zres[ii]
        va = va[np.isfinite(va)]; vz = vz[np.isfinite(vz)]
        if len(va) >= 3:
            f_abs[k] = float(np.median(va)); f_z[k] = float(np.median(vz)) if len(vz) else np.nan; n_date[k] = len(va)
    # Add a smoothed factor for dates with no exact peer observations.
    fd = np.array(sorted(f_abs), float); fa = np.array([f_abs[k] for k in fd]); fz = np.array([f_z[k] for k in fd])
    out: dict[str, np.ndarray] = {k: np.full(len(qidx), np.nan) for k in ["abs", "z", "robust", "peer"]}
    if len(fd):
        for j, qi in enumerate(qidx):
            k = day[qi]
            # Exact-date factor and local +/-7-day weighted factor.
            exact = f_abs.get(int(k), np.nan); exactz = f_z.get(int(k), np.nan)
            dist = np.abs(fd - k); take = dist <= 14
            if take.any():
                w = np.exp(-dist[take] / 5.0); sm = float(np.dot(fa[take], w) / w.sum()); smz = float(np.dot(fz[take], w) / w.sum())
            else: sm = smz = np.nan
            out["abs"][j] = exact if np.isfinite(exact) else sm
            out["z"][j] = exactz if np.isfinite(exactz) else smz
            out["robust"][j] = 0.7 * exact + 0.3 * sm if np.isfinite(exact) and np.isfinite(sm) else out["abs"][j]
    # Fit loadings/intercepts for each AOI/year using its visible residuals and
    # the factor on matching dates.  A ridge-like denominator prevents noisy
    # two-point loadings.  Also fit a cross-year AOI loading as fallback.
    # Build factor lookup for every known date.
    farr = np.array([f_abs.get(int(k), np.nan) for k in day]); fzarr = np.array([f_z.get(int(k), np.nan) for k in day])
    qpos = {int(q): j for j, q in enumerate(qidx)}
    for key, ix0 in groups.items():
        ix = np.asarray(ix0, int); kk = ix[known[ix]]; qq = ix[mask[ix]]
        if len(qq) == 0: continue
        # Use exact date factor; if absent, smoothed query factor below.
        good = kk[np.isfinite(res[kk]) & np.isfinite(farr[kk])]
        if len(good) >= 8:
            x = farr[good]; yy = res[good]; xm = float(np.median(x)); ym = float(np.median(yy))
            den = float(np.sum((x - xm) ** 2) + 0.03)
            slope = float(np.clip(np.sum((x - xm) * (yy - ym)) / den, -2.0, 2.0)); intercept = ym - slope * xm
            gz = kk[np.isfinite(zres[kk]) & np.isfinite(fzarr[kk])]
            if len(gz) >= 8:
                xz = fzarr[gz]; yz = zres[gz]; xzm = float(np.median(xz)); yzm = float(np.median(yz)); denz = float(np.sum((xz-xzm)**2)+0.1)
                slopez = float(np.clip(np.sum((xz-xzm)*(yz-yzm))/denz, -2, 2)); intz = yzm-slopez*xzm
            else: slopez, intz = 0.0, 0.0
            for qi in qq:
                j = qpos[int(qi)]; ff = out["abs"][j]; zz = out["z"][j]
                if np.isfinite(ff): out["peer"][j] = cmq[j] + intercept + slope * ff
                if np.isfinite(zz) and np.isfinite(csq[j]):
                    # A normalized loading estimate is converted to NDVI.
                    out["z"][j] = cmq[j] + csq[j] * (intz + slopez * zz)
        # Fallback direct common-factor shift, no AOI loading.
        for qi in qq:
            j = qpos[int(qi)]
            if not np.isfinite(out["peer"][j]) and np.isfinite(out["abs"][j]) and np.isfinite(cmq[j]):
                out["peer"][j] = cmq[j] + out["abs"][j]
    return np.column_stack([out[k] for k in ["abs", "z", "robust", "peer"]]), out


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    pp = pd.read_csv(RESEARCH / "teammate_sweep_postcorr_preds.csv", parse_dates=["date"], low_memory=False)
    records: list[dict] = []
    for typ in ["exact", "random"]:
        years = list(range(2019, 2025)) if typ == "exact" else [0, 1, 2]
        for val in years:
            if typ == "exact":
                d, _ = make_fold(tr.copy(), pr.copy(), val); m = d.is_synthetic_gap.fillna(False).to_numpy(bool); ds = "exact_hidden_doy"; part = f"exact{val}"
            else:
                d, m = _mask_private(pr.copy(), val); ds = "random_private_like"; part = f"random{val}"
            base = pp[(pp.dataset == ds) & (pp.partition == part) & (pp.method == "blend_lag_0.20")][["anon_polygon_id", "date", "pred"]].rename(columns={"pred": "base"})
            q = d.loc[m, ["anon_polygon_id", "date", "_truth"]].copy(); q["date"] = pd.to_datetime(q.date); q = q.merge(base, on=["anon_polygon_id", "date"], validate="one_to_one")
            F, _ = factor_predict(d, m)
            y = q._truth.to_numpy(float); b = q.base.to_numpy(float)
            # cm/cs are not returned separately; direct factor predictions are
            # already in NDVI units for abs/robust/peer, while z is converted.
            for j, name in enumerate(["factor_abs", "factor_z", "factor_robust", "factor_peer"]):
                x = F[:, j]; ok = np.isfinite(x)
                for w in np.arange(0, 0.51, 0.025):
                    p = b.copy(); p[ok] = (1-w)*b[ok] + w*x[ok]; e = p-y
                    records.append({"protocol": ds, "partition": part, "candidate": name + f"_w{w:.3f}", "n": len(y), "rmse": float(np.sqrt(np.mean(e*e))), "mae": float(np.mean(np.abs(e)))})
            print(typ, val, flush=True)
    m = pd.DataFrame(records); m.to_csv(RESEARCH / "tail_anomaly_v2_factor_metrics.csv", index=False)
    a=[]
    for (p,c),g in m.groupby(["protocol","candidate"],sort=False): a.append({"protocol":p,"candidate":c,"n":int(g.n.sum()),"rmse_pooled":float(np.sqrt(np.average(g.rmse**2,weights=g.n))),"mae_pooled":float(np.average(g.mae,weights=g.n)),"parts":len(g)})
    a=pd.DataFrame(a).sort_values(["protocol","rmse_pooled"]);a.to_csv(RESEARCH/"tail_anomaly_v2_factor_aggregate.csv",index=False);print(a.head(80).to_string(index=False));(RESEARCH/"tail_anomaly_v2_factor_report.md").write_text("# Tail anomaly v2 factor\n\n"+a.head(100).to_string(index=False),encoding="utf-8")

if __name__ == "__main__": main()

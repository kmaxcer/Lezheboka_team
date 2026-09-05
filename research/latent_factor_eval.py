"""Fast leakage-safe AOI x date latent-factor experiments.

Research only.  The target is observed on a sparse, shared acquisition
schedule.  This module evaluates cross-sectional multi-peer ridge and
seasonal low-rank completion on the same private-like holdout used by the
main pipeline.  It never reads dynamic fields from masked rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
R = ROOT / "research"
ID, DATE, TARGET, GAP = "anon_polygon_id", "date", "primary_ndvi", "is_synthetic_gap"


def holdout_mask(pr: pd.DataFrame, seed: int = 70404) -> np.ndarray:
    """The fixed 15% visible-private holdout used by the other audits."""
    known = pr[TARGET].notna().to_numpy(bool) & ~pr[GAP].fillna(False).to_numpy(bool)
    out = np.zeros(len(pr), bool)
    rng = np.random.default_rng(seed)
    years = pd.to_datetime(pr[DATE]).dt.year
    for _, ix0 in pr.loc[known].groupby([ID, years], sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int)
        n = max(1, int(round(.15 * len(ix))))
        out[rng.choice(ix, size=min(n, len(ix)), replace=False)] = True
    return out


def _panel_arrays(ref: pd.DataFrame, gaps: np.ndarray):
    """Return tidy observed arrays and query metadata."""
    d = ref.copy().reset_index(drop=True)
    d[DATE] = pd.to_datetime(d[DATE])
    d["yearx"] = d[DATE].dt.year.astype(int)
    d["doyx"] = d[DATE].dt.dayofyear.astype(int)
    d["datekey"] = d[DATE].dt.strftime("%Y-%m-%d")
    d["id"] = d[ID].astype(str)
    gaps = np.asarray(gaps, bool)
    y = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
    known = np.isfinite(y) & ~gaps
    return d, y, known


def _seasonal_baseline(d: pd.DataFrame, y: np.ndarray, known: np.ndarray, bw: int = 16,
                       by_year: bool = False) -> np.ndarray:
    """Robust circular seasonal baseline per AOI (optionally AOI-year).

    A narrow bin median is deliberately used instead of interpolation: it
    avoids using query labels while remaining stable for sparse acquisition
    schedules.  Missing bins fall back to the AOI median and global bin.
    """
    n = len(d); doy = d["doyx"].to_numpy(int); ids = d["id"].to_numpy(str)
    yr = d["yearx"].to_numpy(int)
    # bins are centered enough to preserve crop phenology without overfitting
    b = ((doy - 1) // int(bw)).astype(int)
    z = pd.DataFrame({"id": ids, "yr": yr, "b": b, "y": y, "ok": known})
    o = z.loc[known & np.isfinite(y)]
    if by_year:
        tab = o.groupby(["id", "yr", "b"], observed=True).y.median()
        aid = o.groupby(["id", "yr"], observed=True).y.median()
    else:
        tab = o.groupby(["id", "b"], observed=True).y.median()
        aid = o.groupby("id", observed=True).y.median()
    glob = o.groupby("b", observed=True).y.median()
    gmed = float(np.nanmedian(o.y)) if len(o) else .35
    out = np.full(n, np.nan)
    # nearest-bin fallback is useful for sensor-specific schedules; circular
    # distance is measured in bin units.
    for i in range(n):
        key = (ids[i], yr[i], b[i]) if by_year else (ids[i], b[i])
        v = tab.get(key, np.nan)
        if not np.isfinite(v):
            # Search +/- 2 bins, weighted toward the query bin.
            vals = []
            for db in range(1, 3):
                for bb, w in ((b[i] - db, 1 / (db + 1)), (b[i] + db, 1 / (db + 1))):
                    k2 = (ids[i], yr[i], bb) if by_year else (ids[i], bb)
                    q = tab.get(k2, np.nan)
                    if np.isfinite(q): vals.append((q, w))
            if vals: v = float(np.average([q for q, _ in vals], weights=[w for _, w in vals]))
        if not np.isfinite(v):
            v = aid.get((ids[i], yr[i]) if by_year else ids[i], np.nan)
        if not np.isfinite(v): v = glob.get(b[i], gmed)
        out[i] = float(v)
    return out


def _date_factor_prediction(d: pd.DataFrame, y: np.ndarray, known: np.ndarray,
                            qidx: np.ndarray, baseline: np.ndarray,
                            robust: str = "median", shrink: float = .0,
                            source_frame: pd.DataFrame | None = None) -> np.ndarray:
    """Estimate residual date factors and AOI loadings.

    Unlike a raw date median, this centers each AOI by its seasonal level and
    estimates a loading for every AOI.  The optional shrink controls loading
    ridge regularization.  This is a compact one-factor matrix completion.
    """
    ids = d["id"].to_numpy(str); dates = d["datekey"].to_numpy(str)
    resid = y - baseline
    # Date factors from observed AOIs.  Trimmed mean is less noisy than a
    # median when only 2-3 peers are present, but median is safer for tails.
    f_by = {}
    for dt, ix0 in pd.Series(np.arange(len(d))).groupby(dates, sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int); v = resid[ix][known[ix]]; v = v[np.isfinite(v)]
        if not len(v): continue
        if robust == "trim":
            sv = np.sort(v); k = int(.15 * len(sv)); v = sv[k:len(sv)-k] if len(sv) > 2*k else sv
            f_by[dt] = float(np.mean(v))
        elif robust == "mean":
            f_by[dt] = float(np.mean(np.clip(v, -.35, .35)))
        else:
            f_by[dt] = float(np.median(v))
    fac = np.array([f_by.get(dt, np.nan) for dt in dates], float)
    # AOI loading fit only on observed entries; shrink to avoid unstable
    # regressions for sparse/new AOIs.
    bet = {}
    for pid, ix0 in pd.Series(np.arange(len(d))).groupby(ids, sort=False).groups.items():
        ix = np.asarray(ix0, dtype=int); ok = known[ix] & np.isfinite(fac[ix]) & np.isfinite(resid[ix])
        ii = ix[ok]; xx = fac[ii]; rr = resid[ii]
        den = float(np.dot(xx, xx))
        bet[pid] = float(np.clip(np.dot(xx, rr) / (den + shrink), -2, 2)) if len(ii) >= 8 else 0.
    pred = baseline[qidx] + np.array([bet.get(pid, 0.) for pid in ids[qidx]]) * np.nan_to_num(fac[qidx])
    return pred


def _lowrank_completion(d: pd.DataFrame, y: np.ndarray, known: np.ndarray,
                        qidx: np.ndarray, rank: int = 4, iters: int = 20,
                        bw: int = 16, baseline_mode: str = "aoi") -> np.ndarray:
    """Iterative SVD on a date x AOI residual matrix.

    Rows are exact dates (year included), columns AOIs.  The matrix is sparse;
    missing entries are initialized by seasonal baselines and are projected
    back to observed values after each truncated-SVD step.  Predictions are
    clipped to robust residual quantiles before adding the baseline.
    """
    ids = np.array(sorted(d["id"].unique()), dtype=object); idpos = {x:i for i,x in enumerate(ids)}
    dates = np.array(sorted(d["datekey"].unique()), dtype=object); dtpos = {x:i for i,x in enumerate(dates)}
    nrow, ncol = len(dates), len(ids)
    base = _seasonal_baseline(d, y, known, bw=bw, by_year=(baseline_mode == "aoi_year"))
    # Use residuals where known.  Duplicate cells (should not occur) are
    # averaged robustly.
    A = np.full((nrow, ncol), np.nan, float); C = np.zeros_like(A)
    ids0 = d["id"].to_numpy(str); dt0 = d["datekey"].to_numpy(str)
    for i in np.flatnonzero(known):
        r, c = dtpos[dt0[i]], idpos[ids0[i]]; v = y[i] - base[i]
        if np.isfinite(v): A[r, c] = np.nan_to_num(A[r, c], nan=0.) + v; C[r, c] += 1
    obs = C > 0; A[obs] /= C[obs]
    # Initial value: per-date residual median, then zero.
    date_med = np.nanmedian(np.where(obs, A, np.nan), axis=1)
    date_med[~np.isfinite(date_med)] = 0.
    X = np.where(obs, A, date_med[:, None])
    # Center columns to remove static AOI offset before SVD.
    colmed = np.nanmedian(np.where(obs, A, np.nan), axis=0); colmed[~np.isfinite(colmed)] = 0.
    X = X - colmed[None, :]
    # Robust scale limits prevent a few sensor outliers dominating factors.
    finite = A[obs]; lim = np.quantile(np.abs(finite), .995) if len(finite) else .4; lim = max(.12, min(float(lim), .8))
    for _ in range(int(iters)):
        try:
            u, s, vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        rr = min(int(rank), len(s)); Z = (u[:, :rr] * s[:rr]) @ vt[:rr]
        Z = np.clip(Z, -lim, lim)
        X[~obs] = Z[~obs]
        X[obs] = A[obs] - colmed[np.where(obs)[1]]
    # Map query cells back to absolute predictions.  Their row baseline is
    # query-specific and static column residual offset is restored.
    out = np.empty(len(qidx), float)
    for j, i in enumerate(qidx):
        r, c = dtpos[dt0[i]], idpos[ids0[i]]
        out[j] = base[i] + colmed[c] + X[r, c]
    return out


def _multi_peer_ridge(d: pd.DataFrame, y: np.ndarray, known: np.ndarray,
                      qidx: np.ndarray, baseline: np.ndarray, k: int = 8,
                      alpha: float = 10., min_common: int = 30,
                      same_crop: bool = False) -> np.ndarray:
    """Per-AOI multi-peer ridge on seasonal residuals.

    Peer coefficients are learned from dates where both AOIs are observed.
    At inference, available same-date peer residuals are combined.  This
    generalizes the fixed affine peer map while remaining query-safe.
    """
    ids = d["id"].to_numpy(str); dates = d["datekey"].to_numpy(str)
    crop = d.get("crop_type", pd.Series("unknown", index=d.index)).fillna("unknown").astype(str).to_numpy()
    resid = y - baseline
    # date x id residual matrix
    idlist = np.array(sorted(np.unique(ids)), object); ip = {x:i for i,x in enumerate(idlist)}
    dtlist = np.array(sorted(np.unique(dates)), object); dp = {x:i for i,x in enumerate(dtlist)}
    M = np.full((len(dtlist), len(idlist)), np.nan)
    for i in np.flatnonzero(known): M[dp[dates[i]], ip[ids[i]]] = resid[i]
    # Correlations based on pairwise observed residuals.
    with np.errstate(invalid="ignore"):
        C = np.ma.corrcoef(np.ma.masked_invalid(M), rowvar=False).filled(np.nan)
    # fallback to static id order if correlation unavailable
    # Precompute affine maps for every target/peer pair once.  The earlier
    # prototype refit these maps for every query and became needlessly slow
    # on the 2.6k-row audit.
    maps = {}
    for ci in range(len(idlist)):
        corr = C[ci].copy(); corr[ci] = np.nan
        cand = np.flatnonzero(np.isfinite(corr))
        if same_crop:
            cand = np.array([j for j in cand if np.any(crop[ids == idlist[j]] == crop[ids == idlist[ci]])], dtype=int)
        cand = cand[np.argsort(corr[cand])[::-1]] if len(cand) else cand
        for cj in cand[:int(k)]:
            ok = np.isfinite(M[:, ci]) & np.isfinite(M[:, cj])
            if ok.sum() < min_common: continue
            x = M[ok, cj]; z = M[ok, ci]
            xm, zm = np.median(x), np.median(z)
            den = np.sum((x-xm)**2) + alpha
            b = np.clip(np.sum((x-xm)*(z-zm))/den, -2., 2.)
            maps[(ci, cj)] = (float(zm - b*xm), float(b), max(float(corr[cj]), .05))
    out = baseline[qidx].copy()
    for jj, i in enumerate(qidx):
        ci = ip[ids[i]]; dt = dp[dates[i]]
        cand = np.asarray([cj for (ci0, cj) in maps.keys() if ci0 == ci], dtype=int)
        avail = cand[np.isfinite(M[dt, cand])] if len(cand) else cand
        if len(avail) == 0: continue
        # Fit a centered ridge map on common dates for each peer, then combine
        # predictions by correlation-weighted precision.
        vals=[]; ws=[]
        for cj in avail:
            a, b, w = maps[(ci, cj)]
            vals.append(a + b*M[dt,cj]); ws.append(w)
        if vals:
            out[jj] = float(np.average(vals, weights=ws))
    return out


def score_frame(q: pd.DataFrame, pred: np.ndarray, name: str) -> dict:
    y = q["truth"].to_numpy(float); ok = np.isfinite(y) & np.isfinite(pred)
    return {"method": name, "n": int(ok.sum()), "rmse": float(np.sqrt(np.mean((pred[ok]-y[ok])**2))), "mae": float(np.mean(np.abs(pred[ok]-y[ok])))}


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=[DATE], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=[DATE], low_memory=False)
    tr[GAP] = False; pr[GAP] = pr[GAP].fillna(False).astype(bool)
    hold = holdout_mask(pr); hidden = pr[GAP].to_numpy(bool); gaps_pr = hold | hidden
    # Keep sidecar truth, but blank every dynamic field on gaps.  Train and
    # private rows have disjoint keys, so concatenation is one-to-one.
    tr2 = tr.copy(); p2 = pr.copy(); tr2["_origin"]="train"; p2["_origin"]="private"
    dyn = [c for c in p2.columns if c not in [ID, DATE, "crop_type", GAP]]
    p2.loc[gaps_pr, dyn] = np.nan; p2.loc[gaps_pr, GAP] = True
    ref = pd.concat([tr2, p2], ignore_index=True, sort=False)
    ref[DATE] = pd.to_datetime(ref[DATE]); ref["_truth"] = pd.concat([tr[TARGET],pr[TARGET]],ignore_index=True)
    # labels from sidecar after sorting are safest via key map
    label_map = pd.concat([tr[[ID,DATE,TARGET]], pr[[ID,DATE,TARGET]]], ignore_index=True).rename(columns={TARGET:"_truth2"})
    ref = ref.merge(label_map, on=[ID,DATE], how="left", validate="one_to_one"); ref["_truth"] = ref["_truth2"]; ref.drop(columns="_truth2", inplace=True)
    gaps_ref = ref[GAP].fillna(False).to_numpy(bool)
    hk = set(map(tuple, pr.loc[hold,[ID,DATE]].to_numpy())); gaps_ref = gaps_ref | np.array([tuple(x) in hk for x in ref[[ID,DATE]].to_numpy()])
    ref.loc[gaps_ref,TARGET] = np.nan
    d,y,known = _panel_arrays(ref,gaps_ref)
    qkeys = pr.loc[hold,[ID,DATE]].copy(); qkeys[DATE]=pd.to_datetime(qkeys[DATE]); qkeys["truth"] = pr.loc[hold,TARGET].to_numpy(float)
    # map query positions in sorted ref
    key_to_i = {(a,b):i for i,(a,b) in enumerate(zip(d[ID].astype(str),d[DATE]))}
    qidx=np.array([key_to_i[(str(a),pd.Timestamp(b))] for a,b in qkeys[[ID,DATE]].itertuples(index=False,name=None)],dtype=int)
    rows=[]
    # Fixed baselines + factor configurations.
    for bw in (8,16,24,32):
        base = _seasonal_baseline(d,y,known,bw=bw,by_year=False)
        for rob in ("median","trim","mean"):
            for sh in (0.,.15,.5,1.5):
                p = _date_factor_prediction(d,y,known,qidx,base,robust=rob,shrink=sh)
                rows.append(score_frame(qkeys,p,f"factor_bw{bw}_{rob}_sh{sh}"))
        for rank in (1,2,3,4,6,8,12):
            for it in (8,20):
                p = _lowrank_completion(d,y,known,qidx,rank=rank,iters=it,bw=bw)
                rows.append(score_frame(qkeys,p,f"svd_bw{bw}_r{rank}_i{it}"))
        for k in (3,5,8,12,16):
            p = _multi_peer_ridge(d,y,known,qidx,base,k=k,alpha=10.,min_common=30)
            rows.append(score_frame(qkeys,p,f"ridge_bw{bw}_k{k}"))
    out=pd.DataFrame(rows).sort_values("rmse"); out.to_csv(R/"latent_factor_eval_results.csv",index=False)
    # cohort breakdown for top configurations
    train_ids=set(tr[ID].astype(str)); qkeys["cohort"]=np.where(qkeys[ID].astype(str).isin(train_ids),"shared","new"); qkeys["yearx"]=qkeys[DATE].dt.year
    top=out.head(12)["method"].tolist(); details=[]
    # recompute only top methods and retain predictions for blending downstream
    predtab=qkeys[[ID,DATE,"truth","cohort","yearx"]].copy()
    for name in top:
        # parse config; easiest rerun matching loops
        import re
        m=re.match(r"factor_bw(\d+)_(\w+)_sh([0-9.]+)",name)
        if m:
            bw,rob,sh=int(m.group(1)),m.group(2),float(m.group(3)); p=_date_factor_prediction(d,y,known,qidx,_seasonal_baseline(d,y,known,bw),robust=rob,shrink=sh)
        else:
            m=re.match(r"svd_bw(\d+)_r(\d+)_i(\d+)",name)
            if m:p=_lowrank_completion(d,y,known,qidx,rank=int(m.group(2)),iters=int(m.group(3)),bw=int(m.group(1)))
            else:
                m=re.match(r"ridge_bw(\d+)_k(\d+)",name); bw,k=int(m.group(1)),int(m.group(2));p=_multi_peer_ridge(d,y,known,qidx,_seasonal_baseline(d,y,known,bw),k=k,alpha=10.,min_common=30)
        predtab[name]=p
        for grp,g in [("all",np.ones(len(qkeys),bool)),("new",qkeys.cohort.eq("new").to_numpy()),("shared",qkeys.cohort.eq("shared").to_numpy()),("history",(qkeys.yearx<2025).to_numpy()),("2025",(qkeys.yearx==2025).to_numpy())]:
            rows2=score_frame(qkeys.loc[g],p[g],name); rows2["cohort"]=grp; details.append(rows2)
    predtab.to_csv(R/"latent_factor_eval_predictions.csv",index=False); pd.DataFrame(details).to_csv(R/"latent_factor_eval_cohorts.csv",index=False)
    print(out.head(30).to_string(index=False)); print(pd.DataFrame(details).sort_values(["cohort","rmse"]).head(30).to_string(index=False))


if __name__ == "__main__": main()

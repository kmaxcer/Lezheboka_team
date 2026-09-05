"""Nested, leakage-safe selection of observable AOI-peer corrections.

Unlike a plain retrospective grid, this experiment tunes the correction
coefficients and the peer configuration on *other* partitions, then scores the
held-out year/seed.  It is intentionally compact and research-only.  The
resulting nested OOF rows provide a conservative check before promoting any
fixed private submission rule.
"""
from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CANON_DOY = {97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289}


def _norm(x: pd.Series) -> pd.Series:
    return x.astype(str).str.replace(r"^(exact)(\d+)$", r"exact_\2", regex=True).str.replace(
        r"^(random)(\d+)$", r"random_\2", regex=True)


def load() -> pd.DataFrame:
    p = pd.read_csv(RESEARCH / "paired_aoi_v2_predictions.csv", parse_dates=["date"], low_memory=False)
    s = pd.read_csv(RESEARCH / "overnight_next_shock_predictions.csv", parse_dates=["date"], low_memory=False)
    s = s[s["candidate"].eq("baseline")].copy()
    s["partition_peer"] = _norm(s["partition"])
    z = p.merge(s[["partition_peer", "anon_polygon_id", "date", "shock", "state"]],
                left_on=["partition", "anon_polygon_id", "date"],
                right_on=["partition_peer", "anon_polygon_id", "date"], how="left", validate="one_to_one")
    if len(z) != len(p):
        raise ValueError("key join changed row count")
    z["dataset"] = np.where(z["family"].eq("exact"), "exact", "random")
    z["canon"] = z["date"].dt.dayofyear.isin(CANON_DOY).to_numpy(bool)
    z["base_hgb"] = z["hgb"].to_numpy(float)
    z["base_lag20"] = 0.8 * z["hgb"].to_numpy(float) + 0.2 * z["lag"].to_numpy(float)
    z["base_lag30"] = 0.7 * z["hgb"].to_numpy(float) + 0.3 * z["lag"].to_numpy(float)
    return z.reset_index(drop=True)


def folds(z: pd.DataFrame):
    """Yield (dataset, partition, train positions, test positions)."""
    for ds in ("exact", "random"):
        q = z.index[z["dataset"].eq(ds)].to_numpy(int)
        for part in sorted(z.loc[q, "partition"].unique()):
            test = q[z.loc[q, "partition"].to_numpy() == part]
            if ds == "exact":
                train = q[z.loc[q, "partition"].to_numpy() != part]
            else:
                # Masks overlap across random seeds.  Never use the same
                # (AOI,date) label in another seed as meta-training evidence.
                keys = set(zip(z.loc[test, "anon_polygon_id"].astype(str), z.loc[test, "date"].astype(str)))
                keep = np.array([tuple(v) not in keys for v in zip(
                    z.loc[q, "anon_polygon_id"].astype(str), z.loc[q, "date"].astype(str))], dtype=bool)
                train = q[keep]
            yield ds, str(part), train, test


def _peer_base(z: pd.DataFrame, base: str, cfg: str, w: float) -> tuple[np.ndarray, np.ndarray]:
    b = z[base].to_numpy(float)
    q = z[cfg].to_numpy(float)
    ok = np.isfinite(q)
    p = b.copy()
    p[ok] = (1.0 - w) * b[ok] + w * q[ok]
    return p, ok


def _fit_theta(resid: np.ndarray, sh: np.ndarray, st: np.ndarray, idx: np.ndarray,
               mode: str, ridge: float = 1e-4) -> tuple[float, float]:
    """Fit correction coefficients on train positions only, with sign bounds."""
    if mode == "none":
        return 0.0, 0.0
    use = np.ones(len(idx), dtype=bool)
    if mode.startswith("canon"):
        # Caller supplies arrays already zeroed on canon rows.
        pass
    x1, x2 = sh[idx], st[idx]
    X = np.c_[x1, x2] if mode.endswith("joint") else (np.c_[x1, np.zeros(len(idx))] if mode.endswith("shock") else np.c_[np.zeros(len(idx)), x2])
    yy = resid[idx]
    good = np.isfinite(yy) & np.isfinite(X).all(axis=1)
    if good.sum() < 30:
        return 0.15 if mode.endswith("joint") or mode.endswith("shock") else 0.0, -0.05 if mode.endswith("joint") else 0.0
    X = X[good]; yy = yy[good]
    gram = X.T @ X + ridge * np.eye(2)
    rhs = X.T @ yy
    try:
        th = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        th = np.array([0.15, -0.05])
    # Sign/range bounds are predeclared safeguards, not tuned on the test fold.
    th[0] = float(np.clip(th[0], 0.0, 0.40))
    th[1] = float(np.clip(th[1], -0.40, 0.10))
    if mode.endswith("shock"):
        th[1] = 0.0
    if mode.endswith("state"):
        th[0] = 0.0
    return float(th[0]), float(th[1])


def _apply(basep: np.ndarray, sh: np.ndarray, st: np.ndarray, canon: np.ndarray,
           a: float, b: float, mode: str) -> np.ndarray:
    c = a * sh + b * st
    if mode.startswith("canon"):
        c = c.copy(); c[canon] = 0.0
    return basep + c


def main() -> None:
    z = load()
    cfgs = sorted(c for c in z.columns if c.startswith("n") and "_c" in c and "_r" in c and "_k" in c)
    bases = ["base_hgb", "base_lag20", "base_lag30"]
    weights = [0.05, 0.08, 0.10, 0.12, 0.15]
    modes = ["none", "canon_joint", "canon_shock", "all_joint", "all_shock", "canon_state"]
    y = z["_truth"].to_numpy(float)
    sh = np.nan_to_num(z["shock"].to_numpy(float), nan=0.0)
    st = np.nan_to_num(z["state"].to_numpy(float), nan=0.0)
    canon = z["canon"].to_numpy(bool)

    # Candidate set is all peer configs but only the scientifically motivated
    # coefficient modes.  Coefficients are estimated on the outer-train side,
    # so no retrospective coefficient grid is needed here.
    rules = list(itertools.product(bases, cfgs, weights, modes))
    out_rows = []
    chosen_rows = []
    for ds, part, train, test in folds(z):
        # Evaluate each candidate's train fit and retain its held-out score.
        best = None
        for base, cfg, w, mode in rules:
            bp, covered = _peer_base(z, base, cfg, w)
            resid = y - bp
            # Canon modes are represented by zeroed features on canon rows.
            ssh = sh.copy(); sst = st.copy()
            if mode.startswith("canon"):
                ssh[canon] = 0.0; sst[canon] = 0.0
            a, b = _fit_theta(resid, ssh, sst, train, mode)
            pred = _apply(bp, sh, st, canon, a, b, mode)
            tr_ok = np.isfinite(y[train]) & np.isfinite(pred[train])
            te_ok = np.isfinite(y[test]) & np.isfinite(pred[test])
            if tr_ok.sum() < 20 or te_ok.sum() == 0:
                continue
            tr_mse = float(np.mean((pred[train][tr_ok] - y[train][tr_ok]) ** 2))
            te_mse = float(np.mean((pred[test][te_ok] - y[test][te_ok]) ** 2))
            key = (tr_mse, te_mse)
            # Selection is train-only; test MSE is retained only for reporting.
            if best is None or key[0] < best[0]:
                best = (tr_mse, te_mse, base, cfg, w, mode, a, b, covered[test].mean(), len(test), te_ok.sum())
        if best is None:
            continue
        tr_mse, te_mse, base, cfg, w, mode, a, b, cov, ntest, nvalid = best
        # Same-base no-peer baseline for interpretable held-out delta.
        bp, _ = _peer_base(z, base, cfg, 0.0)
        bmse = float(np.mean((bp[test] - y[test]) ** 2))
        out_rows.append({"dataset": ds, "partition": part, "n": ntest, "valid": nvalid,
                         "base": base, "peer_config": cfg, "peer_weight": w, "mode": mode,
                         "alpha": a, "beta": b, "coverage": cov,
                         "train_mse": tr_mse, "rmse": np.sqrt(te_mse),
                         "baseline_rmse": np.sqrt(bmse), "delta_rmse": np.sqrt(te_mse) - np.sqrt(bmse)})
        chosen_rows.append({"dataset": ds, "partition": part, "base": base, "peer_config": cfg,
                            "peer_weight": w, "mode": mode, "alpha": a, "beta": b,
                            "train_mse": tr_mse, "test_rmse": np.sqrt(te_mse), "baseline_rmse": np.sqrt(bmse)})

    chosen = pd.DataFrame(chosen_rows)
    out = pd.DataFrame(out_rows)
    out.to_csv(RESEARCH / "ensemble_cv_v2_nested_folds.csv", index=False, float_format="%.9f")
    chosen.to_csv(RESEARCH / "ensemble_cv_v2_nested_choices.csv", index=False, float_format="%.9f")
    lines = ["# Nested observable ensemble selection", "",
             "For each held-out exact year/random seed, peer config, weight, and correction coefficients were selected using other partitions only. Random overlap AOI/date keys were excluded from the fit.", "",
             "## Choices", "", chosen.to_string(index=False), ""]
    if len(out):
        lines += ["## Pooled nested result", ""]
        for ds, g in out.groupby("dataset"):
            rm = float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"])))
            br = float(np.sqrt(np.average(g["baseline_rmse"] ** 2, weights=g["n"])))
            lines.append(f"- {ds}: baseline {br:.6f} -> nested {rm:.6f} (delta {rm-br:+.6f}); folds improved {(g['delta_rmse'] < 0).sum()}/{len(g)}")
    lines += ["", "This nested selection is a stability diagnostic; the deployable private rule remains a predeclared fixed formula from ensemble_cv_v2_apply_peer.py."]
    (RESEARCH / "ensemble_cv_v2_nested_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(chosen.to_string(index=False))
    print("\n", "\n".join(lines[-4:]))


if __name__ == "__main__":
    main()

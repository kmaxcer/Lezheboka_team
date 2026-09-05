"""Fast direct-state sweep for the tail anomaly experiment.

The full ``tail_anomaly_v2.py`` tests learned residual models.  This companion
keeps only physically interpretable state estimates (visible normalized
neighbours and same-date peers) and sweeps conservative blends against the
fixed HGB+lag baseline.  It is useful for finding a stable low-variance rule
without fitting a high-dimensional model.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_archive_inspect" / "agropulse_max_score" / "data"
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))
import tail_anomaly_v2 as tv  # noqa: E402
from teammate_sweep_postcorr import _mask_private  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from validate import make_fold  # noqa: E402


def build(frame: pd.DataFrame, mask: np.ndarray, pmap: pd.DataFrame,
          dataset: str, partition: str) -> pd.DataFrame:
    return tv._make_part(frame, mask, dataset, partition, pmap)


def evaluate(q: pd.DataFrame, tag: str, rows: list[dict]) -> None:
    y = q["truth"].to_numpy(float)
    b = q["baseline"].to_numpy(float)
    # A direct local state estimate in NDVI units.  cm/cs are reconstructed
    # only from visible rows by the feature builder.
    cm = q["cm_local"].to_numpy(float); cs = q["cs_local"].to_numpy(float)
    states = {
        "z_w7": q["z_w7"].to_numpy(float),
        "z_w14": q["z_w14"].to_numpy(float),
        "z_w30": q["z_w30"].to_numpy(float),
        "z_w60": q["z_w60"].to_numpy(float),
        "z_interp": q["z_interp"].to_numpy(float),
        "peer": q["peer_zmed"].to_numpy(float),
        "peer30": q["peer_zmed_w30"].to_numpy(float),
        "local_peer": 0.5 * (q["z_w30"].to_numpy(float) + q["peer_zmed"].to_numpy(float)),
    }
    for sn, z in states.items():
        # State estimate and deviation from the baseline's implied state.
        s = cm + z * cs
        zb = (b - cm) / np.maximum(cs, 0.025)
        delta = (z - zb) * cs
        for mode, x in [("state", s), ("delta", b + delta)]:
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            for w in np.arange(0.0, 0.51, 0.025):
                p = b[ok] * (1.0 - w) + x[ok] * w
                e = p - y[ok]
                rows.append({"tag": tag, "candidate": f"{mode}_{sn}_w{w:.3f}", "n": int(ok.sum()),
                             "rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e)))})
    # A robust sign/gate rule: only act when local and peer states agree and
    # there are enough visible neighbours.  This targets persistent tails
    # while shrinking isolated outliers toward the trusted baseline.
    zl = q["z_w30"].to_numpy(float); zp = q["peer_zmed"].to_numpy(float)
    agree = np.isfinite(zl) & np.isfinite(zp) & (np.sign(zl) == np.sign(zp))
    conf = agree & (q["n30"].to_numpy(float) >= 3) & (q["peer_n"].to_numpy(float) >= 3)
    x = cm + 0.5 * (zl + zp) * cs
    for w in np.arange(0.0, 0.51, 0.025):
        p = b.copy(); ok = conf & np.isfinite(x); p[ok] = (1-w)*b[ok] + w*x[ok]
        e = p-y; rows.append({"tag":tag,"candidate":f"gate_w{w:.3f}","n":len(y),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(abs(e)))})
    # Quantile/tail gate, deliberately conservative and label-free.
    for cut in (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0):
        gate = np.isfinite(zl) & ((zl < cut) if cut < 0 else (zl > cut))
        for w in (0.05, 0.1, 0.2, 0.3):
            p=b.copy(); ok=gate & np.isfinite(x);p[ok]=(1-w)*b[ok]+w*x[ok];e=p-y
            rows.append({"tag":tag,"candidate":f"tail{cut:+.1f}_w{w:.2f}","n":len(y),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(abs(e)))})


def main() -> None:
    tr = pd.read_csv(DATA / "train_dataset.csv", parse_dates=["date"], low_memory=False)
    pr = pd.read_csv(DATA / "private_features.csv", parse_dates=["date"], low_memory=False)
    pp = pd.read_csv(RESEARCH / "teammate_sweep_postcorr_preds.csv", parse_dates=["date"], low_memory=False)
    rows: list[dict] = []
    for year in range(2019, 2025):
        d, _ = make_fold(tr.copy(), pr.copy(), year); m = d["is_synthetic_gap"].fillna(False).to_numpy(bool)
        pmap = pp[(pp.dataset == "exact_hidden_doy") & (pp.partition == f"exact{year}") & (pp.method == tv.BASE_METHOD)][tv.KEY + ["pred"]].rename(columns={"pred":"baseline"})
        q = build(d, m, pmap, "exact_hidden_doy", f"exact{year}"); evaluate(q, f"exact{year}", rows)
        print("exact", year, flush=True)
    for seed in (0, 1, 2):
        d, m = _mask_private(pr.copy(), seed)
        pmap = pp[(pp.dataset == "random_private_like") & (pp.partition == f"random{seed}") & (pp.method == tv.BASE_METHOD)][tv.KEY + ["pred"]].rename(columns={"pred":"baseline"})
        q = build(d, m, pmap, "random_private_like", f"random{seed}"); evaluate(q, f"random{seed}", rows)
        print("random", seed, flush=True)
    met = pd.DataFrame(rows); met.to_csv(RESEARCH / "tail_anomaly_v2_direct_metrics.csv", index=False)
    agg = []
    for (tag, cand), g in met.groupby([met.tag.str.replace(r"exact\\d+|random\\d+", "", regex=True), "candidate"], sort=False):
        pass
    met["protocol"] = np.where(met.tag.str.startswith("exact"), "exact_hidden_doy", "random_private_like")
    for (proto, cand), g in met.groupby(["protocol", "candidate"], sort=False):
        agg.append({"protocol":proto,"candidate":cand,"n":int(g.n.sum()),"rmse_pooled":float(np.sqrt(np.average(g.rmse**2,weights=g.n))),"mae_pooled":float(np.average(g.mae,weights=g.n)),"parts":len(g)})
    ag = pd.DataFrame(agg).sort_values(["protocol","rmse_pooled"]); ag.to_csv(RESEARCH / "tail_anomaly_v2_direct_aggregate.csv",index=False)
    print(ag.head(80).to_string(index=False))
    (RESEARCH / "tail_anomaly_v2_direct_report.md").write_text("# Tail anomaly v2 direct-state sweep\n\n" + ag.head(80).to_string(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()

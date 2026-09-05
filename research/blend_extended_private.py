"""Create separate blends of extended-HGB and the strongest local candidate.

The script only joins already validated three-column components by AOI/date,
checks the hidden-key contract, and writes research candidates.  By default
the extended component is attenuated on 2025 rows because the independent
2025 proxy showed no gain there; global-weight variants are emitted for
comparison.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = Path(r"C:/Users/kmaxc/Documents/Codex/2026-09-04/ml/work/cosmo_latest_20260904")
EXT = {
    "regular": OUT / "model_dani_extended_hgb_regular.csv",
    "wide": OUT / "model_dani_extended_hgb_wide.csv",
}
BASE = OUT / "model_dani_lag40_peer10_a350_b200_submission.csv"
PRIVATE = DATA / "private_features.csv"
KEY = ["anon_polygon_id", "date"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path: Path) -> pd.DataFrame:
    z = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    req = KEY + ["primary_ndvi_pred"]
    if list(z.columns) != req or z.duplicated(KEY).any() or not np.isfinite(z.primary_ndvi_pred).all():
        raise ValueError(f"bad component contract: {path}")
    return z


def main() -> None:
    private = pd.read_csv(PRIVATE, parse_dates=["date"], low_memory=False)
    hidden = private["is_synthetic_gap"].fillna(False).astype(bool)
    keys = private.loc[hidden, KEY].copy().reset_index(drop=True)
    base = read(BASE)
    if len(base) != len(keys) or set(map(tuple, base[KEY].to_numpy())) != set(map(tuple, keys.to_numpy())):
        raise ValueError("base keys do not match hidden rows")
    base = keys.merge(base, on=KEY, how="left", validate="one_to_one")
    years = keys.date.dt.year.to_numpy(int)
    outputs = []
    for kind, path in EXT.items():
        ext = keys.merge(read(path), on=KEY, how="left", validate="one_to_one")["primary_ndvi_pred"].to_numpy(float)
        for w in (0.20, 0.30, 0.40):
            p = (1.0 - w) * base.primary_ndvi_pred.to_numpy(float) + w * ext
            out = keys.copy(); out["primary_ndvi_pred"] = np.clip(p, -.2, 1.1)
            name = f"model_dani_lag40_peer10_ext{kind}{int(round(100*w)):02d}_submission.csv"
            out_path = OUT / name; out.to_csv(out_path, index=False, float_format="%.8f")
            outputs.append({"candidate": name, "kind": kind, "ext_weight": w, "year2025_ext_weight": w,
                            "rows": len(out), "sha256": sha(out_path), "min": float(out.primary_ndvi_pred.min()), "max": float(out.primary_ndvi_pred.max())})
            # Conditional variant: retain the tested lag/peer model on 2025,
            # blend extension only on 2010--2024 hidden rows.
            wc = np.where(years == 2025, 0.0, w)
            pc = (1.0 - wc) * base.primary_ndvi_pred.to_numpy(float) + wc * ext
            outc = keys.copy(); outc["primary_ndvi_pred"] = np.clip(pc, -.2, 1.1)
            namec = f"model_dani_lag40_peer10_ext{kind}{int(round(100*w)):02d}_no2025_submission.csv"
            pathc = OUT / namec; outc.to_csv(pathc, index=False, float_format="%.8f")
            outputs.append({"candidate": namec, "kind": kind, "ext_weight": w, "year2025_ext_weight": 0.0,
                            "rows": len(outc), "sha256": sha(pathc), "min": float(outc.primary_ndvi_pred.min()), "max": float(outc.primary_ndvi_pred.max())})
    meta = {"base": BASE.name, "base_sha256": sha(BASE), "private_sha256": sha(PRIVATE), "hidden_rows": int(hidden.sum()), "outputs": outputs, "production_baseline_overwritten": False}
    (OUT / "model_dani_extended_blends_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

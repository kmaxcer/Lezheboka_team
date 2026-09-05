"""Exact LOO audit for source-specific residual clipping from hparam probe."""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "research"
SEEDS = (0, 1, 2, 70404)

def rm(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))

def main():
    p = pd.read_csv(R / "source_expert_route_v2_hparam_probe_v1_predictions.csv", parse_dates=["date"], low_memory=False)
    p = p[p.variant.eq("current_refit")].copy()
    s = pd.read_csv(R / "source_schedule_route_probe_rows.csv", parse_dates=["date"], low_memory=False)
    s = s[["anon_polygon_id", "date", "seed", "sp_crop_2", "sp_crop_2_n", "sp_crop_8_n"]]
    d = p.merge(s, on=["anon_polygon_id", "date", "seed"], validate="one_to_one")
    idx = d.sp_crop_2.to_numpy(int).copy(); fb = idx < 0; idx[fb] = 0
    # For the very rare no-r2-peer rows the selected source is fallback; using
    # the widest cap is conservative and changes only clipping diagnostics.
    cap = np.where(idx == 0, .10, np.where(idx == 1, .08, .06))
    b = d.baseline.to_numpy(float); y = d.truth.to_numpy(float); e = d.psrc.to_numpy(float); sa = d.seed.to_numpy(int)
    out = []
    for held in SEEDS:
        tr = sa != held; te = ~tr
        for variant in ("none_a040", "srcclip_a040", "srcclip_policy"):
            def pred(mask):
                bb = b[mask]; ee = e[mask]
                if variant.startswith("srcclip"):
                    ee = bb + np.clip(ee - bb, -cap[mask], cap[mask])
                aa = np.full(mask.sum(), .40) if variant.endswith("a040") else d.alpha_policy.to_numpy(float)[mask]
                return (1 - aa) * bb + aa * ee
            out.append({"held_seed": held, "variant": variant, "train_rmse": rm(y[tr], pred(tr)), "test_rmse": rm(y[te], pred(te)), "test_base": rm(y[te], b[te])})
    z = pd.DataFrame(out)
    z.to_csv(R / "source_expert_route_v2_hparam_probe_v1_sourceclip_loo.csv", index=False, float_format="%.10f")
    (R / "source_expert_route_v2_hparam_probe_v1_sourceclip_loo_report.md").write_text("# Exact source-specific clip LOO\n\n" + z.to_string(index=False) + "\n", encoding="utf-8")
    print(z.to_string(index=False))

if __name__ == "__main__": main()

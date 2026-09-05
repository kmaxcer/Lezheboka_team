"""Audit observable fallback source modes for trainaug fixed-r2 route."""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "research"
SEEDS = (0, 1, 2, 70404)

def rm(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))

def main():
    # Use the same deterministic q1 expert matrix (private-only route audit
    # sidecar) so every fallback mode can select among all three experts.
    d = pd.read_csv(R / "source_expert_route_v2_fixed_radius_srcfix_rows.csv", parse_dates=["date"], low_memory=False)
    s = pd.read_csv(R / "source_schedule_route_probe_rows.csv", parse_dates=["date"], low_memory=False)
    cols = ["anon_polygon_id", "date", "seed", "sp_crop_2", "sp_crop_2_n", "sp_crop_8_n", "sp4_sched", "sp1_temp1", "sp2_temp2", "sp1_cross", "sp1_doy", "doy_crop", "doy"]
    d = d.merge(s[cols], on=["anon_polygon_id", "date", "seed"], validate="one_to_one")
    y = d.truth.to_numpy(float); b = d.baseline.to_numpy(float); e = d[["e_s2", "e_landsat", "e_modis"]].to_numpy(float); route = d.sp_crop_2.to_numpy(int); has = route >= 0; sa = d.seed.to_numpy(int); yr = d.year.to_numpy(int); co = d.cohort.to_numpy(str)
    near = d.sp_crop_2_n.to_numpy(int) > 0; mid = (~near) & (d.sp_crop_8_n.to_numpy(int) > 0); a = np.where(near, .5, np.where(mid, .4, .3)); a = np.where((co == "new") & (yr == 2025), .6, a); a = np.where((co == "shared") & (yr == 2025), .35, a); post = d.route_post_mode.to_numpy(int)
    fallbacks = {"current_post": post, "sp4_sched": d.sp4_sched.to_numpy(int), "sp1_temp1": d.sp1_temp1.to_numpy(int), "sp2_temp2": d.sp2_temp2.to_numpy(int), "sp1_cross": d.sp1_cross.to_numpy(int), "sp1_doy": d.sp1_doy.to_numpy(int), "doy_crop": d.doy_crop.to_numpy(int), "doy": d.doy.to_numpy(int)}
    rec = []; loo = []
    for fn, fv in fallbacks.items():
        idx = route.copy(); use = ~has; idx[use] = fv[use]; idx[idx < 0] = post[idx < 0]; psrc = e[np.arange(len(e)), idx]
        for pol, aa in [("a040", np.full(len(y), .4)), ("cyd", a)]:
            p = (1 - aa) * b + aa * psrc; rec.append({"fallback": fn, "policy": pol, "n": len(y), "rmse": rm(y, p), "base_rmse": rm(y, b), "per_seed": ";".join(f"{z}:{rm(y[sa == z], p[sa == z]):.6f}" for z in SEEDS), "fallback_rows": int(use.sum())})
        true = d.true_src.map({"s2": 0, "landsat": 1, "modis": 2}).to_numpy(int); rec.append({"fallback": fn, "policy": "source_acc_fallback", "n": int(use.sum()), "rmse": np.nan, "base_rmse": np.nan, "per_seed": f"acc={np.mean(idx[use] == true[use]):.6f}", "fallback_rows": int(use.sum())})
        for held in SEEDS:
            te = sa == held; p = (1 - a) * b + a * psrc; pc = (1 - a[te]) * b[te] + a[te] * e[np.arange(len(e))[te], post[te]]; loo.append({"fallback": fn, "held_seed": held, "test_rmse": rm(y[te], p[te]), "test_base": rm(y[te], b[te]), "test_current": rm(y[te], pc)})
    md = pd.DataFrame(rec).sort_values("rmse", na_position="last"); ld = pd.DataFrame(loo); stem = "source_expert_route_v2_trainaug_r2_fallback"; md.to_csv(R / (stem + "_metrics.csv"), index=False, float_format="%.10f"); ld.to_csv(R / (stem + "_loo.csv"), index=False, float_format="%.10f"); (R / (stem + "_report.md")).write_text("# Trainaug r2 fallback audit\n\n" + md.to_string(index=False) + "\n\nLOO\n" + ld.to_string(index=False) + "\n", encoding="utf-8"); print(md.to_string(index=False)); print(ld.to_string(index=False))

if __name__ == "__main__": main()

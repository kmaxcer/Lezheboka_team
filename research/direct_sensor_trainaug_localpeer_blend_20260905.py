"""Stack direct spatial sensor residual on trainaug-r2 local-peer base."""
from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"
REP = ROOT / "reports"
ID, DATE = "anon_polygon_id", "date"
keys = [ID, DATE, "seed"]

rows = pd.read_csv(R / "source_expert_route_v2_fixed_radius_trainaug_rows.csv",
                   parse_dates=[DATE], low_memory=False)
probe = pd.read_csv(R / "source_schedule_route_probe_rows.csv",
                    parse_dates=[DATE], low_memory=False)
rows = rows.merge(probe[keys + ["sp_crop_2", "sp_crop_2_n", "sp_crop_8_n"]],
                  on=keys, how="left", validate="one_to_one")
fallback = pd.read_csv(R / "source_expert_route_v2_fixed_radius_srcfix_rows.csv",
                       parse_dates=[DATE], low_memory=False)
rows = rows.merge(fallback[keys + ["route_post_mode"]], on=keys,
                  how="left", validate="one_to_one")
idx = rows.sp_crop_2.to_numpy(int).copy()
fb = idx < 0
idx[fb] = rows.route_post_mode.to_numpy(int)[fb]
E = rows.expert_trainaug_r2.to_numpy(float)
B = rows.baseline.to_numpy(float)
# The fixed-radius trainaug sidecar already stores the routed expert scalar;
# no source matrix remains in this compact file.
Psrc = E
n2 = rows.sp_crop_2_n.to_numpy(float)
n8 = rows.sp_crop_8_n.to_numpy(float)
near = n2 > 0
mid = (~near) & (n8 > 0)
aa = np.where(near, .5, np.where(mid, .4, .3))
yy = rows.year.to_numpy(int)
co = rows.cohort.astype(str).to_numpy()
aa = np.where((co == "new") & (yy == 2025), .6, aa)
aa = np.where((co == "shared") & (yy == 2025), .35, aa)
base = B + aa * (Psrc - B)

lf = pd.read_csv(R / "local_peer_residual_v1_features.csv",
                 parse_dates=[DATE], low_memory=False)
lf = lf[[ID, DATE, "seed", "r8_crop_resmean"]]
rbase = rows[[ID, DATE, "seed"]].copy()
rbase["base_trainaug_local"] = base
rbase["near_trainaug"] = near
g = lf.merge(rbase, on=keys, how="inner", validate="one_to_one")
d = pd.read_csv(R / "direct_spatial_sensor_fast_rows_20260905.csv",
                 parse_dates=[DATE], low_memory=False)
g = g.merge(d, on=keys, how="inner", suffixes=("", "_d"),
            validate="one_to_many")
g["local"] = np.clip(g.base_trainaug_local.to_numpy(float)
                     + .20 * g.r8_crop_resmean.fillna(0).to_numpy(float), -.2, 1.1)
g["D"] = np.where(np.isfinite(g.mix.to_numpy(float)), g.mix.to_numpy(float), g.local.to_numpy(float))

def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))

out = []
for (radius, method), z in g.groupby(["radius", "method"]):
    y = z.truth.to_numpy(float)
    L = z.local.to_numpy(float)
    D = z.D.to_numpy(float)
    ss = z.seed.to_numpy(int)
    for beta in [-.10, -.05, -.03, -.02, -.01, 0, .005, .01,
                 .015, .02, .03, .05, .08, .10]:
        p = L + beta * (D - L)
        for s in sorted(np.unique(ss)):
            ix = ss == s
            out.append({"radius": int(radius), "method": method,
                        "pred": f"local_plus_direct_{beta:g}", "seed": int(s),
                        "n": int(ix.sum()), "rmse": rmse(p[ix] - y[ix])})
        out.append({"radius": int(radius), "method": method,
                    "pred": f"local_plus_direct_{beta:g}", "seed": -1,
                    "n": len(y), "rmse": rmse(p - y)})
    nr = z.near_trainaug.to_numpy(bool)
    for bn in [0, .005, .01, .02, .03]:
        for bf in [0, .005, .01, .02, .03, .05]:
            be = np.where(nr, bn, bf)
            p = L + be * (D - L)
            tag = f"bucket_{bn:g}_{bf:g}"
            for s in sorted(np.unique(ss)):
                ix = ss == s
                out.append({"radius": int(radius), "method": method,
                            "pred": tag, "seed": int(s), "n": int(ix.sum()),
                            "rmse": rmse(p[ix] - y[ix])})
            out.append({"radius": int(radius), "method": method,
                        "pred": tag, "seed": -1, "n": len(y),
                        "rmse": rmse(p - y)})
m = pd.DataFrame(out)
m.to_csv(R / "direct_sensor_trainaug_localpeer_blend_metrics_20260905.csv", index=False)
pooled = []
for (radius, method, pred), z in m[m.seed == -1].groupby(
        ["radius", "method", "pred"]):
    per = m[(m.radius == radius) & (m.method == method)
            & (m.pred == pred) & (m.seed >= 0)]
    pooled.append({"radius": radius, "method": method, "pred": pred,
                   "pooled_rmse": float(z.rmse.iloc[0]),
                   "min_seed_rmse": float(per.rmse.min()),
                   "max_seed_rmse": float(per.rmse.max())})
p = pd.DataFrame(pooled).sort_values("pooled_rmse")
p.to_csv(R / "direct_sensor_trainaug_localpeer_blend_pooled_20260905.csv", index=False)
print("merged", len(g), flush=True)
print(p.head(50).to_string(index=False), flush=True)
REP.mkdir(exist_ok=True)
best = p.iloc[0].to_dict() if len(p) else {}
(REP / "direct_sensor_trainaug_localpeer_blend_report_20260905.md").write_text(
    "# Direct sensor stacked on trainaug local-peer\n\n"
    "Trainaug-r2 cohort/year route plus fixed local peer correction (.20) is the base. "
    "Direct same-date crop sensor summaries use visible train + private rows only.\n\n"
    + "Best: " + json.dumps(best) + "\n"
    + "Artifacts: `research/direct_sensor_trainaug_localpeer_blend_metrics_20260905.csv`, "
      "`research/direct_sensor_trainaug_localpeer_blend_pooled_20260905.csv`.\n",
    encoding="utf-8")

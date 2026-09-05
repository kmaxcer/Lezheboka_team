from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\kmaxc\Documents\Codex\2026-09-04\ml")
for p in ROOT.rglob("*.csv"):
    if any(x in p.parts for x in [".venv", "venv", "site-packages", "node_modules"]):
        continue
    try:
        d = pd.read_csv(p, nrows=3)
    except Exception:
        continue
    cols = list(d.columns)
    if "primary_ndvi_pred" in cols or "anon_polygon_id" in cols:
        print(p, "cols=", cols, "bytes=", p.stat().st_size)

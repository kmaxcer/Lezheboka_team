from pathlib import Path
p=Path('research/feature_hgb_v3_calfix.py'); s=p.read_text(); s=s.replace(').to_numpy(float)`n            out', ').to_numpy(float)\n            out'); p.write_text(s)

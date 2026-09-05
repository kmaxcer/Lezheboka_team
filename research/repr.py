from pathlib import Path
s=Path('research/feature_hgb_v3_calfix.py').read_text(); i=s.index('out[f"{col}_global_cal"] = intercept'); print(repr(s[i:i+250]))

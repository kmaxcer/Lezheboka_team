from pathlib import Path
p=Path('research/feature_hgb_v3_calfix.py'); ls=p.read_text().splitlines(); out=[]
for line in ls:
 if 'out[f"{col}_global_cal"] = intercept' in line:
  out.append('            out[f"{col}_global_cal"] = intercept + slope * out[f"{col}_prev"].to_numpy(float)')
  out.append('            out[f"{col}_global_cal_next"] = intercept + slope * out[f"{col}_next"].to_numpy(float)')
 else: out.append(line)
p.write_text('\n'.join(out)+'\n')

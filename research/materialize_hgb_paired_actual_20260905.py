from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
base_path = ROOT / 'outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_20260905_submission.csv'
hgb_path = ROOT / 'research/hgb_actual_sqclip_predictions_20260905.csv'
private_path = Path(r'C:\Users\kmaxc\Documents\Codex\2026-09-04\ml\work\cosmo_latest_20260904\private_features.csv')
out_path = ROOT / 'outputs/model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_paired_n12_c40_r100_k2_w008_hgb_sqclip_w015_20260905_submission.csv'
meta_path = out_path.with_suffix('.metadata.json')
report_path = ROOT / 'reports/hgb_sqclip_paired_actual_20260905.md'

if out_path.exists():
    raise FileExistsError(out_path)
base = pd.read_csv(base_path)
hgb = pd.read_csv(hgb_path)
priv = pd.read_csv(private_path, encoding='cp1251')
gaps = priv.loc[priv['is_synthetic_gap'].astype(bool), ['anon_polygon_id','date']].copy()
gaps['date'] = pd.to_datetime(gaps['date']).dt.strftime('%Y-%m-%d')
for df in (base, hgb):
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
keys = ['anon_polygon_id','date']
if len(base) != 3112 or len(hgb) != 3112 or base[keys].duplicated().any() or hgb[keys].duplicated().any():
    raise ValueError('input contract failure')
if set(map(tuple, base[keys].to_numpy())) != set(map(tuple, gaps[keys].to_numpy())):
    raise ValueError('base keys differ from exact synthetic gaps')
if set(map(tuple, hgb[keys].to_numpy())) != set(map(tuple, gaps[keys].to_numpy())):
    raise ValueError('hgb keys differ from exact synthetic gaps')
hgb = base[keys].merge(hgb, on=keys, how='left', validate='one_to_one')
pred = 0.85 * base['primary_ndvi_pred'].to_numpy(float) + 0.15 * hgb['hgb_sq_clip'].to_numpy(float)
pred = np.clip(pred, -0.2, 1.1)
out = base[keys].copy()
out['primary_ndvi_pred'] = pred
if not np.isfinite(pred).all() or out[keys].duplicated().any() or len(out) != 3112:
    raise ValueError('output contract failure')
out.to_csv(out_path, index=False, float_format='%.10f')

sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
meta = {
    'formula': 'pair08 = finite(peer) ? 0.92*base25 + 0.08*peer : base25; final = clip(0.85*pair08 + 0.15*hgb_sq_clip, [-0.2, 1.1])',
    'base': str(base_path), 'hgb': str(hgb_path), 'rows': int(len(out)),
    'coverage_hgb': float(np.isfinite(hgb['hgb_sq_clip']).mean()), 'sha256': sha,
    'upload_performed': False,
}
meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

rows_path = ROOT / 'research/hgb_exact_mask_validation_20260905_rows.csv'
rows = pd.read_csv(rows_path)
peer = rows['n12_c40_r100_k2'].to_numpy(float)
base_arr = rows['base25'].to_numpy(float)
pair08 = np.where(np.isfinite(peer), 0.92 * base_arr + 0.08 * peer, base_arr)
blend = np.clip(0.85 * pair08 + 0.15 * rows['hgb_sq_clip'].to_numpy(float), -0.2, 1.1)
err = blend - rows['truth'].to_numpy(float)
base_err = pair08 - rows['truth'].to_numpy(float)
def rmse(mask):
    return float(np.sqrt(np.mean(err[mask] ** 2)))
def rmse0(mask):
    return float(np.sqrt(np.mean(base_err[mask] ** 2)))
metrics = []
for seed in sorted(rows['seed'].unique()):
    m = rows['seed'].to_numpy() == seed
    metrics.append((f'seed{seed}', int(m.sum()), rmse(m), rmse0(m), rmse(m)-rmse0(m)))
allm = np.ones(len(rows), dtype=bool)
metrics.append(('pooled', len(rows), rmse(allm), rmse0(allm), rmse(allm)-rmse0(allm)))
lines = [
    '# HGB sq_clip paired actual-gap candidate (2026-09-05)', '',
    f'Output: `{out_path}`', f'SHA256: `{sha}`', '',
    'Formula: `pair08 = finite(peer) ? 0.92 * base25 + 0.08 * paired(n12_c40_r100_k2) : base25`; `pred = clip(0.85 * pair08 + 0.15 * hgb_sq_clip, -0.2, 1.1)`.',
    'HGB was trained only from train plus visible private rows using three leakage-safe pseudo-gap blocks; exact-mask validation joins predictions only to identical `(anon_polygon_id,date)` holdout keys.', '',
    'Proxy exact-mask RMSE (not hidden-label score):', '',
    '| mask | n | blend RMSE | pair08 RMSE | delta |', '|---|---:|---:|---:|---:|'
]
for name,n,r,b,d in metrics:
    lines.append(f'| {name} | {n} | {r:.6f} | {b:.6f} | {d:+.6f} |')
lines += ['', 'Actual-gap coverage: 3112 / 3112 finite HGB values; output coverage 3112 / 3112.', 'The output has exactly `anon_polygon_id,date,primary_ndvi_pred`, 3112 unique keys, finite predictions, and was not uploaded or submitted.']
report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False))
print('\n'.join(lines))

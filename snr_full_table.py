# -*- coding: utf-8 -*-
"""全部模型 × 全部 SNR 档 × P/S 的 ≤0.1 s 命中率与 MAE（窗口级）"""
import pandas as pd, numpy as np, glob, os

MODELS = [('BiMamba 199K+长窗6000', 'eval_all_pm_w6000_all'),
          ('BiMamba 199K（3000）',   'eval_all_pm_pn197k'),
          ('BiMamba 328K（3000）',   'eval_all_pm_pm322k'),
          ('BiMamba 463K（3000）',   'eval_all_pm_eqt466k'),
          ('BiMamba 813K（3000）',   'eval_all_pm_813k'),
          ('BiMamba 199K+增广',      'eval_all_pm_offaug'),
          ('PhaseNet-3000',         'eval_all_pm_phasenet'),
          ('PhaseNet-6000',         'eval_all_pm_phasenet6000'),
          ('PhaseMamba-3000',       'eval_all_pm_phasemamba'),
          ('PhaseMamba-6000',       'eval_all_pm_phasemamba6000'),
          ('EQTransformer-3000',    'eval_all_pm_eqt'),
          ('EQTransformer-6000',    'eval_all_pm_eqt6000')]
BINS = [(-5, 5), (5, 10), (10, 15), (15, 25), (25, 60)]
BLAB = ['-5~5', '5~10', '10~15', '15~25', '25~60']

rows = []
for ds in ('STEAD-CH5C', 'INSTANCE'):
    sf = f'eval_all/SNR_{ds}.csv'
    if not os.path.exists(sf): continue
    snr = pd.read_csv(sf)[['trace', 'snr_p', 'snr_s']]
    for name, d in MODELS:
        g = (glob.glob(f'{d}/{ds}_*_picking.csv')
             or glob.glob(f'{d}/STEAD-DET_*_picking.csv'))
        if not g: continue
        x = pd.read_csv(g[0])
        for wave, sk, ek in (('P', 'snr_p', 'p_err'), ('S', 'snr_s', 's_err')):
            if ek not in x.columns: continue
            y = x[['trace', ek]].rename(columns={ek: 'err'})
            y['err'] = pd.to_numeric(y.err, errors='coerce').abs()
            m = snr.merge(y, on='trace', how='inner').dropna(subset=[sk, 'err'])
            if len(m) < 100: continue
            r = dict(dataset=ds, wave=wave, model=name)
            for (lo, hi), lab in zip(BINS, BLAB):
                s = m[(m[sk] >= lo) & (m[sk] < hi)]
                if len(s) < 20:
                    r[f'hit_{lab}'] = np.nan; r[f'mae_{lab}'] = np.nan; continue
                r[f'hit_{lab}'] = float((s.err <= 10).mean() * 100)
                r[f'mae_{lab}'] = float(s.err.mean())
            rows.append(r)

t = pd.DataFrame(rows)
t.to_csv('eval_all/SNR_FULL_TABLE.csv', index=False)
import shutil

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)

dst = os.environ.get('FIG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
if os.path.isdir(dst): shutil.copy('eval_all/SNR_FULL_TABLE.csv', dst)
print(f'  已保存 SNR_FULL_TABLE.csv（{len(t)} 行）')

for ds in ('STEAD-CH5C', 'INSTANCE'):
    sub = t[t.dataset == ds]
    if not len(sub): continue
    print()
    print('#' * 128)
    print(f'### {ds}')
    print('#' * 128)
    for wave in ('P', 'S'):
        w = sub[sub.wave == wave]
        if not len(w): continue
        print()
        print(f'===== {wave} 波 · ≤0.1 s 命中率（%） =====')
        print('   {:24s}'.format('模型') + ''.join(f'{b:>9s}' for b in BLAB))
        for _, r in w.iterrows():
            print('   {:24s}'.format(r.model) + ''.join(
                '{:9.2f}'.format(r[f'hit_{b}']) if r[f'hit_{b}'] == r[f'hit_{b}'] else '{:>9s}'.format('—')
                for b in BLAB))
        print(f'===== {wave} 波 · MAE（采样点，100 点 = 1 s） =====')
        print('   {:24s}'.format('模型') + ''.join(f'{b:>9s}' for b in BLAB))
        for _, r in w.iterrows():
            print('   {:24s}'.format(r.model) + ''.join(
                '{:9.1f}'.format(r[f'mae_{b}']) if r[f'mae_{b}'] == r[f'mae_{b}'] else '{:>9s}'.format('—')
                for b in BLAB))

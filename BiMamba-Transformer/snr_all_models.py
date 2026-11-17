# -*- coding: utf-8 -*-
"""所有模型在 STEAD / INSTANCE 上的 误差 vs SNR（复用已算好的 SNR 表）"""
import pandas as pd, numpy as np, glob, os

MODELS = [('BiMamba 199K+长窗6000', 'w6000',      'eval_all_pm_w6000_all'),
          ('BiMamba 199K（3000）',   'pn197k',     'eval_all_pm_pn197k'),
          ('BiMamba 328K',          'pm322k',     'eval_all_pm_pm322k'),
          ('BiMamba 463K',          'eqt466k',    'eval_all_pm_eqt466k'),
          ('BiMamba 813K',          '813k',       'eval_all_pm_813k'),
          ('BiMamba 199K+增广',      'offaug',     'eval_all_pm_offaug'),
          ('PhaseNet-3000',         'phasenet',   'eval_all_pm_phasenet'),
          ('PhaseNet-6000',         'phasenet6000','eval_all_pm_phasenet6000'),
          ('PhaseMamba-3000',       'phasemamba', 'eval_all_pm_phasemamba'),
          ('PhaseMamba-6000',       'phasemamba6000','eval_all_pm_phasemamba6000'),
          ('EQTransformer-3000',    'eqt',        'eval_all_pm_eqt'),
          ('EQTransformer-6000',    'eqt6000',    'eval_all_pm_eqt6000')]
BINS = [(-5, 5), (5, 10), (10, 15), (15, 25), (25, 60)]

for ds in ('STEAD-CH5C', 'INSTANCE'):
    sf = f'eval_all/SNR_{ds}.csv'
    if not os.path.exists(sf):
        print(f'{ds}: 缺 {sf}'); continue
    snr = pd.read_csv(sf)[['trace', 'snr_p', 'snr_s']]
    print()
    print('#' * 120)
    print(f'### {ds} · 各模型的 ≤0.1s 命中率(%) 与 MAE(点) 随 SNR 的变化')
    print('#' * 120)
    for wave, sk, ek in (('P', 'snr_p', 'p_err'), ('S', 'snr_s', 's_err')):
        print()
        print(f'===== {wave} 波 =====')
        hdr = '   {:24s}'.format('模型')
        for lo, hi in BINS:
            hdr += f' {"≤0.1s "+f"{lo}-{hi}dB":>16s}'
        print(hdr)
        for name, tag, d in MODELS:
            g = glob.glob(f'{d}/{ds}_*_picking.csv') or glob.glob(f'{d}/STEAD-DET_*_picking.csv')
            if not g:
                continue
            x = pd.read_csv(g[0])
            if ek not in x.columns: continue
            y = x[['trace', ek]].rename(columns={ek: 'err'})
            y['err'] = pd.to_numeric(y.err, errors='coerce').abs()
            m = snr.merge(y, on='trace', how='inner').dropna(subset=[sk, 'err'])
            if len(m) < 100: continue
            line = '   {:24s}'.format(name[:24])
            for lo, hi in BINS:
                s = m[(m[sk] >= lo) & (m[sk] < hi)]
                if len(s) < 20:
                    line += f' {"—":>16s}'
                else:
                    line += f' {format((s.err<=10).mean()*100, "6.1f"):>7s}/{format(s.err.mean(),"6.1f"):>8s}'
            print(line)
        if wave == 'P':
            print('   ' + '-' * 114)
            print('   （每格格式：**≤0.1s % / MAE(点)**）')

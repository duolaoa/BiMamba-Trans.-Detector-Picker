# -*- coding: utf-8 -*-
"""窗口级（锚点=元数据 P/S 精确位置）四模型汇总表
   ours = BiMamba-Trans. 199K (W=6000)
   PhaseNet (6000) / PhaseMamba (6000) / EQTransformer (3000，按正文口径)
"""
import glob, os, sys
import numpy as np
import pandas as pd

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


sys.stdout.reconfigure(encoding='utf-8')
H = os.path.dirname(os.path.abspath(__file__))
os.chdir(H)

DATASETS = ['STEAD-DET', 'INSTANCE', 'EQS-D-0dB', 'EQS-D-imp', 'CREW']

MODELS = [
    ('BiMamba-Trans.', 6000, ['eval_all_pm_w6000_all/{ds}_*_picking.csv']),
    ('PhaseNet',       6000, ['eval_all_pm_phasenet6000/{ds}_*_picking.csv']),
    ('PhaseMamba',     6000, ['eval_all_pm_phasemamba6000/{ds}_*_picking.csv']),
    ('EQTransformer',  3000, ['eval_all_pm_eqt/{ds}_*_picking.csv']),
]

def load(patterns, ds, col):
    pats = list(patterns)
    if ds == 'STEAD-DET':
        pats.append('eval_all_pm_w6000_all/STEAD-CH5C_*_picking.csv')
    for pat in pats:
        g = sorted(glob.glob(pat.format(ds=ds)))
        g = [f for f in g if not f.endswith('summary_picking.csv')]
        if not g:
            continue
        v = []
        for f in g:
            d = pd.read_csv(f)
            if col in d.columns:
                a = pd.to_numeric(d[col], errors='coerce').dropna().values.astype(float)
                if len(a):
                    v.append(a)
        if v:
            v = np.concatenate(v)
            if len(v) >= 50:
                return v, os.path.basename(g[0])
    return None, None

rows = []
for ds in DATASETS:
    for name, W, pats in MODELS:
        p, src = load(pats, ds, 'p_err')
        if p is None:
            rows.append(dict(dataset=ds, model=name, W=W, P_n=0))
            continue
        s, _ = load(pats, ds, 's_err')
        r = dict(dataset=ds, model=name, W=W,
                 P_n=len(p),
                 P_le10=float((np.abs(p) <= 10).mean() * 100),
                 P_le50=float((np.abs(p) <= 50).mean() * 100),
                 P_MAE=float(np.abs(p).mean() / 100),
                 P_med=float(np.median(p) / 100),
                 P_p95=float(np.percentile(np.abs(p), 95) / 100))
        if s is not None and len(s) >= 50:
            r.update(S_n=len(s),
                     S_le10=float((np.abs(s) <= 10).mean() * 100),
                     S_le50=float((np.abs(s) <= 50).mean() * 100),
                     S_MAE=float(np.abs(s).mean() / 100),
                     S_med=float(np.median(s) / 100))
        rows.append(r)

t = pd.DataFrame(rows)

print('=' * 118)
print('窗口级结果（锚点 = 元数据 P/S 精确位置；P 窗 [p-500,p+2500)，S 窗独立）')
print('=' * 118)
for ds in DATASETS:
    sub = t[t.dataset == ds]
    print(f'\n■ {ds}')
    print('  {:17s} {:>5s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} | {:>6s} {:>8s} {:>8s} {:>8s}'.format(
        '模型', 'W', 'P_n', 'P≤0.1s', 'P≤0.5s', 'P-MAE', 'P-Med', 'P-P95', 'S_n', 'S≤0.5s', 'S-MAE', 'S-Med'))
    for _, r in sub.iterrows():
        if r.P_n == 0:
            print(f'  {r.model:17s} {r.W:5d}   ——  无数据')
            continue
        s_n = int(r.S_n) if 'S_n' in r and pd.notna(r.get('S_n')) else 0
        s50 = f"{r.S_le50:7.2f}%" if s_n else '      —'
        smae = f"{r.S_MAE:7.3f}" if s_n else '      —'
        smed = f"{r.S_med:7.3f}" if s_n else '      —'
        print('  {:17s} {:5d} {:6d} {:7.2f}% {:7.2f}% {:8.3f} {:8.3f} {:8.3f} | {:6d} {:>8s} {:>8s} {:>8s}'.format(
            r.model, r.W, int(r.P_n), r.P_le10, r.P_le50, r.P_MAE, r.P_med, r.P_p95,
            s_n, s50, smae, smed))

t.to_csv('eval_all/WINDOW_LEVEL_4MODELS_199K.csv', index=False)
print('\n  saved eval_all/WINDOW_LEVEL_4MODELS_199K.csv')

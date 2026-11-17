# -*- coding: utf-8 -*-
"""SNR 柱状图（高对比配色 + 图例置于坐标区外，图顶部共享）"""
import pandas as pd, numpy as np, glob, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 高对比 Set1 配色；本文模型用最醒目的红
MODELS = [('BiMamba (ours, W=6000)', 'eval_all_pm_w6000_all',      '#E41A1C'),  # 红
          ('PhaseNet-3000',          'eval_all_pm_phasenet',       '#377EB8'),  # 蓝
          ('PhaseMamba-6000',        'eval_all_pm_phasemamba6000',  '#4DAF4A'),  # 绿
          ('EQTransformer-3000',     'eval_all_pm_eqt',            '#984EA3')]  # 紫
BINS = [(-5, 5), (5, 10), (10, 15), (15, 25), (25, 60)]
LABELS = ['−5~5', '5~10', '10~15', '15~25', '25~60']

def collect(ds, wave, d):
    sf = f'eval_all/SNR_{ds}.csv'
    if not os.path.exists(sf): return None
    snr = pd.read_csv(sf)[['trace', 'snr_p', 'snr_s']]
    sk = 'snr_p' if wave == 'P' else 'snr_s'
    ek = 'p_err' if wave == 'P' else 's_err'
    g = (glob.glob(f'{d}/{ds}_*_picking.csv')
         or glob.glob(f'{d}/STEAD-DET_*_picking.csv'))
    if not g: return None
    x = pd.read_csv(g[0])
    if ek not in x.columns: return None
    y = x[['trace', ek]].rename(columns={ek: 'err'})
    y['err'] = pd.to_numeric(y.err, errors='coerce').abs()
    m = snr.merge(y, on='trace', how='inner').dropna(subset=[sk, 'err'])
    hit, mae = [], []
    for lo, hi in BINS:
        s = m[(m[sk] >= lo) & (m[sk] < hi)]
        if len(s) < 20:
            hit.append(np.nan); mae.append(np.nan); continue
        hit.append((s.err <= 10).mean() * 100)
        mae.append(s.err.mean())
    return np.array(hit), np.array(mae)

for ds, fname in (('STEAD-CH5C', 'snr_bars_stead.png'),
                  ('INSTANCE',   'snr_bars_instance.png')):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    for ax_idx, wave in enumerate(('P', 'S')):
        ax = axes[ax_idx]; ax2 = ax.twinx()
        nb = len(MODELS); w = 0.8 / nb
        xpos = np.arange(len(BINS))
        mae_max = 1
        for k, (name, d, color) in enumerate(MODELS):
            r = collect(ds, wave, d)
            if r is None: continue
            hit, mae = r
            mae_max = max(mae_max, np.nanmax(mae))
            off = (k - (nb - 1) / 2) * w
            ax.bar(xpos + off, hit, width=w * 0.9, color=color, alpha=0.9,
                   edgecolor='black', lw=0.5, zorder=2)
            ax2.plot(xpos + off, mae, color=color, marker='o', ms=4.5, lw=1.8,
                     ls='--', zorder=3)
        ax.set_xticks(xpos); ax.set_xticklabels(LABELS, fontsize=10.5)
        ax.set_xlabel('SNR (dB)', fontsize=11.5)
        ax.set_ylabel('Hit rate  $\\leq$ 0.1 s (%)', fontsize=11.5)
        ax2.set_ylabel('MAE (samples)  —  dashed', fontsize=11.5)
        ax.set_ylim(0, 105); ax2.set_ylim(0, mae_max * 1.18)
        ax.set_title(f'{ds} — {wave} wave', fontsize=12.5, pad=8)
        ax.grid(alpha=0.25, axis='y', zorder=0)
        for sp in ('top',):
            ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)

    # 图例放在坐标区之外（图底部共享，一行四个）
    handles = [Patch(facecolor=c, edgecolor='black', lw=0.5, label=n)
               for n, _, c in MODELS]
    handles += [Line2D([0], [0], color='k', ls='--', marker='o', ms=4.5,
                       lw=1.6, label='MAE (right axis)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = f'eval_all/figs/{fname}'
    fig.savefig(out, dpi=175, bbox_inches='tight'); plt.close(fig)
    print(f'  saved {out}')

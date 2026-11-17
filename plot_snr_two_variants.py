# -*- coding: utf-8 -*-
"""两个版本：
   v1 = 柱彩色 + 折线彩色但**标记点黑色**（区分线与柱）
   v2 = 柱彩色 + **折线全黑**，用不同线型/标记区分模型
   均加总标题，图例置于坐标区外。
"""
import pandas as pd, numpy as np, glob, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


MODELS = [('BiMamba (ours)', 'eval_all_pm_w6000_all',       '#E41A1C', '-',  'o'),
          ('PhaseNet',          'eval_all_pm_phasenet',        '#377EB8', '--', 's'),
          ('PhaseMamba',        'eval_all_pm_phasemamba6000',  '#4DAF4A', '-.', '^'),
          ('EQTransformer',     'eval_all_pm_eqt',             '#984EA3', ':',  'D')]
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

TITLES = {'STEAD-CH5C': 'STEAD dataset (10,000 events), tolerance 0.1 s',
          'INSTANCE': 'INSTANCE dataset (6,063 events with S labels), tolerance 0.1 s'}

def draw(ds, variant, fname):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    for ax_idx, wave in enumerate(('P', 'S')):
        ax = axes[ax_idx]; ax2 = ax.twinx()
        nb = len(MODELS); w = 0.8 / nb
        xpos = np.arange(len(BINS))
        mae_max = 1
        for k, (name, d, color, ls, mk) in enumerate(MODELS):
            r = collect(ds, wave, d)
            if r is None: continue
            hit, mae = r
            mae_max = max(mae_max, np.nanmax(mae))
            off = (k - (nb - 1) / 2) * w
            ax.bar(xpos + off, hit, width=w * 0.9, color=color, alpha=0.9,
                   edgecolor='black', lw=0.5, zorder=2)
            if variant == 1:   # 彩色线 + 黑色/白色标记
                ax2.plot(xpos, mae, color=color, lw=2.0, ls='-',
                         marker=mk, ms=10.5, mfc='black', mec='white', mew=1.8, zorder=5)
            else:              # 全黑线 + 不同线型/标记
                ax2.plot(xpos, mae, color='black', lw=1.9, ls=ls,
                         marker=mk, ms=10.5, mfc='white', mec='black', mew=1.8, zorder=5)
        ax.set_xticks(xpos); ax.set_xticklabels(LABELS, fontsize=10.5)
        ax.set_xlabel('SNR (dB)', fontsize=11.5)
        ax.set_ylabel('Hit rate  $\\leq$ 0.1 s (%)', fontsize=11.5)
        ax2.set_ylabel('MAE (samples)  —  lines', fontsize=11.5)
        ax.set_ylim(0, 105); ax2.set_ylim(0, mae_max * 1.18)
        ax.set_title(f'{wave} wave', fontsize=13, pad=8)
        ax.grid(alpha=0.25, axis='y', zorder=0)
        ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)

    # 图例：柱（颜色）+ 线（每模型一种线型/标记）
    h = [Patch(facecolor=c, edgecolor='black', lw=0.5, label=n)
         for n, _, c, _, _ in MODELS]
    if variant == 1:
        h += [Line2D([0], [0], color=c, lw=1.8, marker=mk, ms=9,
                     mfc='black', mec='white', mew=1.1, label=f'{n} — MAE')
              for n, _, c, _, mk in MODELS]
    else:
        h += [Line2D([0], [0], color='black', lw=1.6, ls=ls, marker=mk, ms=9,
                     mfc='white', mec='black', label=f'{n} — MAE')
              for n, _, _, ls, mk in MODELS]
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06),
               title='Bars: hit rate <= 0.1 s (left)  |  Lines: MAE (right)',
               title_fontsize=9.5)
    fig.suptitle(TITLES[ds], fontsize=14, y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f'eval_all/figs/{fname}', dpi=175, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {fname}')

for ds, tag in (('STEAD-CH5C', 'stead'), ('INSTANCE', 'instance')):
    draw(ds, 1, f'snr_bars_{tag}_v1.png')
    draw(ds, 2, f'snr_bars_{tag}_v2.png')

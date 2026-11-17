# -*- coding: utf-8 -*-
"""误差分布直方图（12 幅）：每个「模型 × 数据集」一幅。
   · **有符号残差** (t̂ − t)，不用绝对值 → 中间为 0 ✓
   · 分箱 0.1 s，范围 −1 ~ +1 s；|误差| > 1 s 归入两端箱
   · 每幅图含 P 与 S 两个子图（EQS-D 无 S → 只画 P）
"""
import os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = r"E:\Project\DPCNN\microseis_aug\STEAD_data\1Dsignal\eval_figs"

DATASETS = {
    'STEAD-DET': dict(label='STEAD (10,000 events)',
        files={'BiMamba-Trans.': 'eval_all_pm_w6000_all/STEAD-CH5C_*_picking.csv',
               'PhaseNet':       'eval_all_pm_phasenet6000/STEAD-DET_*_picking.csv',
               'EQTransformer':  'eval_all_pm_eqt/STEAD-DET_*_picking.csv',
               'PhaseMamba':     'eval_all_pm_phasemamba6000/STEAD-CH5C_*_picking.csv'}),
    'INSTANCE': dict(label='INSTANCE (10,000 events)',
        files={'BiMamba-Trans.': 'eval_all_pm_w6000_all/INSTANCE_*_picking.csv',
               'PhaseNet':       'eval_all_pm_phasenet6000/INSTANCE_*_picking.csv',
               'EQTransformer':  'eval_all_pm_eqt/INSTANCE_*_picking.csv',
               'PhaseMamba':     'eval_all_pm_phasemamba6000/INSTANCE_*_picking.csv'}),
    'EQS-D-0dB': dict(label='EQS-Denoiser 0 dB (10,000 events)',
        files={'BiMamba-Trans.': 'eval_all_pm_w6000_all/EQS-D-0dB_*_picking.csv',
               'PhaseNet':       'eval_all_pm_phasenet6000/EQS-D-0dB_*_picking.csv',
               'EQTransformer':  'eval_all_pm_eqt/EQS-D-0dB_*_picking.csv',
               'PhaseMamba':     'eval_all_pm_phasemamba6000/EQS-D-0dB_*_picking.csv'}),
}
COLORS = {'BiMamba-Trans.': '#2ca02c', 'PhaseNet': '#ff7f0e',
          'EQTransformer': '#d62728', 'PhaseMamba': '#9467bd'}

EDGES = np.arange(-1.0, 1.0 + 0.1, 0.1)     # -1.0 ... +1.0

def load_signed(pat, col):
    g = sorted(glob.glob(pat))
    if not g:
        return None
    xs = []
    for f in g:
        d = pd.read_csv(f)
        if col not in d.columns:
            continue
        v = pd.to_numeric(d[col], errors='coerce').values / 100.0   # 秒，保留符号
        v = v[np.isfinite(v)]
        xs.append(v)
    return np.concatenate(xs) if xs else None

def draw(ds, cfg, model):
    pat = cfg['files'][model]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    any_plotted = False
    for ax, wave, col in ((axes[0], 'P', 'p_err'), (axes[1], 'S', 's_err')):
        v = load_signed(pat, col)
        if v is None or not len(v):
            ax.text(0.5, 0.5, f'no {wave}-wave annotation', ha='center', va='center',
                    fontsize=11, transform=ax.transAxes, color='#777777')
            ax.set_title(f'({wave[0].lower()}) {wave} wave', fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        any_plotted = True
        lo = np.clip(v, -1.0, 1.0)
        # 把 |e|>1 放入两端箱
        lo = np.where(v < -1.0, -1.0 + 1e-9, lo)
        lo = np.where(v > 1.0, 1.0 - 1e-9, lo)
        h, _ = np.histogram(lo, bins=EDGES)
        xs = EDGES[:-1] + 0.05
        ax.bar(xs, h, width=0.092, color=COLORS[model], edgecolor='black', lw=0.45, alpha=0.9)
        ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.axvline(v.mean(), color='red', lw=1.4, ls='-', alpha=0.9,
                   label=f'mean = {v.mean():+.3f} s')
        ax.axvline(np.median(v), color='blue', lw=1.2, ls=':', alpha=0.9,
                   label=f'median = {np.median(v):+.3f} s')
        ax.set_ylim(0, h.max() * 1.18)
        ax.set_xlim(-1.08, 1.08)
        ax.set_xticks(np.arange(-1.0, 1.01, 0.2))
        ax.set_xlabel('Signed error $\\hat{t}-t$ (s;  0.1 s bins;  |e|>1 pooled at ±1)', fontsize=9)
        ax.set_ylabel('Number of picks (count)', fontsize=9)
        ax.set_title(f'({wave[0].lower()}) {wave} wave  —  n = {len(v):,},  '
                     f'|e|≤0.1 s = {(np.abs(v)<=0.1).mean()*100:.1f}%', fontsize=10.5)
        ax.grid(alpha=0.25, which='both')
        ax.legend(fontsize=8.4, loc='upper right', framealpha=0.9)
        ax.tick_params(labelsize=8.5)
    fig.suptitle(f'{cfg["label"]}  —  {model}', fontsize=12.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUTDIR, f'errdist_{ds.replace("-", "")}_{model.replace(".", "")}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)
    return any_plotted

n = 0
for ds, cfg in DATASETS.items():
    for model in cfg['files']:
        if draw(ds, cfg, model):
            n += 1
print(f'\n  共生成 {n} 幅（模型×数据集）')

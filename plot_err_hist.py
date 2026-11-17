# -*- coding: utf-8 -*-
"""每个数据集一张 P/S 误差分布直方图（四模型对比）。
   · 分箱宽度 0.1 s
   · 误差 > 1 s 的全部归入最后一个箱（标为 ">1"）
   · 纵轴为"占该模型有效样本的百分比"，便于跨模型比较
   · 数据：窗口级 picking CSV（真值摆窗、无门控），与示例图的误差定义一致
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
    'STEAD-DET': dict(
        label='STEAD (10,000 events)',
        files=[('BiMamba-Trans.', 'eval_all_pm_w6000_all/STEAD-CH5C_*_picking.csv', '#2ca02c'),
               ('PhaseNet',       'eval_all_pm_phasenet6000/STEAD-DET_*_picking.csv', '#ff7f0e'),
               ('EQTransformer',  'eval_all_pm_eqt/STEAD-DET_*_picking.csv', '#d62728'),
               ('PhaseMamba',     'eval_all_pm_phasemamba6000/STEAD-CH5C_*_picking.csv', '#9467bd')]),
    'INSTANCE': dict(
        label='INSTANCE (10,000 events)',
        files=[('BiMamba-Trans.', 'eval_all_pm_w6000_all/INSTANCE_*_picking.csv', '#2ca02c'),
               ('PhaseNet',       'eval_all_pm_phasenet6000/INSTANCE_*_picking.csv', '#ff7f0e'),
               ('EQTransformer',  'eval_all_pm_eqt/INSTANCE_*_picking.csv', '#d62728'),
               ('PhaseMamba',     'eval_all_pm_phasemamba6000/INSTANCE_*_picking.csv', '#9467bd')]),
    'EQS-D-0dB': dict(
        label='EQS-Denoiser 0 dB (10,000 events)',
        files=[('BiMamba-Trans.', 'eval_all_pm_w6000_all/EQS-D-0dB_*_picking.csv', '#2ca02c'),
               ('PhaseNet',       'eval_all_pm_phasenet6000/EQS-D-0dB_*_picking.csv', '#ff7f0e'),
               ('EQTransformer',  'eval_all_pm_eqt/EQS-D-0dB_*_picking.csv', '#d62728'),
               ('PhaseMamba',     'eval_all_pm_phasemamba6000/EQS-D-0dB_*_picking.csv', '#9467bd')]),
}

BIN = 0.1                      # 0.1 s
EDGES = np.arange(0, 1.0 + BIN, BIN)   # 0, 0.1, ..., 1.0  （>1 归入最后）

def load_err(pat, col):
    g = sorted(glob.glob(pat))
    if not g:
        return None
    xs = []
    for f in g:
        d = pd.read_csv(f)
        if col not in d.columns:
            continue
        v = pd.to_numeric(d[col], errors='coerce').abs().values / 100.0   # 转秒
        v = v[np.isfinite(v)]
        xs.append(v)
    return np.concatenate(xs) if xs else None

def hist_pct(v):
    """把 >1 s 的并入最后一箱，返回百分比。"""
    v = np.clip(v, 0, None)
    h, _ = np.histogram(np.minimum(v, 1.0 - 1e-9), bins=EDGES)
    return h / max(len(v), 1) * 100.0

def draw(ds, cfg, wave):
    col = 'p_err' if wave == 'P' else 's_err'
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    width = 0.2
    n = len(cfg['files'])
    for i, (lab, pat, color) in enumerate(cfg['files']):
        v = load_err(pat, col)
        if v is None or not len(v):
            print(f'    {ds}/{wave}  {lab}: 无数据'); continue
        pct = hist_pct(v)
        x = np.arange(len(pct)) + (i - (n - 1) / 2) * width
        ax.bar(x, pct, width * 0.92, color=color, edgecolor='black', lw=0.5,
               alpha=0.88, label=f'{lab}  (n={len(v):,}, '
                                 f'≤0.1s={(v<=0.1).mean()*100:.1f}%)')
    # 在 ">1" 箱上方标注各自的比例
    for i, (lab, pat, color) in enumerate(cfg['files']):
        v = load_err(pat, col)
        if v is None or not len(v):
            continue
        p_over = (v > 1.0).mean() * 100
        if p_over <= 0:
            continue
        x_last = (len(EDGES) - 1) + (i - (len(cfg['files']) - 1) / 2) * width
        ax.text(x_last, max(p_over, 0.008) * 1.8, f'{p_over:.2f}%',
                rotation=90, fontsize=7.5, color=color, ha='center', va='bottom',
                fontweight='bold')
    ticks = [f'{e:.1f}' for e in EDGES[:-1]] + ['>1']
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks, fontsize=8.5)
    ax.set_xlabel('Picking error (s, bin = 0.1 s; errors > 1 s pooled)', fontsize=10)
    ax.set_ylabel('Fraction of picks (%)  (log scale)', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(0.005, 120)
    ax.set_title(f"{cfg['label']} — {wave}-wave error distribution", fontsize=11)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8.2, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(OUTDIR, f'err_hist_{ds.replace("-", "")}_{wave}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)

for ds, cfg in DATASETS.items():
    for wave in ('P', 'S'):
        draw(ds, cfg, wave)

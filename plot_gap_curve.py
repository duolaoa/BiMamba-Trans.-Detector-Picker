# -*- coding: utf-8 -*-
"""S−P 时差 → S 拾取命中率/中位误差：3,000 vs 6,000 窗（CREW），
并叠加 STEAD / INSTANCE 的时差区间。"""
import os
import pandas as pd, numpy as np, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)

def load(d, ds):
    g = glob.glob(f'{d}/{ds}_*_picking.csv')
    if not g: return None
    x = pd.read_csv(g[0])
    for c in ('s_true', 'p_true', 's_err'):
        if c not in x.columns: return None
    gap = (x.s_true - x.p_true)
    se = pd.to_numeric(x.s_err, errors='coerce').abs()
    m = gap.notna() & se.notna() & (gap > 0)
    return gap[m].values / 100.0, se[m].values / 100.0     # 转为秒

CURVES = [('W = 3,000', 'eval_all_pm_pn197k', 25.0, '#1f77b4', 'o'),
          ('W = 6,000', 'eval_all_pm_w6000_all', 55.0, '#2ca02c', 's')]
BINS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 55, 60, 75, 90, 120, 150, 200, 300])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
for name, d, fwd, color, mk in CURVES:
    r = load(d, 'CREW')
    if r is None:
        print(f'  {name}: CREW 无数据'); continue
    gap, se = r
    ctr, hit, med = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (gap >= lo) & (gap < hi)
        if s.sum() < 15: continue
        ctr.append((lo + hi) / 2)
        hit.append((se[s] <= 0.5).mean() * 100)
        med.append(np.median(se[s]))
    axes[0].plot(ctr, hit, marker=mk, ms=5, lw=1.8, color=color, label=name)
    axes[1].plot(ctr, med, marker=mk, ms=5, lw=1.8, color=color, label=name)
    axes[0].axvline(fwd, color=color, ls='--', lw=1.2, alpha=0.6)
    axes[1].axvline(fwd, color=color, ls='--', lw=1.2, alpha=0.6)

for ax, ylab, ttl, ylim in (
        (axes[0], 'S ≤ 0.5 s hit rate (%)', '(a) Hit rate vs S−P time difference', (-2, 100)),
        (axes[1], 'S median |error| (s)',    '(b) Median error vs S−P time difference', None)):
    ax.set_xlabel('S − P time difference (s)', fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(ttl, fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 200)
    if ylim: ax.set_ylim(*ylim)
axes[1].set_yscale('log')

fig.suptitle('CREW (300 s): single-window S picking vs S−P time difference', fontsize=13)
fig.tight_layout()
fig.savefig('eval_all/figs/gap_vs_error_crew.png', dpi=170)
plt.close(fig)
print('  已保存 eval_all/figs/gap_vs_error_crew.png')

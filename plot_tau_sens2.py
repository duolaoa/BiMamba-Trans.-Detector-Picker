# -*- coding: utf-8 -*-
"""(a) 检测 Pr–Re 曲线；(b) 检测 F1 与端到端 P F1@0.5s 随门控阈值的变化（跨域三个数据集）。"""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = pd.read_csv('eval_all/TAU_SWEEP_Q.csv')
OUT = os.environ.get('FIG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
STYLE = {'STEAD-DET': ('#C0392B', 'o', '-'),
         'INSTANCE':  ('#2980B9', 's', '--'),
         'EQS-D-0dB': ('#27AE60', '^', '-.'),
         'EQS-D-imp': ('#8E44AD', 'D', ':')}

# ---------- (a) Pr–Re ----------
fig, ax = plt.subplots(figsize=(6.6, 4.7))
for ds, (c, mk, ls) in STYLE.items():
    s = t[t.dataset == ds].sort_values('Pr')
    # 只取 F1 不低于其最大值 5% 以内的区间，避免极端端点的折返
    fmax = s.F1.max()
    keep = s[(s.F1 >= fmax - 0.05) | (s.Re > 0.3)]
    ax.plot(keep.Pr, keep.Re, color=c, ls=ls, lw=1.8, marker=mk, ms=6,
            mfc='white', mec=c, mew=1.3, label=ds)
    # 标出 F1 最优的操作点
    b = s.loc[s.F1.idxmax()]
    ax.plot(b.Pr, b.Re, marker=mk, ms=11, mfc=c, mec='black', mew=1.4, zorder=5)
ax.set_xlabel('Precision', fontsize=10.5)
ax.set_ylabel('Recall', fontsize=10.5)
ax.set_xlim(0.75, 1.005); ax.set_ylim(0.25, 1.01)
ax.grid(alpha=0.3, ls=':')
ax.legend(fontsize=9, loc='lower left', title='filled marker = F1-optimal $\\tau$',
          title_fontsize=8)
ax.set_title('(a) Detection precision\u2013recall trade-off', fontsize=11.5, fontweight='bold')
fig.tight_layout()
p = os.path.join(OUT, 'fig7a_tau_pr_curve.png')
fig.savefig(p, dpi=240, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('saved', p)

# ---------- (b) F1 & P_F1@0.5 vs noise pass rate ----------
fig, ax = plt.subplots(figsize=(6.6, 4.7))
CROSS = ['INSTANCE', 'EQS-D-0dB', 'EQS-D-imp']
for ds in CROSS:
    c, mk, ls = STYLE[ds]
    s = t[t.dataset == ds].sort_values('noise_pass', ascending=False)
    x = s.noise_pass * 100
    ax.plot(x, s.F1, color=c, ls=ls, lw=2.0, marker=mk, ms=6.5, mfc=c,
            mec='black', mew=0.9, label=f'{ds} \u2014 detection F1')
    ax.plot(x, s.P_F1_05, color=c, ls=ls, lw=1.5, marker=mk, ms=5.5, mfc='white',
            mec=c, mew=1.3, label=f'{ds} \u2014 P F1 @ 0.5 s')
ax.set_xscale('log')
ax.set_xlabel('Noise pass rate (%)  (higher threshold $\\rightarrow$)', fontsize=10.5)
ax.set_ylabel('F1', fontsize=10.5)
ax.set_ylim(0.72, 1.0)
ax.grid(alpha=0.3, which='both', ls=':')
ax.legend(fontsize=7.6, ncol=2, loc='lower right', framealpha=0.95)
ax.set_title('(b) Cross-domain threshold sensitivity', fontsize=11.5, fontweight='bold')
fig.tight_layout()
p = os.path.join(OUT, 'fig7b_tau_sensitivity.png')
fig.savefig(p, dpi=240, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('saved', p)

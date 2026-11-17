# -*- coding: utf-8 -*-
"""检测示例②（改进版）：一条**被 PhaseNet / PhaseMamba 误判为事件**的 INSTANCE 噪声记录。
   记录 20100303T101636.IV.PTRJ..HH
   本文 s = 4.6e-6（阈值 6.2e-4）→ 正确拒绝
   PhaseNet 0.7318（阈值 0.0255）、PhaseMamba 0.5329（阈值 0.0186）→ 虚警
   下图用**对数轴 + 各自阈值虚线**，使"是否越阈"一目了然。
"""
import os, numpy as np, torch, matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_evsplit_models as E

OUT = os.path.join(os.environ.get('FIG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')), 'det_example_noise_fp.png')
TARGET = '20100303T101636.IV.PTRJ'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = [
 ('BiMamba-Trans.', 'BiMamba-Detector', 'best_detection_model_3C_det_pn197k.pth', '#2ca02c', 6.18e-4),
 ('PhaseNet',            'PhaseNet',         'best_phasenet_baseline_6000.pth',        '#ff7f0e', 2.55e-2),
 ('PhaseMamba',          'PhaseMamba',       'best_phasemamba_evsplit__phasemamba6000.pth', '#9467bd', 1.86e-2),
 ('EQTransformer',       'EQTransformer',    'best_eqtransformer_baseline_6000.pth',   '#d62728', 2.91e-3),
]
WRAP = {'BiMamba-Detector': E.BiMambaDetector, 'PhaseNet': E.PhaseNetWrap,
        'PhaseMamba': E.PhaseMambaWrap, 'EQTransformer': E.EQTWrap}

rec = None
for nm, x, p, s in E.DATASETS['INSTANCE']('noise', 0):
    if TARGET in str(nm):
        rec = (str(nm), x); break
assert rec is not None, 'target noise record not found'
name, x = rec
T = x.shape[0]
print(f'  记录 {name}  T={T}')

def load(cls, ck):
    os.environ['EVAL_WINDOW'] = '6000'
    E.CKPT['picker'] = os.path.abspath(ck)
    m = WRAP[cls](dev)
    if not hasattr(m, 'prep'):
        m.prep = E.preprocess_40
    return m

def curve(m, cls, x):
    W = getattr(m, 'window', 6000)
    st = max(0, (x.shape[0] - W) // 2)
    win = x[st:st + W]
    prep = m.prep if hasattr(m, 'prep') else E.preprocess_40
    out = m.score(m.to_input(np.stack([prep(win)])))
    if cls == 'BiMamba-Detector':
        c = np.maximum(np.asarray(out['det'])[0],
                       np.asarray(out.get('detS', out['det']))[0]).astype(np.float64)
    elif cls == 'PhaseNet':
        c = (1.0 - np.asarray(out['noise'])[0]).astype(np.float64)   # 无存在头 → 1-noise
    elif 'det' in out:
        c = np.asarray(out['det'])[0].astype(np.float64)
    else:
        c = np.asarray(out['P'])[0].astype(np.float64)
    xs = np.linspace(0, 1, len(c))
    return np.interp(np.linspace(0, 1, W), xs, c), st, W

ms = {cls: load(cls, ck) for _, cls, ck, _, _ in MODELS}

fig = plt.figure(figsize=(13.0, 9.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.24)

# 上：波形
ax = fig.add_subplot(gs[0])
t = np.arange(T) / 100.0
sc = np.std(x[:, 0]) + 1e-9
for i, (ci, lab, col) in enumerate([(2, 'Z', '#1f77b4'), (1, 'N', '#2ca02c'), (0, 'E', '#d62728')]):
    ax.plot(t, x[:, ci] / sc + (2 - i) * 6.5, color=col, lw=0.6, label=lab)
ax.set_yticks([13, 6.5, 0]); ax.set_yticklabels(['Z', 'N', 'E'])
W_ = 6000
ST_ = max(0, (T - W_) // 2)
ax.set_xlim(ST_ / 100, (ST_ + W_) / 100)
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('Amplitude (normalised, offset)', fontsize=10)
ax.set_title(f'INSTANCE noise record {name}  (T = {T/100:.0f} s) — no earthquake', fontsize=11)
ax.legend(fontsize=8.5, loc='upper right', ncol=3, framealpha=0.9)
ax.grid(alpha=0.2)

# 下：存在概率（对数轴）+ 各自阈值
ax = fig.add_subplot(gs[1])
rows = []
for lab, cls, ck, color, tau in MODELS:
    try:
        c, st, W = curve(ms[cls], cls, x)
    except Exception as e:
        print(f'  {lab}: {e}'); continue
    tt = (st + np.arange(W)) / 100.0
    cc = np.clip(c, 1e-9, None)
    ax.semilogy(tt, cc, color=color, lw=1.3, label=f'{lab} (s = {c.max():.2g})')
    ax.axhline(tau, color=color, ls='--', lw=1.2, alpha=0.85)
    rows.append((lab, color, float(c.max()), tau))

# 阈值标签放在右侧
_fac = [1.7, 3.2, 1.7, 3.2]
for i, (lab, color, smax, tau) in enumerate(rows):
    xx = (ST_ + W_) / 100 * 0.995 - i * 6.0
    ax.text(xx, tau * _fac[i % len(_fac)], f'τ = {tau:.2g}', color=color,
            fontsize=8.8, ha='right', va='bottom', fontweight='bold')

ax.set_xlim(ST_ / 100, (ST_ + W_) / 100)
ax.set_ylim(1e-9, 2)
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('Phase-presence probability (log scale)', fontsize=10)
ax.set_title('Presence response with each model\'s operating threshold (dashed). '
             'Curves above their threshold are false alarms.', fontsize=11)
ax.grid(alpha=0.25, which='both')
ax.legend(fontsize=8.2, loc='lower left', ncol=2, framealpha=0.9)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('\n  记录分数 vs 阈值：')
for lab, color, smax, tau in rows:
    print(f'    {lab:22s} s={smax:.4g}  τ={tau:.3g}  '
          f'{"虚警 ✗" if smax >= tau else "正确拒绝 ✓"}  (s/τ = {smax/tau:.3g})')
print('  图已保存:', OUT)

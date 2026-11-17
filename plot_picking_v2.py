# -*- coding: utf-8 -*-
"""拾取示例图 v2（新建文件，不改原图）：
   上：三分量波形；下：四模型的 P/S 响应曲线。
   关键修正：本文模型的原始输出是 **logits**，先沿时间轴做 softmax 变成概率密度，
   再与基线（本身即概率）一样按自身峰值归一化 → 四条曲线形状可直接比较。
   另：把 ΔP / ΔS（相对真值的偏差）标在图例里。
"""
import os, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_evsplit_models as E

OUT = os.path.join(os.environ.get('FIG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')), 'picking_examples_stead_v2.png')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = [
 ('BiMamba (ours, W=6000)', 'BiMamba-Picker', 'best_lightweight_model_3C_pn197k_w6000v3.pth', '#2ca02c'),
 ('PhaseNet (W=6000)',      'PhaseNet',       'best_phasenet_baseline_6000.pth',             '#ff7f0e'),
 ('PhaseMamba (W=6000)',    'PhaseMamba',     'best_phasemamba_evsplit__phasemamba6000.pth', '#9467bd'),
 ('EQTransformer (W=6000)', 'EQTransformer',  'best_eqtransformer_baseline_6000.pth',        '#d62728'),
]
WRAP = {'BiMamba-Picker': E.BiMambaPicker, 'PhaseNet': E.PhaseNetWrap,
        'PhaseMamba': E.PhaseMambaWrap, 'EQTransformer': E.EQTWrap}

# ---------- 选一条 P/S 都清晰、P 位置适中的记录 ----------
best = None
for nm, x, p, s in E.DATASETS['STEAD-DET']('events', 0):
    if not (np.isfinite(p) and np.isfinite(s)):
        continue
    if 1300 <= p <= 2100 and 900 <= s - p <= 2000:
        best = (str(nm), x, float(p), float(s)); break
if best is None:
    for nm, x, p, s in E.DATASETS['STEAD-DET']('events', 0):
        best = (str(nm), x, float(p), float(s)); break
name, x, p_true, s_true = best
T = x.shape[0]
print(f'  记录 {name}  T={T}  P={p_true:.0f}  S={s_true:.0f}')

def softmax_t(a):
    z = a.astype(np.float64)
    z = np.exp(z - z.max())
    return z / (z.sum() + 1e-12)

def forward(ck, cls):
    os.environ['EVAL_WINDOW'] = '6000'
    E.CKPT['picker'] = os.path.abspath(ck)
    m = WRAP[cls](dev)
    if not hasattr(m, 'prep'):
        m.prep = E.preprocess_40
    W = m.window
    st = int(np.clip(p_true - W // 2, 0, max(0, T - W)))
    win = x[st:st + W]
    prep = m.prep if hasattr(m, 'prep') else E.preprocess_40
    out = m.score(m.to_input(np.stack([prep(win)])))

    def curve(key):
        c = np.asarray(out[key])[0].astype(np.float64)
        # 本文模型输出为 logits → 先 softmax 成密度；基线本身已是概率，softmax 对单调无关，保险起见只对 ours 做
        if cls == 'BiMamba-Picker':
            c = softmax_t(c)
        xs = np.linspace(0, 1, len(c))
        return np.interp(np.linspace(0, 1, W), xs, c)

    return st, W, curve('P'), curve('S'), out

fig, axes = plt.subplots(2, 1, figsize=(13, 8.6),
                         gridspec_kw={'height_ratios': [1.05, 1.25]}, sharex=True)

# ---------------- 上：波形 ----------------
ax = axes[0]
t = np.arange(T) / 100.0
scale = np.std(x[:, 0]) + 1e-9
for i, (ci, lab, col) in enumerate([(2, 'Z', '#1f77b4'), (1, 'N', '#2ca02c'), (0, 'E', '#d62728')]):
    ax.plot(t, x[:, ci] / scale + (2 - i) * 6, color=col, lw=0.7, label=lab)
ax.axvline(p_true / 100, color='green', ls='--', lw=1.6, alpha=0.85, label='true P')
ax.axvline(s_true / 100, color='purple', ls='--', lw=1.6, alpha=0.85, label='true S')
ax.set_yticks([12, 6, 0]); ax.set_yticklabels(['Z', 'N', 'E'])
ax.set_ylabel('Amplitude (normalised, offset)', fontsize=10)
ax.set_title(f'STEAD record {name}  ({T/100:.0f} s, 100 Hz) — three components', fontsize=11)
ax.legend(fontsize=8, loc='upper right', ncol=5, framealpha=0.9)
ax.grid(alpha=0.2)

# ---------------- 下：响应曲线 ----------------
ax = axes[1]
rows = []
for k, (lab, cls, ck, color) in enumerate(MODELS):
    try:
        st, W, cp, cs, raw = forward(ck, cls)
    except Exception as e:
        print(f'  {lab}: {e}'); continue
    tt = (st + np.arange(W)) / 100.0
    off = (len(MODELS) - 1 - k) * 1.35
    nP = cp / (cp.max() + 1e-12)
    nS = cs / (cs.max() + 1e-12)
    ax.plot(tt, nP + off, color=color, lw=1.25, label=f'{lab} — P')
    ax.plot(tt, nS + off, color=color, lw=1.25, ls=':', label=f'{lab} — S')
    p_hat = st + int(np.argmax(cp)); s_hat = st + int(np.argmax(cs))
    ax.axvline(p_hat / 100, color=color, lw=0.8, alpha=0.5)
    ax.axvline(s_hat / 100, color=color, lw=0.8, alpha=0.5, ls=':')
    rows.append((lab, color, p_hat - p_true, s_hat - s_true, np.asarray(raw['P'])[0]))
    ax.text(0.6, off + 0.95, f'ΔP={p_hat-p_true:+.0f}  ΔS={s_hat-s_true:+.0f} samples',
            fontsize=8.6, color=color, va='center', ha='left', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=color, alpha=0.85, lw=0.7))

ax.axvline(p_true / 100, color='green', ls='--', lw=1.4, alpha=0.8)
ax.axvline(s_true / 100, color='purple', ls='--', lw=1.4, alpha=0.8)
ax.set_yticks([])
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('P (solid) / S (dotted) — peak-normalised', fontsize=10)
ax.set_title('Picking response (dashed = ground truth). '
             'The proposed picker is a logit output and is softmaxed over time before normalisation.',
             fontsize=10.5)
ax.grid(alpha=0.2); ax.set_xlim(0, T / 100)
ax.legend(fontsize=7.4, loc='center left', bbox_to_anchor=(1.005, 0.5),
          ncol=1, framealpha=0.95, borderaxespad=0.0)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('\n  各模型误差（采样点）:')
for lab, color, dp, ds, raw in rows:
    print(f'    {lab:28s} ΔP={dp:+6.0f}  ΔS={ds:+6.0f}   '
          f'raw P range=[{raw.min():+.2f}, {raw.max():+.2f}]')
print(f'\n  图已保存: {OUT}')

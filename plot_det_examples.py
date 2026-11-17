# -*- coding: utf-8 -*-
"""检测示例图（两张）：
   (a) STEAD 事件记录：四模型的 presence 曲线 + 目标区间 + 各自阈值 τ
   (b) STEAD 噪声记录：同上，用来看谁把噪声误判为事件（FP）
说明：STEAD 记录长度 = 检测窗长 6000，因此单窗即整条记录。
"""
import os, numpy as np, torch, matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_evsplit_models as E

OUTDIR = os.environ.get('FIG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = [
 ('BiMamba-Trans.', 'BiMamba-Detector', 'best_detection_model_3C_det_pn197k.pth', '#2ca02c'),
 ('PhaseNet',            'PhaseNet',         'best_phasenet_baseline_6000.pth',        '#ff7f0e'),
 ('PhaseMamba',          'PhaseMamba',       'best_phasemamba_evsplit__phasemamba6000.pth', '#9467bd'),
 ('EQTransformer',       'EQTransformer',    'best_eqtransformer_baseline_6000.pth',   '#d62728'),
]
WRAP = {'BiMamba-Detector': E.BiMambaDetector, 'PhaseNet': E.PhaseNetWrap,
        'PhaseMamba': E.PhaseMambaWrap, 'EQTransformer': E.EQTWrap}

# 阈值：来自 detscore 文件里的正负样本标定（这里用全量交叉拟合的最优 τ）
TAU = {'INSTANCE': 5.4258e-04}   # 仅作参考，下面用 STEAD 自己的 τ

def load_model(cls, ck):
    os.environ['EVAL_WINDOW'] = '6000'
    ckpt_key = 'picker'
    E.CKPT[ckpt_key] = os.path.abspath(ck)
    m = WRAP[cls](dev)
    if not hasattr(m, 'prep'):
        m.prep = E.preprocess_40
    return m

def det_curve(m, cls, x):
    """返回 (presence 曲线, 记录分数)。检测器用 'det'，拾取器用 'P'。"""
    W = getattr(m, 'window', 6000)
    T = x.shape[0]
    st = max(0, T - W) // 2
    win = x[st:st + W]
    prep = m.prep if hasattr(m, 'prep') else E.preprocess_40
    out = m.score(m.to_input(np.stack([prep(win)])))
    key = 'det' if cls == 'BiMamba-Detector' else 'P'
    if key not in out:
        key = 'P' if 'P' in out else list(out)[0]
    c = np.asarray(out[key])[0].astype(np.float64)
    xs = np.linspace(0, 1, len(c))
    return np.interp(np.linspace(0, 1, W), xs, c), float(c.max()), st, W

def pick_event():
    best = None
    for nm, x, p, s in E.DATASETS['STEAD-DET']('events', 0):
        if not (np.isfinite(p) and np.isfinite(s)):
            continue
        if 1300 <= p <= 2100 and 900 <= s - p <= 2000:
            best = (str(nm), x, float(p), float(s)); break
    return best

def pick_noise():
    for nm, x, p, s in E.DATASETS['STEAD-DET']('noise', 0):
        return (str(nm), x, np.nan, np.nan)
    return None

ms = {cls: load_model(cls, ck) for _, cls, ck, _ in MODELS}

def draw(fname, rec, kind):
    name, x, p_true, s_true = rec
    T = x.shape[0]
    t = np.arange(T) / 100.0
    fig = plt.figure(figsize=(13.0, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.24)

    # ---- 上：波形 ----
    ax = fig.add_subplot(gs[0])
    sc = np.std(x[:, 0]) + 1e-9
    for i, (ci, lab, col) in enumerate([(2, 'Z', '#1f77b4'), (1, 'N', '#2ca02c'), (0, 'E', '#d62728')]):
        ax.plot(t, x[:, ci] / sc + (2 - i) * 6.5, color=col, lw=0.65, label=lab)
    if kind == 'event':
        ax.axvline(p_true / 100, color='green', ls='--', lw=1.8, alpha=0.9, label='true P')
        ax.axvline(s_true / 100, color='purple', ls='--', lw=1.8, alpha=0.9, label='true S')
    ax.set_yticks([13, 6.5, 0]); ax.set_yticklabels(['Z', 'N', 'E'])
    ax.set_xlim(0, T / 100)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude (normalised, offset)', fontsize=10)
    ttl = ('STEAD event record' if kind == 'event' else 'STEAD noise record')
    ax.set_title(f'{ttl}  {name}  (T = {T/100:.0f} s)', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right', ncol=5, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ---- 下：presence 曲线 ----
    ax = fig.add_subplot(gs[1])
    rows = []
    for k, (lab, cls, ck, color) in enumerate(MODELS):
        try:
            c, sc_max, st, W = det_curve(ms[cls], cls, x)
        except Exception as e:
            print(f'  {lab}: {e}'); continue
        tt = (st + np.arange(W)) / 100.0
        cn = c / (c.max() + 1e-12)
        off = (len(MODELS) - 1 - k) * 1.35
        ax.plot(tt, cn + off, color=color, lw=1.25, label=f'{lab}')
        rows.append((lab, color, sc_max, off))
        ax.text(0.6, off + 0.95, f'$s$ = {sc_max:.3g}',
                fontsize=9.2, color=color, va='center', ha='left', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.9, lw=0.8))
    if kind == 'event':
        ax.axvline(p_true / 100, color='green', ls='--', lw=1.5, alpha=0.85)
        ax.axvline(s_true / 100, color='purple', ls='--', lw=1.5, alpha=0.85)
    ax.set_yticks([])
    ax.set_xlim(0, T / 100)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Presence response (peak-normalised)', fontsize=10)
    ax.set_title('Per-sample phase-presence response of the four models '
                 '(each normalised to its own maximum)', fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8.2, loc='upper right', ncol=2, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(OUTDIR, fname)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)
    for lab, color, s, off in rows:
        print(f'    {lab:22s} record score = {s:.4g}')

ev = pick_event(); no = pick_noise()
print('  事件记录:', ev[0], ' 噪声记录:', no[0] if no else None)
draw('det_example_event.png', ev, 'event')
if no is not None:
    draw('det_example_noise.png', no, 'noise')

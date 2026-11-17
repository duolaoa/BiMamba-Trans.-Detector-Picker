# -*- coding: utf-8 -*-
"""定量核实：各阶段在 P 区 vs S 区的通道平均响应（多记录统计）。

对每条记录：
    z_c(t) = (a_c(t) - mean_t a_c) / std_t a_c        逐通道沿时间标准化
    m(t)   = mean_c z_c(t)                            通道平均曲线（无 P 偏置 ✓）
指标：
    A_P = m(t) 在 [p-50, p+50] 采样点内的最大绝对值
    A_S = m(t) 在 [s-50, s+50] 采样点内的最大绝对值（S 须落在窗内）
    base= m(t) 全程的标准差（作为噪声标度）
汇总：mean ± sem over traces，以及 A_S/A_P 比
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_evsplit_models as E   # noqa: E402

DATASETS = ['STEAD-CH5C', 'INSTANCE', 'CREW']
CKPTS = [('199K', 'best_lightweight_model_3C_pn197k.pth'),
         ('328K', 'best_lightweight_model_3C_pm322k.pth')]
STAGES = [('LFEM1D', 'proj', '#1f77b4'), ('BiMamba', 'mamba', '#2ca02c'),
          ('Transformer', 'transformer', '#9467bd'), ('Fusion', 'fusion', '#d62728')]
N = 200
WIN = 50            # ±50 采样点窗口
OUT = os.path.join(os.environ.get('FIG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')), 'featuremaps', 'stage_traces')
os.makedirs(OUT, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build(ck):
    E.CKPT['picker'] = os.path.join(HERE, ck)
    return E.BiMambaPicker(device)


def zmap(A):
    A = A if A.shape[0] <= A.shape[1] else A.T
    return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-8)


models = {t: build(c) for t, c in CKPTS}
W = models['199K'].window
rows = []

for ds in DATASETS:
    items = []
    for name, x, p, s in E.DATASETS[ds]('events', 0):
        if not (np.isfinite(p) and np.isfinite(s)):
            continue
        T = x.shape[0]
        if T < W:
            continue
        st = max(0, min(int(round(p - 500)), T - W))
        items.append((str(name), x[st:st + W], float(p) - st, float(s) - st))
        if len(items) >= 1200:
            break
    rng = np.random.RandomState(0)
    sel = sorted(rng.choice(len(items), size=min(N, len(items)), replace=False).tolist())
    acc = {tag: {lbl: [] for lbl, _a, _c in STAGES} for tag, _ in CKPTS}
    accS = {tag: {lbl: [] for lbl, _a, _c in STAGES} for tag, _ in CKPTS}
    base = {tag: {lbl: [] for lbl, _a, _c in STAGES} for tag, _ in CKPTS}
    nS = 0
    for k in sel:
        nm, wave, p_rel, s_rel = items[k]
        xin = E.preprocess_40(wave)
        for tag, _ in CKPTS:
            store, hs = {}, []
            for lbl, attr, _c in STAGES:
                mod = getattr(models[tag].m, attr, None)
                if mod is None: continue
                def mk(l):
                    def hk(m, i, o):
                        t = o[0] if isinstance(o, (tuple, list)) else o
                        store[l] = t.detach().float().cpu()[0].numpy()
                    return hk
                hs.append(mod.register_forward_hook(mk(attr)))
            ten = torch.from_numpy(np.ascontiguousarray(
                models[tag].to_input(np.stack([xin])))).to(device)
            with torch.no_grad():
                models[tag].m(ten)
            for h in hs: h.remove()
            for lbl, attr, _c in STAGES:
                if attr not in store: continue
                Z = zmap(np.asarray(store[attr]))
                mt = Z.mean(0)                      # 通道平均曲线（长度 750，单位=4 采样点）
                step = 4
                pi = int(round(p_rel / step)); si = int(round(s_rel / step))
                wf = max(1, WIN // step)
                lo, hi = max(0, pi - wf), min(len(mt), pi + wf + 1)
                AP = float(np.max(np.abs(mt[lo:hi]))) if hi > lo else np.nan
                if 0 <= si < len(mt):
                    lo2, hi2 = max(0, si - wf), min(len(mt), si + wf + 1)
                    AS = float(np.max(np.abs(mt[lo2:hi2]))) if hi2 > lo2 else np.nan
                else:
                    AS = np.nan
                acc[tag][lbl].append(AP); accS[tag][lbl].append(AS)
                base[tag][lbl].append(float(np.std(mt)))
        nS += 1
    print(f'  {ds}: 用 {len(sel)} 条（S 落在窗内可算的比例见下）')
    for tag, _ in CKPTS:
        for lbl, _a, _c in STAGES:
            a = np.array(acc[tag][lbl], float)
            b = np.array(accS[tag][lbl], float)
            okS = np.isfinite(b)
            rows.append(dict(model=tag, dataset=ds, stage=lbl, n=len(a),
                             A_P_mean=float(np.nanmean(a)), A_P_sem=float(np.nanstd(a)/np.sqrt(len(a))),
                             A_S_mean=float(np.nanmean(b)), A_S_sem=float(np.nanstd(b[okS])/np.sqrt(max(okS.sum(),1))),
                             A_S_over_A_P=float(np.nanmean(b)/max(np.nanmean(a),1e-9)),
                             S_in_window_pct=float(okS.mean()*100),
                             baseline_std=float(np.mean(base[tag][lbl]))))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, 's_region_response.csv'), index=False)

print()
print('=' * 116)
print('各阶段「通道平均响应」：P 区 vs S 区（±50 采样点内的最大 |m(t)|；200 条记录平均）')
print('=' * 116)
print(f"{'数据集':11s} {'模型':5s} {'阶段':12s} | {'A_P':>8s} {'A_S':>8s} {'A_S/A_P':>8s} "
      f"| {'全程 std':>9s} | {'S在窗内%':>8s}")
for ds in DATASETS:
    for tag, _ in CKPTS:
        for lbl, _a, _c in STAGES:
            r = t[(t.model == tag) & (t.dataset == ds) & (t.stage == lbl)]
            if not len(r): continue
            r = r.iloc[0]
            print(f"{ds if lbl=='LFEM1D' else '':11s} {tag:5s} {lbl:12s} | {r.A_P_mean:8.3f} "
                  f"{r.A_S_mean:8.3f} {r.A_S_over_A_P:8.3f} | {r.baseline_std:9.3f} | "
                  f"{r.S_in_window_pct:8.1f}")
    print()

# 图：分组柱状（P 区 vs S 区）
fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.0 * len(DATASETS), 4.4), squeeze=False)
for j, ds in enumerate(DATASETS):
    ax = axes[0][j]
    x = np.arange(len(STAGES)); w = 0.38
    for k, (tag, lab) in enumerate((('199K', '199K'), ('328K', '328K'))):
        ap, as_, = [], []
        for lbl, _a, _c in STAGES:
            r = t[(t.model == tag) & (t.dataset == ds) & (t.stage == lbl)].iloc[0]
            ap.append(r.A_P_mean); as_.append(r.A_S_mean)
        ax.bar(x + (k - 0.5) * w * 2 - w/2 + k*0.0, ap, w, label=f'{lab} · P region',
               color='#2563eb' if k == 0 else '#60a5fa')
        ax.bar(x + (k - 0.5) * w * 2 + w/2, as_, w, label=f'{lab} · S region',
               color='#dc2626' if k == 0 else '#f87171')
    ax.set_xticks(x); ax.set_xticklabels([l for l, _a, _c in STAGES], fontsize=8.5, rotation=15)
    ax.set_ylabel('channel-mean |response| (max in ±50 samples)', fontsize=9)
    ax.set_title(ds, fontsize=10.5); ax.grid(alpha=.25, axis='y')
    if j == 0: ax.legend(fontsize=7.5, ncol=2)
fig.suptitle('Channel-averaged response at the P region vs the S region, per stage\n'
             '(mean over 200 traces; a module that ignores S should show a much smaller red bar)',
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
p = os.path.join(OUT, 's_region_response.png')
fig.savefig(p, dpi=165); plt.close(fig)
print('图:', p)

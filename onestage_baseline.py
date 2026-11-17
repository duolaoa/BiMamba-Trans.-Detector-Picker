# -*- coding: utf-8 -*-
"""对比：基线的两种端到端方案
   (A) 2-stage：共用本文检测器 → 锚点 + ROI → 窗内 argmax 投票   （现稿口径）
   (B) 1-stage：直接用基线自身的到时头，滑窗取**全局 argmax**，无门控、无投票
   数据：INSTANCE（跨域，最有代表性）与 STEAD（域内）
   容差 0.5 s；P/S 各自统计
"""
import os, sys, numpy as np, torch, pandas as pd
import eval_evsplit_models as E

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = [
 ('BiMamba-Trans. (ours)', 'BiMamba-Picker', 'best_lightweight_model_3C_pn197k_w6000v3.pth'),
 ('PhaseNet',              'PhaseNet',       'best_phasenet_baseline_6000.pth'),
 ('EQTransformer',         'EQTransformer',  'best_eqtransformer_baseline_6000.pth'),
 ('PhaseMamba',            'PhaseMamba',     'best_phasemamba_evsplit__phasemamba6000.pth'),
]
WRAP = {'BiMamba-Picker': E.BiMambaPicker, 'PhaseNet': E.PhaseNetWrap,
        'PhaseMamba': E.PhaseMambaWrap, 'EQTransformer': E.EQTWrap}
TOL = 50.0          # 0.5 s = 50 点
STRIDE = 1000

def softmax_t(a):
    z = np.asarray(a, dtype=np.float64)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / (z.sum(axis=1, keepdims=True) + 1e-12)

def load(cls, ck):
    os.environ['EVAL_WINDOW'] = '6000'
    E.CKPT['picker'] = os.path.abspath(ck)
    m = WRAP[cls](dev)
    if not hasattr(m, 'prep'):
        m.prep = E.preprocess_40
    return m

def predict_1stage(m, cls, x):
    """滑窗取全局 argmax（无门控、无 ROI、无投票）。"""
    W = getattr(m, 'window', 6000)
    T = x.shape[0]
    prep = m.prep if hasattr(m, 'prep') else E.preprocess_40
    starts = list(range(0, max(1, T - W + 1), STRIDE))
    if not starts:
        starts = [0]
    wins = np.stack([prep(x[s:s + W]) for s in starts])
    out = m.score(m.to_input(wins))
    def flat(key):
        c = np.asarray(out[key])
        if cls == 'BiMamba-Picker':
            c = softmax_t(c)                       # logits → 概率
        return c
    cP, cS = flat('P'), flat('S')
    # 拼成绝对时间轴
    Tlen = W + (len(starts) - 1) * STRIDE
    fullP = np.full(Tlen, -np.inf); fullS = np.full(Tlen, -np.inf)
    for i, s in enumerate(starts):
        fullP[s:s + W] = np.maximum(fullP[s:s + W], cP[i])
        fullS[s:s + W] = np.maximum(fullS[s:s + W], cS[i])
    return int(np.argmax(fullP)), int(np.argmax(fullS))

def run(ds):
    print(f'\n{"="*100}\n■ {ds}\n{"="*100}')
    rows = []
    ms = {cls: load(cls, ck) for _, cls, ck in MODELS}
    n = 0
    for nm, x, p, s in E.DATASETS[ds]('events', 0):
        if not np.isfinite(p):
            continue
        T = x.shape[0]
        if T < 6000:
            continue
        rec = {}
        for lab, cls, ck in MODELS:
            try:
                ph, sh = predict_1stage(ms[cls], cls, x)
                rec[lab] = (ph, sh)
            except Exception as e:
                rec[lab] = None
        rows.append((str(nm), float(p), float(s) if np.isfinite(s) else np.nan, rec))
        n += 1
        if n >= 3000:
            break
    print(f'  评估事件数 n = {n}')
    print(f'  {"模型":24s} {"P命中":>8s} {"P-MAE(s)":>10s} {"S命中":>8s} {"S-MAE(s)":>10s}')
    for lab, cls, ck in MODELS:
        dp, ds_ = [], []
        for nm, p, s, rec in rows:
            if rec.get(lab) is None:
                continue
            ph, sh = rec[lab]
            dp.append(ph - p)
            if np.isfinite(s):
                ds_.append(sh - s)
        dp = np.array(dp); ds_ = np.array(ds_)
        hp = (np.abs(dp / 100.0) <= 0.5).mean() * 100 if len(dp) else np.nan
        hs = (np.abs(ds_ / 100.0) <= 0.5).mean() * 100 if len(ds_) else np.nan
        mp = np.abs(dp / 100.0).mean() if len(dp) else np.nan
        ms_ = np.abs(ds_ / 100.0).mean() if len(ds_) else np.nan
        print(f'  {lab:24s} {hp:7.2f}% {mp:10.3f} {hs:7.2f}% {ms_:10.3f}')
        rows_out.append(dict(dataset=ds, model=lab, scheme='1-stage (own head)', n_P=len(dp),
                             hit_P=hp, mae_P=mp, n_S=len(ds_), hit_S=hs, mae_S=ms_))
    return rows_out

rows_out = []
for ds in ('INSTANCE', 'STEAD-DET'):
    try:
        run(ds)
    except Exception as e:
        print(f'  {ds} 失败: {e}')
if rows_out:
    pd.DataFrame(rows_out).to_csv('eval_all/ONESTAGE_VS_TWOSTAGE.csv', index=False)
    print('\n  saved eval_all/ONESTAGE_VS_TWOSTAGE.csv')

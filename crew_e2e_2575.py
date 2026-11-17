# -*- coding: utf-8 -*-
"""CREW 端到端评估（严格按正文 §2.6 的两阶段协议）

阶段 1（门控 + 锚点）
    · 本文：**BiMamba-Trans. 检测器**（det 通道 = P 存在）滑窗 W=6000, stride=Δs
            → 逐窗取 max → a_j → Ŝ = max_j a_j → 与 τ 比较 → 锚点 = 主峰位置
    · 基线：EQT / PhaseMamba 用各自 det 通道；PhaseNet 用 1 − noise
阶段 2（ROI 内投票拾取）
    · 本文：**BiMamba 拾取器**
    · 基线：各自 P/S 头
    · 在 [锚点−B, 锚点+F] 内以 Δv 滑窗，每窗一个投票，众数箱取中位数
阈值 τ
    · 25% 事件校准（正窗 = P 居中；负窗 = 距 P/S ≥ Δ）用 F1 选 τ*
    · 剩余 75% 评估
"""
from __future__ import annotations
import argparse, os, sys, time
import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_evsplit_models as E

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


CREW = os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'CREW_Dataset_files', '000_CREW.hdf5')
# (显示名, gate类, gate权重, pick类, pick权重)
MODELS = [
 ('BiMamba-Trans.', 'BiMambaDetector', 'best_detection_model_3C_det_pn197k.pth',
                    'BiMambaPicker',   'best_lightweight_model_3C_pn197k_w6000v3.pth'),
 ('PhaseNet',       'PhaseNet',        'best_phasenet_baseline_6000.pth',
                    'PhaseNet',        'best_phasenet_baseline_6000.pth'),
 ('PhaseMamba',     'PhaseMamba',      'best_phasemamba_evsplit__phasemamba6000.pth',
                    'PhaseMamba',      'best_phasemamba_evsplit__phasemamba6000.pth'),
 ('EQTransformer',  'EQTransformer',   'best_eqtransformer_baseline_6000.pth',
                    'EQTransformer',   'best_eqtransformer_baseline_6000.pth'),
]
WRAP = {'BiMambaDetector': E.BiMambaDetector, 'BiMambaPicker': E.BiMambaPicker,
        'PhaseNet': E.PhaseNetWrap, 'PhaseMamba': E.PhaseMambaWrap,
        'EQTransformer': E.EQTWrap}

def load(cls, ck, dev):
    os.environ['EVAL_WINDOW'] = '6000'
    key = 'detector' if 'Detector' in cls else 'picker'
    E.CKPT[key] = os.path.abspath(ck)
    m = WRAP[cls](dev)
    if not hasattr(m, 'prep'):
        m.prep = E.preprocess_40
    return m

def presence(out, cls):
    """门控用的存在曲线（与主协议一致）。"""
    if cls == 'BiMambaDetector':
        return np.asarray(out['det']).astype(float)       # P 存在通道
    if cls == 'PhaseNet':
        return 1.0 - np.asarray(out['noise']).astype(float)
    if 'det' in out:
        return np.asarray(out['det']).astype(float)
    return np.asarray(out['P']).astype(float)

def sm(a):
    z = np.asarray(a, dtype=np.float64)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / (z.sum(axis=1, keepdims=True) + 1e-12)

def prf(tp, fp, fn):
    P = tp/max(tp+fp, 1e-12); R = tp/max(tp+fn, 1e-12)
    return P, R, 2*P*R/max(P+R, 1e-12)

def best_tau(sp, sn):
    sp = np.asarray(sp, float); sn = np.asarray(sn, float)
    qs = np.quantile(np.unique(np.r_[sp, sn]), np.linspace(0.002, 0.998, 400))
    best = (-1, float(qs[0]))
    for th in qs:
        tp = int((sp >= th).sum()); fp = int((sn >= th).sum()); fn = int((sp < th).sum())
        _, _, F = prf(tp, fp, fn)
        if F > best[0]:
            best = (F, float(th))
    return best[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--pre', type=int, default=500)
    ap.add_argument('--delta', type=int, default=4000)
    ap.add_argument('--det-stride', type=int, default=1000)
    ap.add_argument('--pick-stride', type=int, default=250)
    ap.add_argument('--B', type=int, default=4000)
    ap.add_argument('--F', type=int, default=2000)
    ap.add_argument('--rho', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    a = ap.parse_args()
    dev = __import__('torch').device('cuda')

    models = {}
    for lab, gcls, gck, pcls, pck in MODELS:
        models[lab] = (load(gcls, gck, dev), gcls, load(pcls, pck, dev), pcls)
        print(f'  {lab:16s} gate={gcls:16s} W={models[lab][0].window}  pick={pcls}', flush=True)

    recs = []
    with h5py.File(CREW, 'r') as f:
        for k in f['data'].keys():
            d = f['data'][k]
            x = np.asarray(d[()], dtype=np.float32)
            if x.shape[0] == 3 and x.shape[1] != 3: x = x.T
            x = np.ascontiguousarray(x[:, E.ENZ_TO_ZNE])
            p = float(d.attrs.get('P_arrival_sample', np.nan))
            s = float(d.attrs.get('S_arrival_sample', np.nan))
            if not np.isfinite(p) or p <= 0 or p >= x.shape[0]: continue
            recs.append((k, x, p, s))
            if a.limit and len(recs) >= a.limit: break
    print(f'  CREW 有效 {len(recs)} 条', flush=True)

    rng = np.random.RandomState(a.seed)
    idx = rng.permutation(len(recs)); ncal = int(round(a.rho*len(recs)))
    cal = [recs[i] for i in idx[:ncal]]; tst = [recs[i] for i in idx[ncal:]]
    print(f'  校准 {len(cal)} / 测试 {len(tst)}', flush=True)

    def batch_score(m, cls, xs, mode):
        out = []
        for i in range(0, len(xs), a.bs):
            wb = np.stack([m.prep(v) for v in xs[i:i+a.bs]])
            o = m.score(m.to_input(wb))
            out.append(presence(o, cls).max(axis=1) if mode == 'max' else presence(o, cls))
        return np.concatenate(out) if out else np.array([])

    # ---------- 校准 τ ----------
    taus = {}
    for lab, (gm, gcls, pm, pcls) in models.items():
        W = getattr(gm, 'window', 6000)
        sp, sn = [], []
        for k, x, p, s in cal:
            T = x.shape[0]
            if T < W: continue
            st = int(np.clip(round(p - a.pre), 0, T - W)); sp.append(x[st:st+W])
            cd = []
            if p - a.delta - W >= 0: cd.append((0, int(p - a.delta - W)))
            if np.isfinite(s) and s + a.delta + W <= T: cd.append((int(s + a.delta), T - W))
            if cd:
                lo, hi = cd[rng.randint(len(cd))]
                ns = int(rng.randint(lo, hi+1)) if hi > lo else int(lo)
                sn.append(x[ns:ns+W])
        svp = batch_score(gm, gcls, sp, 'max'); svn = batch_score(gm, gcls, sn, 'max')
        taus[lab] = best_tau(svp, svn) if len(svp) and len(svn) else np.nan
        print(f'  [{lab}] τ* = {taus[lab]:.4e}   正窗 {len(svp)} 负窗 {len(svn)}', flush=True)

    # ---------- 测试集 ----------
    res = {l: dict(tp=0, fp=0, fn=0, dp=[], ds=[], anc=[]) for l in models}
    for lab, (gm, gcls, pm, pcls) in models.items():
        Wg = getattr(gm, 'window', 6000); Wp = getattr(pm, 'window', 6000)
        tau = taus[lab]; t0 = time.time()
        for cnt, (k, x, p, s) in enumerate(tst):
            T = x.shape[0]
            if T < Wg: continue
            starts = list(range(0, max(1, T - Wg + 1), a.det_stride))
            full = np.full(Wg + (len(starts)-1)*a.det_stride, -np.inf, dtype=np.float64)
            for i in range(0, len(starts), 16):
                ss = starts[i:i+16]
                wb = np.stack([gm.prep(x[o:o+Wg]) for o in ss])
                pv = presence(gm.score(gm.to_input(wb)), gcls)
                for j, o in enumerate(ss):
                    full[o:o+Wg] = np.maximum(full[o:o+Wg], pv[j])
            # 检测混淆矩阵（正= P 居中窗；负= 远离窗）
            stp = int(np.clip(round(p - a.pre), 0, T - Wg))
            zp = float(presence(gm.score(gm.to_input(np.stack([gm.prep(x[stp:stp+Wg])]))), gcls).max())
            zn = np.nan
            cd = []
            if p - a.delta - Wg >= 0: cd.append((0, int(p - a.delta - Wg)))
            if np.isfinite(s) and s + a.delta + Wg <= T: cd.append((int(s + a.delta), T - Wg))
            if cd:
                lo, hi = cd[rng.randint(len(cd))]
                ns = int(rng.randint(lo, hi+1)) if hi > lo else int(lo)
                zn = float(presence(gm.score(gm.to_input(np.stack([gm.prep(x[ns:ns+Wg])]))), gcls).max())
            res[lab]['tp'] += int(zp >= tau); res[lab]['fn'] += int(zp < tau)
            if np.isfinite(zn): res[lab]['fp'] += int(zn >= tau)
            # 锚点
            above = np.where(full >= tau)[0]
            if len(above) == 0: continue
            anc = int(np.argmax(full))
            res[lab]['anc'].append(abs(anc - p))
            # ROI 投票
            lo, hi = max(0, anc - a.B), min(T, anc + a.F)
            vs = [v for v in range(lo, max(lo+1, hi - Wp + 1), a.pick_stride) if v + Wp <= T]
            if not vs: continue
            wb = np.stack([pm.prep(x[v:v+Wp]) for v in vs])
            o = pm.score(pm.to_input(wb))
            ap_, as_ = np.asarray(o['P']), np.asarray(o['S'])
            if pcls == 'BiMambaPicker':
                ap_, as_ = sm(ap_), sm(as_)
            vp = np.array(vs) + ap_.argmax(1); vsm = np.array(vs) + as_.argmax(1)
            def mode(arr):
                b = np.round(arr/10).astype(int)
                u, c = np.unique(b, return_counts=True)
                return float(np.median(arr[b == u[c.argmax()]]))
            res[lab]['dp'].append(abs(mode(vp) - p))
            if np.isfinite(s): res[lab]['ds'].append(abs(mode(vsm) - s))
            if cnt % 300 == 0:
                print(f'    {lab} {cnt}/{len(tst)} {time.time()-t0:.0f}s', flush=True)

    print('\n' + '=' * 108)
    print('CREW 端到端（两阶段协议，25% 校准 / 75% 测试）')
    print('=' * 108)
    print('  {:16s} {:>10s} {:>6s} {:>5s} {:>6s} | {:>7s} {:>7s} {:>7s} | {:>8s} {:>9s} {:>8s} {:>9s}'.format(
        '模型','τ*','TP','FP','FN','Pr','Re','F1','P≤0.5s','P-MAE(s)','S≤0.5s','S-MAE(s)'))
    rows = []
    for lab in models:
        r = res[lab]
        P_, R_, F_ = prf(r['tp'], r['fp'], r['fn'])
        dp = np.array(r['dp'])/100.0; ds = np.array(r['ds'])/100.0
        hp = (dp <= 0.5).mean()*100 if len(dp) else np.nan
        hs = (ds <= 0.5).mean()*100 if len(ds) else np.nan
        print('  {:16s} {:10.3e} {:6d} {:5d} {:6d} | {:7.4f} {:7.4f} {:7.4f} | {:7.2f}% {:9.3f} {:7.2f}% {:9.3f}'.format(
            lab, taus[lab], r['tp'], r['fp'], r['fn'], P_, R_, F_, hp,
            dp.mean() if len(dp) else np.nan, hs, ds.mean() if len(ds) else np.nan))
        rows.append(dict(model=lab, tau=taus[lab], TP=r['tp'], FP=r['fp'], FN=r['fn'],
                         Pr=P_, Re=R_, F1=F_, hit_P=hp, mae_P=dp.mean() if len(dp) else np.nan,
                         n_P=len(dp), hit_S=hs, mae_S=ds.mean() if len(ds) else np.nan,
                         n_S=len(ds), anc_med=float(np.median(r['anc'])) if r['anc'] else np.nan))
    pd.DataFrame(rows).to_csv('eval_all/CREW_E2E_2575.csv', index=False)
    print('\n  saved eval_all/CREW_E2E_2575.csv')

if __name__ == '__main__':
    main()

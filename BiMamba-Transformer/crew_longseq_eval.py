# -*- coding: utf-8 -*-
"""CREW 长序列（30000 点）的两种推断方案。

背景：CREW 全部是事件记录、没有噪声集，所以此前检测指标"不适用"。本脚本用两种
方案给 CREW 补上检测（P/R/F1/混淆矩阵）与**不用真值定窗的盲拾取**：

方案 A（窗口拼接）：对每条记录
    * 正样本窗：把 P 摆在窗内 ~pre 处（与训练/评测口径一致）
    * 负样本窗：从"距 P、S 都 >= Δ"的区段里随机取（pre-P 或 post-S）
    -> 在混合集上做记录级检测判决（取该模型检测通道在窗内的最大值）
    -> 得到 CREW 的检测 Pr/Re/F1 + 混淆矩阵 + PR-AUC

方案 B（整段滑窗）：在整条 30000 点记录上以 stride 滑窗、逐点 max 聚合得到曲线
    * 盲拾取：取 **P 通道曲线** 的全局 argmax 作为预测 P（**不使用真值定窗**）
      -> 给出比"真值锚定窗"更严格、更接近连续监测场景的 P 精度
    * 并统计每个 300 s 记录里检测通道超过阈值的时间点数（连续监测的虚警负担）
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_evsplit_models as E                       # noqa: E402

CREW = os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'CREW_Dataset_files', '000_CREW.hdf5')
PICKERS = ['BiMamba-Picker', 'PhaseNet', 'EQTransformer', 'PhaseMamba']
BUILDERS = {'BiMamba-Picker': lambda d: E.BiMambaPicker(d),
            'PhaseNet': lambda d: E.PhaseNetWrap(d),
            'EQTransformer': lambda d: E.EQTWrap(d),
            'PhaseMamba': lambda d: E.PhaseMambaWrap(d),
            'BiMamba-Detector': lambda d: E.BiMambaDetector(d)}


def prep_windows(model, wins):
    prep = getattr(model, 'prep', None)
    return np.stack([(prep(w) if prep else E.preprocess_40(w)) for w in wins])


def score_windows(model, wins, bs=32):
    """返回 (检测通道窗内最大值, P 通道 argmax) 两个数组。"""
    dets, pargs = [], []
    for i in range(0, len(wins), bs):
        chunk = prep_windows(model, wins[i:i + bs])
        o = model.score(model.to_input(chunk))
        det = o['det'] if 'det' in o else o['P']
        dets.append(np.asarray(det).max(axis=1))
        if 'P' in o:
            pargs.append(np.asarray(o['P']).argmax(axis=1))
        else:
            pargs.append(np.full(len(chunk), -1))
    if not dets:
        return np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(dets), np.concatenate(pargs).astype(int)


def prf(tp, fp, fn):
    P = tp / max(tp + fp, 1e-9); R = tp / max(tp + fn, 1e-9)
    return P, R, 2 * P * R / max(P + R, 1e-9)


# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 条（0=全部）')
    ap.add_argument('--pre', type=int, default=500, help='正样本窗内 P 的位置')
    ap.add_argument('--delta', type=int, default=2000, help='负样本窗距 P/S 的最小点数')
    ap.add_argument('--stride', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--det-thr', type=float, default=0.3)
    a = ap.parse_args()
    dev = __import__('torch').device('cuda')
    models = {n: BUILDERS[n](dev) for n in PICKERS + ['BiMamba-Detector']}
    for n, m in models.items():
        print(f'  {n:18s} window={m.window}')
    rng = np.random.default_rng(a.seed)

    pos_scores = {n: [] for n in models}          # 方案A：正样本窗的检测分
    neg_scores = {n: [] for n in models}          # 方案A：负样本窗的检测分
    blind = {n: [] for n in PICKERS}              # 方案B：盲拾取(预测绝对到时, 真值)
    hits = {n: [] for n in PICKERS}               # 方案B：阈值以上的时间点数
    n_trace = 0
    t0 = time.time()

    with h5py.File(CREW, 'r') as f:
        ks = list(f['data'].keys())
        if a.limit:
            ks = ks[:a.limit]
        for ti, k in enumerate(ks):
            d = f['data'][k]
            x = np.asarray(d[()], dtype=np.float32)
            if x.shape[0] == 3 and x.shape[1] != 3:
                x = x.T
            x = np.ascontiguousarray(x[:, E.ENZ_TO_ZNE])     # -> (T,3) [Z,N,E]
            T = x.shape[0]
            p = float(d.attrs.get('P_arrival_sample', np.nan))
            s = float(d.attrs.get('S_arrival_sample', np.nan))
            if not np.isfinite(p) or p <= 0 or p >= T:
                continue
            n_trace += 1

            for name, m in models.items():
                W = m.window
                if T < W:
                    continue
                # ---------- 方案 A ----------
                st = int(np.clip(round(p - a.pre), 0, T - W))
                pos_scores[name].append((st, x[st:st + W]))
                cands = []
                if p - a.delta - W >= 0:
                    cands.append((0, int(p - a.delta - W)))
                if np.isfinite(s) and s + a.delta + W <= T:
                    cands.append((int(s + a.delta), T - W))
                if cands:
                    lo, hi = cands[rng.integers(len(cands))]
                    ns = int(rng.integers(lo, hi + 1)) if hi > lo else int(lo)
                    neg_scores[name].append((ns, x[ns:ns + W]))
                # ---------- 方案 B（只对拾取模型；P 通道曲线） ----------
                if name in PICKERS:
                    starts = list(range(0, max(1, T - W + 1), a.stride))
                    wins = [x[o:o + W] for o in starts]
                    # 一次打分同时拿到"每窗 P 通道最大值"与"每窗 P 通道 argmax"
                    dt, pa = score_windows(m, wins)
                    # 读出1：P 通道响应最强的窗
                    bw = int(np.argmax(dt))
                    g_pick = int(starts[bw] + pa[bw])
                    # 读出2：最早的"强响应"窗（>=50% 全局最强），更接近"先到先取"
                    thr_rel = 0.5 * float(dt.max())
                    idx = np.where(dt >= thr_rel)[0]
                    fw = int(idx[0]) if len(idx) else bw
                    f_pick = int(starts[fw] + pa[fw])
                    blind[name].append((g_pick, f_pick, int(p)))
                    hits[name].append(int((dt >= a.det_thr).sum()))
            if (ti + 1) % 500 == 0:
                print(f'  {ti+1}/{len(ks)}  {n_trace} 条有效  '
                      f'{time.time()-t0:.0f}s', flush=True)

    # ---------------- 汇总：方案 A 检测 ----------------
    print('\n' + '=' * 108)
    print('【方案 A】CREW 窗口拼接检测集（正=P 窗，负=远离 P/S 的噪声窗）')
    print('=' * 108)
    print(f"{'模型':18s} {'窗长':>5s} {'n正':>6s} {'n负':>6s} | {'Pr':>7s} {'Re':>7s} {'F1':>7s} "
          f"{'F1_best':>8s} {'阈值best':>10s} | {'正分中位':>9s} {'负分中位':>9s}")
    out = []
    for name, m in models.items():
        pos = np.array([v for _, v in []])  # placeholder
        # 重新按窗打分（延迟到汇总阶段，避免内存里存全部波形过多）
        stp = [w for _, w in pos_scores[name]]
        stn = [w for _, w in neg_scores[name]]
        if not stp or not stn:
            print(f'{name:18s} 数据不足'); continue
        sp, _ = score_windows(m, stp)
        sn, _ = score_windows(m, stn)
        thr = a.det_thr
        tp = int((sp >= thr).sum()); fp = int((sn >= thr).sum()); fn = int((sp < thr).sum())
        P, R, F = prf(tp, fp, fn)
        best = (0, 0)
        for th in np.unique(np.r_[sp, sn]):
            t2 = (sp >= th).sum(); f2 = (sn >= th).sum(); n2 = (sp < th).sum()
            _, _, Ff = prf(t2, f2, n2)
            if Ff > best[0]: best = (Ff, float(th))
        print(f'{name:18s} {m.window:5d} {len(sp):6d} {len(sn):6d} | {P:7.4f} {R:7.4f} {F:7.4f} '
              f'{best[0]:8.4f} {best[1]:10.2e} | {np.median(sp):9.2e} {np.median(sn):9.2e}')
        out.append(dict(scheme='A', model=name, window=m.window, n_pos=len(sp), n_neg=len(sn),
                        Pr=P, Re=R, F1=F, F1_best=best[0], thr_best=best[1],
                        pos_med=float(np.median(sp)), neg_med=float(np.median(sn)),
                        TP=tp, FP=fp, FN=fn))

    # ---------------- 汇总：方案 B 盲拾取 ----------------
    print('\n' + '=' * 108)
    print('【方案 B】CREW 整段滑窗的**盲拾取**（P 通道响应最强处，不使用真值定窗）')
    print('=' * 108)
    print(f"{'模型':18s} {'读出':8s} {'n':>6s} {'MAE':>9s} {'RMSE':>10s} {'中位':>8s} "
          f"{'≤10':>7s} {'≤20':>7s} {'≤100':>7s} | {'每300s记录超阈值窗数中位':>22s}")
    for name in PICKERS:
        b = np.array(blind[name], dtype=float)
        if not len(b):
            continue
        for col, tag in ((0, '全局最强窗'), (1, '最早强响应窗')):
            err = b[:, col] - b[:, 2]
            ae = np.abs(err)
            print(f'{name:18s} {tag:8s} {len(b):6d} {ae.mean():9.2f} '
                  f'{np.sqrt((err**2).mean()):10.2f} {np.median(ae):8.1f} '
                  f'{np.mean(ae<=10)*100:6.1f}% {np.mean(ae<=20)*100:6.1f}% '
                  f'{np.mean(ae<=100)*100:6.1f}% | {np.median(hits[name]):21.0f}')
            out.append(dict(scheme='B', readout=tag, model=name,
                            window=models[name].window, n_pos=len(b),
                            P_MAE=float(ae.mean()), P_RMSE=float(np.sqrt((err**2).mean())),
                            P_med=float(np.median(ae)), P_le10=float(np.mean(ae<=10)*100),
                            P_le20=float(np.mean(ae<=20)*100),
                            thr_hits_med=float(np.median(hits[name]))))
    import pandas as pd
    pd.DataFrame(out).to_csv(os.path.join(HERE, 'eval_all', 'CREW_longseq_schemes.csv'), index=False)
    print('\n已保存: eval_all/CREW_longseq_schemes.csv')


if __name__ == '__main__':
    main()

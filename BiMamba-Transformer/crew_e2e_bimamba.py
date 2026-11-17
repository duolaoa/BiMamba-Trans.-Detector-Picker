# -*- coding: utf-8 -*-
"""CREW 长记录的 **BiMamba 端到端（两阶段）盲拾取**。

修正：BiMamba 是"检测器 + 拾取器"两阶段架构，拾取器只在"锚点定窗"下工作，
不能用拾取器单独滑窗来评价（那是让它做它没被训练做的事）。正确流程：

    阶段1 检测器滑窗（W=6000, stride=1000）-> 逐点 max 聚合得检测曲线
          -> 取锚点：
             (a) a_edge  = 首个 >= 阈值的采样点（部署口径，之前验证过锚点误差中位 2-25 点）
             (b) a_argmax= 曲线全局最强峰位置（对照）
    阶段2 在 [锚点-500, 锚点+2500) 切 3000 点窗 -> 拾取器 -> P 通道 argmax -> 绝对到时
    与真值 P 比较（全程不使用真值来定锚点）

同时用同样的"两阶段"口径给出检测器的判决（score>=阈值）以便与方案A对齐。
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_evsplit_models as E                       # noqa: E402

CREW = os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'CREW_Dataset_files', '000_CREW.hdf5')


# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--pre', type=int, default=500)
    ap.add_argument('--stride', type=int, default=1000)
    ap.add_argument('--det-ckpt', default=os.path.join(HERE, 'best_detection_model_3C_wofocal.pth'))
    ap.add_argument('--thr', type=float, nargs='+', default=[0.3, 1e-4],
                    help='检测阈值（0.3=论文口径；1e-4≈检测器自身工作点）')
    ap.add_argument('--tol', type=int, default=10)
    ap.add_argument('--tag', default='', help='输出文件后缀（避免覆盖其它拾取器的结果）')
    a = ap.parse_args()
    os.environ['EVAL_DET_CKPT'] = a.det_ckpt
    import torch
    dev = torch.device('cuda')
    det = E.BiMambaDetector(dev)
    pick = E.BiMambaPicker(dev)
    print(f'检测器: {os.path.basename(a.det_ckpt)}  window={det.window}')
    print(f'拾取器: window={pick.window}  预取 {a.pre} 点')

    # 读入共识多事件标志（用于分层：单事件 vs 多事件记录）
    flag = {}
    fc = os.path.join(HERE, 'eval_all', 'consensus', 'consensus_CREW.csv')
    if os.path.exists(fc):
        cd = pd.read_csv(fc)
        flag = dict(zip(cd['trace'].astype(str), cd['multi_consensus']))
        print(f'共识标志已载入: {len(flag)} 条 '
              f'（多事件 {int(sum(flag.values()))} 条）')
    rows = []
    t0 = time.time()
    with h5py.File(CREW, 'r') as f:
        ks = list(f['data'].keys())
        if a.limit:
            ks = ks[:a.limit]
        for i, k in enumerate(ks):
            d = f['data'][k]
            x = np.asarray(d[()], dtype=np.float32)
            if x.shape[0] == 3 and x.shape[1] != 3:
                x = x.T
            x = np.ascontiguousarray(x[:, E.ENZ_TO_ZNE])
            T = x.shape[0]
            p = float(d.attrs.get('P_arrival_sample', np.nan))
            if not np.isfinite(p) or p <= 0 or p >= T:
                continue
            W = det.window
            if T < W + pick.window:
                continue
            starts = list(range(0, T - W + 1, a.stride))
            wins = [x[o:o + W] for o in starts]
            prepped = np.stack([det.prep(w) for w in wins])
            outs = []
            for j in range(0, len(prepped), 32):
                o = det.score(det.to_input(prepped[j:j + 32]))['det']
                outs.append(o)
            outs = np.concatenate(outs, axis=0)
            curve = np.zeros(T, np.float32)
            for o, s in zip(outs, starts):
                np.maximum(curve[s:s + W], o, out=curve[s:s + W])
            a_arg = int(np.argmax(curve))
            rec = dict(trace=str(k), p_true=p, det_peak=float(curve.max()),
                       a_argmax=a_arg, a_argmax_err=float(a_arg - p),
                       multi=int(flag.get(str(k), -1)))
            # 阶段2：对两种锚点 + 每个阈值都跑一次拾取器
            for th in a.thr:
                idx = np.where(curve >= th)[0]
                a_edge = int(idx[0]) if len(idx) else a_arg
                for tag, anc in (('a_edge', a_edge), ('a_argmax', a_arg)):
                    st = int(np.clip(anc - a.pre, 0, T - pick.window))
                    w = x[st:st + pick.window]
                    o = pick.score(pick.to_input(E.preprocess_40(w)[None]))['P'][0]
                    pk = st + int(np.argmax(o))
                    rec[f'pick_{tag}_thr{th:g}'] = pk
                    rec[f'err_{tag}_thr{th:g}'] = float(pk - p)
                    rec[f'ancerr_{tag}_thr{th:g}'] = float(anc - p)
                    rec[f'pin_{tag}_thr{th:g}'] = int(st <= p < st + pick.window)
            rows.append(rec)
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(ks)}  {len(rows)} 条  {time.time()-t0:.0f}s', flush=True)
    t = pd.DataFrame(rows)
    tag = f'_{a.tag}' if a.tag else ''
    t.to_csv(os.path.join(HERE, 'eval_all', f'CREW_e2e_bimamba{tag}.csv'), index=False)

    print('\n' + '=' * 112)
    print('【CREW】BiMamba 端到端盲拾取（检测器滑窗定锚点 -> 拾取器出到时；不用真值定窗）')
    print('=' * 112)
    print(f"{'锚点读出':10s} {'阈值':>8s} {'n':>6s} {'锚点误差中位':>12s} {'P在窗内':>8s} | "
          f"{'MAE':>9s} {'RMSE':>10s} {'中位':>8s} {'≤10':>7s} {'≤20':>7s} {'≤100':>7s}")
    out = []
    for th in a.thr:
        for rd in ('a_edge', 'a_argmax'):
            tag = rd          # 注意：不得复用后面的文件名变量
            e = t[f'err_{tag}_thr{th:g}'].values.astype(float)
            ae = np.abs(e)
            an = np.abs(t[f'ancerr_{tag}_thr{th:g}'].values.astype(float))
            pin = t[f'pin_{tag}_thr{th:g}'].values
            print(f'{tag:10s} {th:8.0e} {len(e):6d} {np.median(an):12.1f} '
                  f'{pin.mean()*100:7.1f}% | {ae.mean():9.2f} {np.sqrt((e**2).mean()):10.2f} '
                  f'{np.median(ae):8.1f} {np.mean(ae<=10)*100:6.1f}% {np.mean(ae<=20)*100:6.1f}% '
                  f'{np.mean(ae<=100)*100:6.1f}%')
            out.append(dict(model='BiMamba-e2e', readout=tag, thr=th, n=len(e),
                            anchor_med=float(np.median(an)), p_in_win=float(pin.mean()*100),
                            P_MAE=float(ae.mean()), P_RMSE=float(np.sqrt((e**2).mean())),
                            P_med=float(np.median(ae)), P_le10=float(np.mean(ae<=10)*100),
                            P_le20=float(np.mean(ae<=20)*100), P_le100=float(np.mean(ae<=100)*100)))
    pd.DataFrame(out).to_csv(
        os.path.join(HERE, 'eval_all', f'CREW_e2e_bimamba_summary{tag}.csv'), index=False)
    print('\n' + '=' * 112)
    print('【分层】单事件记录 vs 多事件记录（共识判据）—— 端到端锚点误差与拾取精度')
    print('=' * 112)
    print(f"{'子集':16s} {'锚点读出':10s} {'阈值':>8s} {'n':>6s} {'锚点误差中位':>12s} "
          f"{'P在窗内':>8s} | {'MAE':>9s} {'≤10':>7s} {'≤20':>7s} {'≤100':>7s}")
    for sub, name in ((0, '单事件(清洗后)'), (1, '多事件'), (-1, '未标注')):
        s = t[t.multi == sub]
        if not len(s):
            continue
        for th in a.thr:
            for tag in ('a_edge', 'a_argmax'):
                e = s[f'err_{tag}_thr{th:g}'].values.astype(float)
                ae = np.abs(e)
                an = np.abs(s[f'ancerr_{tag}_thr{th:g}'].values.astype(float))
                pin = s[f'pin_{tag}_thr{th:g}'].values
                print(f'{name:16s} {tag:10s} {th:8.0e} {len(e):6d} {np.median(an):12.1f} '
                      f'{pin.mean()*100:7.1f}% | {ae.mean():9.1f} {np.mean(ae<=10)*100:6.1f}% '
                      f'{np.mean(ae<=20)*100:6.1f}% {np.mean(ae<=100)*100:6.1f}%')
        print()
    print('对照：PhaseNet/EQT/PhaseMamba 是单模型，方案B 用其自身 P 通道曲线；')
    print('      BiMamba 必须用本表的两阶段口径才有可比性。')
    print('已保存: eval_all/CREW_e2e_bimamba.csv / _summary.csv')


if __name__ == '__main__':
    main()

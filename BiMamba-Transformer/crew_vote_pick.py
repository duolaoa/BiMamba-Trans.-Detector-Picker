# -*- coding: utf-8 -*-
"""拾取器直接定位（长记录）—— 投票/Hough 读出。

依据位置敏感度扫描的结论：BiMamba-Picker 在窗内 17%~60% 区间定位准确（误差 4~15 点），
因此滑窗时"凡是含 P 的窗都会为同一个绝对到时投票" -> 票数直方图的众数 = 真 P。
（此前认为投票会退化，前提是"模型只输出窗中心"，已被扫描推翻。）

对每个模型只用它自己的原生窗长，保证公平；不使用任何真值来定窗。
"""
from __future__ import annotations
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_evsplit_models as E                       # noqa: E402

CREW = os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'CREW_Dataset_files', '000_CREW.hdf5')
MODELS = ['BiMamba-Picker', 'PhaseNet', 'EQTransformer', 'PhaseMamba']
BUILDERS = {'BiMamba-Picker': lambda d: E.BiMambaPicker(d),
            'PhaseNet': lambda d: E.PhaseNetWrap(d),
            'EQTransformer': lambda d: E.EQTWrap(d),
            'PhaseMamba': lambda d: E.PhaseMambaWrap(d)}


def vote_pick(model, x, stride, binw=10):
    """滑窗 -> 每窗 argmax 投票 -> 返回 (众数位置, 投票集中度, 窗数)。"""
    W = model.window
    T = x.shape[0]
    starts = list(range(0, T - W + 1, stride))
    wins = [x[o:o + W] for o in starts]
    prep = getattr(model, 'prep', None)
    votes = []
    for i in range(0, len(wins), 32):
        chunk = np.stack([(prep(w) if prep else E.preprocess_40(w))
                          for w in wins[i:i + 32]])
        out = model.score(model.to_input(chunk))
        chan = out['P'] if 'P' in out else out['det']
        am = np.asarray(chan).argmax(axis=1)
        votes.extend((np.array(starts[i:i + 32]) + am).tolist())
    votes = np.array(votes, dtype=np.int64)
    if not len(votes):
        return -1, 0.0, 0
    lo = int(votes.min())
    hist = np.bincount((votes - lo) // binw)
    bi = int(hist.argmax())
    top = hist[bi]
    # 集中度 = 众数箱票数 / 总窗数（1.0 = 所有窗都投同一处）
    return lo + bi * binw + binw // 2, float(top / len(votes)), len(votes)


# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=300)
    ap.add_argument('--stride', type=int, default=250)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    import torch
    dev = torch.device('cuda')
    models = {n: BUILDERS[n](dev) for n in MODELS}
    for n, m in models.items():
        print(f'  {n:16s} window={m.window}  stride={a.stride}')
    rows = []
    with h5py.File(CREW, 'r') as f:
        n_done = 0
        for k in f['data'].keys():
            if n_done >= a.limit:
                break
            d = f['data'][k]
            x = np.asarray(d[()], dtype=np.float32)
            if x.shape[0] == 3 and x.shape[1] != 3:
                x = x.T
            x = np.ascontiguousarray(x[:, E.ENZ_TO_ZNE])
            p = float(d.attrs.get('P_arrival_sample', np.nan))
            if not np.isfinite(p) or p <= 0 or p >= x.shape[0]:
                continue
            n_done += 1
            for name, m in models.items():
                pk, conc, nv = vote_pick(m, x, a.stride)
                rows.append(dict(model=name, window=m.window, p_true=p, pred=pk,
                                 err=float(pk - p), concentration=conc, n_win=nv))
    t = pd.DataFrame(rows)
    tag = f'_{a.tag}' if a.tag else ''
    t.to_csv(os.path.join(HERE, 'eval_all', f'crew_vote_pick{tag}.csv'), index=False)
    print('\n' + '=' * 106)
    print('【拾取器直接定位 · CREW 300s 记录】投票/Hough 读出（不用真值定窗）')
    print('=' * 106)
    print(f"{'模型':16s} {'窗长':>5s} {'窗数':>6s} {'投票集中度':>10s} | {'MAE':>9s} {'RMSE':>10s} "
          f"{'中位':>8s} {'≤10':>7s} {'≤20':>7s} {'≤100':>7s}")
    out = []
    for name in MODELS:
        s = t[t.model == name]
        if not len(s):
            continue
        e = s.err.values.astype(float)
        ae = np.abs(e)
        print(f"{name:16s} {int(s.window.iloc[0]):5d} {int(s.n_win.iloc[0]):6d} "
              f"{s.concentration.mean():10.3f} | {ae.mean():9.2f} "
              f"{np.sqrt((e**2).mean()):10.2f} {np.median(ae):8.1f} {np.mean(ae<=10)*100:6.1f}% "
              f"{np.mean(ae<=20)*100:6.1f}% {np.mean(ae<=100)*100:6.1f}%")
        out.append(dict(model=name, window=int(s.window.iloc[0]), n=len(s),
                        concentration=float(s.concentration.mean()),
                        P_MAE=float(ae.mean()), P_RMSE=float(np.sqrt((e**2).mean())),
                        P_med=float(np.median(ae)), P_le10=float(np.mean(ae<=10)*100),
                        P_le20=float(np.mean(ae<=20)*100), P_le100=float(np.mean(ae<=100)*100)))
    pd.DataFrame(out).to_csv(os.path.join(HERE, 'eval_all',
                                          f'crew_vote_pick_summary{tag}.csv'), index=False)
    print('\n（已把 ≤100 一并落盘，补齐方案B CSV 缺列的问题）')
    print('已保存: eval_all/crew_vote_pick{.csv,_summary.csv}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""位置敏感度扫描：拾取器究竟在"定位 P"，还是在"输出窗中心先验"？

做法：取一段含 P 的波形，把窗起点摆在 p-pos（pos = P 在窗内的位置），跑拾取器，
      记录 误差 = 预测到时 - 真值。
      * 若 误差 ≈ (pos - pos_train)（即斜率≈1）-> 模型只是把 P 放在训练时的位置（位置先验）
      * 若 误差 ≈ 0（斜率≈0）-> 模型真的在定位 P（位置不变）

同时对每个模型用它自己的原生窗长，保证公平。
在 CREW（300 s 长记录、区域震）上做，因为这是锚点误差最大的场景。
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


# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=600)
    ap.add_argument('--train-pos', type=int, default=500, help='训练时 P 在窗内的位置')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    import torch
    dev = torch.device('cuda')
    models = {n: BUILDERS[n](dev) for n in MODELS}
    for n, m in models.items():
        print(f'  {n:16s} window={m.window}')

    rows = []
    with h5py.File(CREW, 'r') as f:
        ks = list(f['data'].keys())
        n_done = 0
        for k in ks:
            if n_done >= a.limit:
                break
            d = f['data'][k]
            x = np.asarray(d[()], dtype=np.float32)
            if x.shape[0] == 3 and x.shape[1] != 3:
                x = x.T
            x = np.ascontiguousarray(x[:, E.ENZ_TO_ZNE])
            T = x.shape[0]
            p = float(d.attrs.get('P_arrival_sample', np.nan))
            if not np.isfinite(p) or p <= 0 or p >= T:
                continue
            n_done += 1
            for name, m in models.items():
                W = m.window
                # 对每个模型按"标准位:扫到位"的相对比例设扫描点
                frac = np.array([0.10, 0.17, 0.27, 0.40, 0.60, 0.80])
                poss = np.unique(np.clip((frac * W).round().astype(int),
                                         100, W - 100))
                wins, meta = [], []
                for pos in poss:
                    st = int(p - pos)
                    if st < 0 or st + W > T:
                        continue
                    wins.append(x[st:st + W])
                    meta.append((int(pos), st))
                if not wins:
                    continue
                prep = getattr(m, 'prep', None)
                batch = np.stack([(prep(w) if prep else E.preprocess_40(w)) for w in wins])
                out = m.score(m.to_input(batch))
                chan = out['P'] if 'P' in out else out['det']
                am = np.asarray(chan).argmax(axis=1)
                for (pos, st), j in zip(meta, am):
                    rows.append(dict(model=name, window=W, pos=pos, pos_frac=pos / W,
                                     pred_abs=st + int(j), p_true=p,
                                     err=float(st + int(j) - p)))
    t = pd.DataFrame(rows)
    tag = f'_{a.tag}' if a.tag else ''
    t.to_csv(os.path.join(HERE, 'eval_all', f'position_sensitivity{tag}.csv'), index=False)

    print('\n' + '=' * 104)
    print('【位置敏感度扫描 · CREW】误差 = 预测 - 真值（单位：采样点）')
    print('  斜率≈0 => 真在定位 P；斜率≈1 => 只是输出"窗中心先验"（位置不变性缺失）')
    print('=' * 104)
    print(f"{'模型':16s} {'窗长':>5s} {'训练位':>6s} | " +
          ' '.join(f'{int(f*100):>6d}%' for f in (0.10, 0.17, 0.27, 0.40, 0.60, 0.80)) +
          f" | {'斜率':>7s} {'截距':>8s} {'|误差|中位':>10s}")
    out = []
    for name, m in models.items():
        s = t[t.model == name]
        if not len(s):
            continue
        fr = np.array([0.10, 0.17, 0.27, 0.40, 0.60, 0.80])
        cols = []
        for f0 in fr:
            sub = s[np.isclose(s.pos_frac, f0, atol=0.012)]
            cols.append(sub.err.median() if len(sub) else np.nan)
        ok = ~np.isnan(cols)
        if ok.sum() >= 2:
            slope, inter = np.polyfit(fr[ok] * m.window, np.array(cols)[ok], 1)
        else:
            slope, inter = np.nan, np.nan
        print(f'{name:16s} {m.window:5d} {a.train_pos:6d} | ' +
              ' '.join(f'{c:6.0f}' if not np.isnan(c) else '     -' for c in cols) +
              f' | {slope:7.3f} {inter:8.0f} {np.median(np.abs(s.err)):10.1f}')
        out.append(dict(model=name, window=m.window, train_pos=a.train_pos,
                        slope=float(slope), intercept=float(inter),
                        abs_med=float(np.median(np.abs(s.err))),
                        **{f'err_frac_{f0:.2f}': float(c) for f0, c in zip(fr, cols)}))
    pd.DataFrame(out).to_csv(os.path.join(HERE, 'eval_all',
                                          f'position_sensitivity_summary{tag}.csv'), index=False)
    print(f'\n每格 = 该位置下误差的中位数；n≈{a.limit} 条 × 每模型 6 个位置')
    print('已保存: eval_all/position_sensitivity{.csv,_summary.csv}')


if __name__ == '__main__':
    main()

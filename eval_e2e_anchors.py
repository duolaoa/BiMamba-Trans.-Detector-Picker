# -*- coding: utf-8 -*-
"""修正版端到端 B 判据：锚点用**检测曲线左边缘**（与部署流水线 a_edge 一致），
并同时给出 argmax 锚点作对比，说明锚点约定对结果的影响。

锚点定义：
  * left_edge : 逐点检测曲线 first index where curve >= thr   (部署约定)
  * argmax    : 曲线全局最大位置            (我上一版误用的约定)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_evsplit_models as E   # noqa: E402

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


OUT = os.path.join(HERE, 'eval_all')
TAG = os.environ.get('EVAL_TAG', '')
PRE, WIN = 500, 3000
DET_THR = 0.5
STRIDE = 1000


def prf(tp, fp, fn):
    P = tp / max(tp + fp, 1e-9); R = tp / max(tp + fn, 1e-9)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def curve_and_anchors(det, x, bs=32):
    """滑窗 + 逐点 max 聚合 -> 全长度检测曲线；返回 (score, a_argmax, a_edge)。"""
    T = x.shape[0]
    W = det.window
    starts = list(range(0, max(1, T - W + 1), STRIDE)) or [0]
    wins = np.stack([det.prep(x[s:s + W]) for s in starts])
    curve = np.zeros(T, np.float32)
    for i, s in enumerate(starts):          # 分块跑，避免一次性占显存
        pass
    outs = []
    for i in range(0, len(wins), bs):
        outs.append(det.score(det.to_input(wins[i:i + bs]))['det'])
    outs = np.concatenate(outs, axis=0)     # (n_win, W)
    for i, s in enumerate(starts):
        np.maximum(curve[s:s + W], outs[i], out=curve[s:s + W])
    score = float(curve.max())
    a_am = int(np.argmax(curve))
    idx = np.where(curve >= DET_THR)[0]
    a_le = int(idx[0]) if len(idx) else a_am
    return score, a_am, a_le


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)
    det = E.BiMambaDetector(device)
    picker = E.BiMambaPicker(device)
    rows = []
    for ds in ('INSTANCE', 'EQS-D-0dB', 'EQS-D-imp'):
        rec = []
        for split in ('events', 'noise'):
            for name, x, p, s in tqdm(E.DATASETS[ds](split, 0), desc=f'{ds}/{split}'):
                if x.shape[0] < det.window:
                    continue
                score, a_am, a_le = curve_and_anchors(det, x)
                item = dict(trace=str(name), split=split, score=score,
                            a_argmax=a_am, a_edge=a_le,
                            p_true=float(p) if np.isfinite(p) else np.nan)
                rec.append(item)
        d = pd.DataFrame(rec)
        # 两种锚点各跑一次拾取器
        for key in ('a_edge', 'a_argmax'):
            pack, meta = [], []
            for _, r in d.iterrows():
                # 需要波形 -> 再遍历一次数据（这里直接从上面缓存不好，简化：用下一次遍历）
                pass
        # 为省内存：改为按数据集再遍历一次，同时算两种锚点
        anchors = {r['trace']: (r['a_edge'], r['a_argmax'], r['score'], r['split'],
                                r['p_true']) for _, r in d.iterrows()}
        picks = {'a_edge': {}, 'a_argmax': {}}
        for split in ('events', 'noise'):
            pack = {'a_edge': [], 'a_argmax': []}
            meta = {'a_edge': [], 'a_argmax': []}
            for name, x, p, s in E.DATASETS[ds](split, 0):
                k = str(name)
                if k not in anchors:
                    continue
                a_le, a_am, score, sp, pt = anchors[k]
                T = x.shape[0]
                if T < WIN:
                    continue
                for key, anc in (('a_edge', a_le), ('a_argmax', a_am)):
                    st = int(np.clip(anc - PRE, 0, T - WIN))
                    pack[key].append((k, x[st:st + WIN], st))
                    meta[key].append((k, score, anc, pt, sp, st))
                    if len(pack[key]) >= 256:
                        picks[key].update(_run(picker, pack[key], meta[key]))
                        pack[key], meta[key] = [], []
            for key in ('a_edge', 'a_argmax'):
                if pack[key]:
                    picks[key].update(_run(picker, pack[key], meta[key]))

        # 汇总
        for key in ('a_edge', 'a_argmax'):
            pk = picks[key]
            ev = [(anchors[t][4], pk[t]) for t in pk
                  if anchors[t][3] == 'events' and np.isfinite(anchors[t][4])]
            nz = [anchors[t][2] for t in pk if anchors[t][3] == 'noise']
            err = np.abs(np.array([v for _, v in ev]) - np.array([t for t, _ in ev]))
            ths = np.unique(np.concatenate([[anchors[t][2] for t in pk if anchors[t][3] == 'events'],
                                            nz] if nz else [np.array([0.5])]))
            best = (0, 0, 0, 0)
            se = np.array([anchors[t][2] for t in pk if anchors[t][3] == 'events'])
            sn = np.array(nz) if nz else np.array([0.0])
            for th in ths:
                ke = se >= th
                P, R, F = prf(ke.sum(), (sn >= th).sum(), (~ke).sum())
                if F > best[2]:
                    best = (P, R, F, th)
            th = best[3]
            ke = se >= th
            ok = err <= 10
            kB = ke & ok
            B = prf(kB.sum(), (sn >= th).sum(), (~kB).sum())
            rows.append(dict(dataset=ds, anchor=key,
                             A_F1=best[2], A_P=best[0], A_R=best[1], A_thr=th,
                             B_F1=B[2], B_P=B[0], B_R=B[1],
                             e2e_med=float(np.median(err)),
                             e2e_ok10=float(np.mean(ok) * 100),
                             e2e_ok50=float(np.mean(err <= 50) * 100),
                             anchor_med=float(np.median([abs(anchors[t][0] - anchors[t][4])
                                                          for t in pk if anchors[t][3] == 'events'
                                                          and np.isfinite(anchors[t][4])]))
                             if key == 'a_edge' else
                             float(np.median([abs(anchors[t][1] - anchors[t][4])
                                              for t in pk if anchors[t][3] == 'events'
                                              and np.isfinite(anchors[t][4])]))))
            r = rows[-1]
            print(f"  {ds:12s} [{key:9s}] 锚点误差中位={r['anchor_med']:6.0f}  "
                  f"到时中位={r['e2e_med']:7.1f}  ±0.1s={r['e2e_ok10']:5.1f}%  "
                  f"±0.5s={r['e2e_ok50']:5.1f}%  A_F1={r['A_F1']:.4f}  B_F1={r['B_F1']:.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'summary_B_e2e_anchors{TAG}.csv'),
                                  index=False)
    print('\n保存:', os.path.join(OUT, f'summary_B_e2e_anchors{TAG}.csv'))


def _run(picker, pack, meta):
    wins = np.stack([E.preprocess_40(it[1]) for it in pack])
    out = picker.score(picker.to_input(wins))
    res = {}
    for i, (k, score, anc, pt, sp, st) in enumerate(meta):
        pk = st + int(np.argmax(out['P'][i]))
        res[k] = float(pk)
    return res


if __name__ == '__main__':
    main()

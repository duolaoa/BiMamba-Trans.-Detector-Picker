# -*- coding: utf-8 -*-
"""盘点：窗口级评估中各数据集的记录长度 T、窗口放置 st、锚点相对位置
   目的：回答"粗定位=元数据 P/S 位置"这一设定下，现有结果到底对应什么。"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_evsplit_models as E

W = 6000
print('=' * 88)
print(f'各数据集：T 分布与窗口放置 st = clip(round(p-500), 0, T-W)，W={W}')
print('=' * 88)

for ds in ('STEAD-DET', 'INSTANCE', 'EQS-D-0dB', 'EQS-D-imp'):
    Ts, sts, rel = [], [], []
    n = 0
    npi = 0
    for name, x, p, s in E.DATASETS[ds]('events', 0):
        T = x.shape[0]
        Ts.append(T)
        if np.isfinite(p):
            npi += 1
            if T >= W:
                st = max(0, min(int(round(p - 500)), T - W))
            else:
                st = 0
            sts.append(st)
            rel.append(p - st)
        n += 1
        if n >= 3000:
            break
    Ts = np.array(Ts); sts = np.array(sts); rel = np.array(rel)
    print(f'\n■ {ds}  样本 {n}（有 P 标签 {npi}）')
    print(f'   T: min={Ts.min()}  中位={int(np.median(Ts))}  max={Ts.max()}  '
          f'唯一值个数={len(np.unique(Ts))}')
    if len(sts):
        print(f'   st: min={sts.min()}  中位={int(np.median(sts))}  max={sts.max()}  '
              f'st==0 的比例={100*(sts==0).mean():.1f}%')
        print(f'   锚点相对位置 p-st: min={rel.min():.0f}  中位={np.median(rel):.0f}  '
              f'max={rel.max():.0f}   (r = 中位/W = {np.median(rel)/W:.4f})')
        print(f'   p-st 的 P5/P95 = {np.percentile(rel,5):.0f} / {np.percentile(rel,95):.0f}')

# -*- coding: utf-8 -*-
"""校验事件级划分表，并量化"按 trace 划分"造成的同事件泄漏（供论文 Methods 引用）。

用法:
    python verify_split.py --split-table event_split_table.csv
    python verify_split.py --split-table event_split_table.csv --data-dir <dir> \
        --chunks chunk2 chunk3 --compare-trace-split --seed 42
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SPLITS = ('train', 'val', 'test')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-table', required=True)
    ap.add_argument('--data-dir',
                    default=os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'STEAD_data', '1Dsignal'))
    ap.add_argument('--chunks', nargs='+', default=['chunk2', 'chunk3'])
    ap.add_argument('--compare-trace-split', action='store_true',
                    help='再算一遍"按 trace 随机划分"的泄漏率做对照')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(argv)

    df = pd.read_csv(a.split_table, dtype={'trace_name': str, 'event_id': str})
    for c in ('trace_name', 'event_id', 'split'):
        if c not in df.columns:
            raise SystemExit(f'划分表缺少列 {c}')

    print('=' * 78)
    print('事件级划分校验:', a.split_table)
    print('=' * 78)
    print('%-6s %10s %8s %10s %10s' % ('split', 'traces', 'trace%', 'events',
                                       'rec/event'))
    n = len(df)
    for s in SPLITS:
        m = df.split == s
        ne = df.loc[m, 'event_id'].nunique()
        print('%-6s %10d %7.1f%% %10d %10.2f'
              % (s, int(m.sum()), m.mean() * 100, ne, m.sum() / max(ne, 1)))

    ev = {s: set(df.loc[df.split == s, 'event_id']) for s in SPLITS}
    tn = {s: set(df.loc[df.split == s, 'trace_name']) for s in SPLITS}
    print('\n集合间共享的事件数 (必须全为 0):')
    ok = True
    for i, x in enumerate(SPLITS):
        for y in SPLITS[i + 1:]:
            k, v = f'{x}&{y}', len(ev[x] & ev[y])
            vt = len(tn[x] & tn[y])
            ok &= (v == 0 and vt == 0)
            print('   %-12s events=%d  traces=%d' % (k, v, vt))
    print('   =>', '通过' if ok else '*** 不通过：仍有事件跨集合 ***')

    # 同一个事件在划分表里只应有一个 split
    bad = df.groupby('event_id')['split'].nunique()
    print('  事件被分到多个 split 的个数: %d' % int((bad > 1).sum()))
    print('  重复 trace_name 个数: %d' % int(df.trace_name.duplicated().sum()))

    if a.compare_trace_split:
        print('\n' + '=' * 78)
        print('对照：按 trace 随机划分（原实现）的同事件泄漏率')
        print('=' * 78)
        from sklearn.model_selection import train_test_split
        frames = []
        for c in a.chunks:
            csv = os.path.join(a.data_dir, f'{c}.csv')
            if not os.path.exists(csv):
                print(f'  跳过 {csv}（不存在）'); continue
            d = pd.read_csv(csv, usecols=['trace_name', 'source_id',
                                          'source_origin_time'],
                            low_memory=False)
            d['chunk'] = c
            frames.append(d)
        if frames:
            m = pd.concat(frames, ignore_index=True)
            m['event_id'] = (m['source_id'].astype(str) + '|'
                             + m['source_origin_time'].astype(str))
            idx = np.arange(len(m))
            tr, te = train_test_split(idx, test_size=0.2, random_state=a.seed)
            tr, va = train_test_split(tr, test_size=0.2, random_state=a.seed)
            ev_tr = set(m.event_id.values[tr])
            for nm, ix in (('test', te), ('val', va)):
                e = m.event_id.values[ix]
                frac = np.mean([x in ev_tr for x in e]) * 100
                print('  %-5s 中「事件也出现在训练集」的 trace 比例 = %.1f%%'
                      % (nm, frac))
            print('  （这就是审稿人担心的泄漏；事件级划分把它降到 0%）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

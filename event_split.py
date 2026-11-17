# -*- coding: utf-8 -*-
"""事件级（event-level）数据划分工具 —— 回应审稿人 "Please split by event"。

背景
----
STEAD 是按 trace 分块存储的（chunk1..chunk5）。原始训练脚本用
``train_test_split`` 按 **trace** 随机划分，而同一个地震事件会以多条记录
（不同台站/不同 chunk）出现，于是同一事件会同时落进训练集和测试集。在
chunk2+chunk3（40 万条、19.2 万个事件）上实测：**65.1% 的测试 trace 其事件
也出现在训练集中**。本模块按事件划分，使三个集合的事件集合两两不相交。

事件标识
--------
优先用 ``source_id``，并联合 ``source_origin_time`` 组成组合键（同一 source_id
偶尔对应不同发震时刻；且 chunk2 与 chunk3 之间有 39,420 个 source_id 重叠，
必须**跨 chunk 合并后再分组**）。若某列不存在则自动降级。

用法
----
    from event_split import (load_chunks, build_split_table, names_for_split,
                             pool_names, describe_splits, assert_disjoint)

    meta = load_chunks(DATA_DIR, ['chunk2', 'chunk3'])
    meta = build_split_table(meta, ratios=(0.8, 0.1, 0.1), seed=42)
    print(describe_splits(meta)); assert_disjoint(meta)
    tr = names_for_split(meta, 'train', 'chunk2', max_samples=50000, seed=42)

命令行自检：
    python event_split.py --data-dir <dir> --chunks chunk2 chunk3 --seed 42
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

GROUP_KEYS = ('source_id', 'source_origin_time')
SPLITS = ('train', 'val', 'test')


def _event_key(df: pd.DataFrame, group_keys: Sequence[str] = GROUP_KEYS) -> pd.Series:
    """构造事件标识；给缺失列自动降级。"""
    use = [k for k in group_keys if k in df.columns]
    if not use:
        # 兜底：STEAD 的 trace_name 形如 NET.STA_YYYYMMDDHHMMSS_EV，取时间戳
        tn = df['trace_name'].astype(str)
        return tn.str.extract(r'_(\d{14})_', expand=False).fillna(tn)
    key = df[use[0]].astype(str)
    for k in use[1:]:
        key = key + '|' + df[k].astype(str)
    # source_id 为 NaN 的行（例如纯噪声 chunk1）单独成组
    if use[0] in df.columns:
        bad = df[use[0]].isna()
        if bad.any():
            key = key.where(~bad, 'NOISE|' + df['trace_name'].astype(str))
    return key


def _chunk_label(name: str) -> str:
    """chunk 标签统一取 basename（去掉目录前缀与扩展名）。

    这样 'chunk2' / 'autodl-tmp/chunk2' / '/abs/path/chunk2.hdf5' 都归一成 'chunk2'，
    划分表与各脚本的路径写法无关，可安全共用。
    """
    b = os.path.basename(str(name))
    for ext in ('.csv', '.hdf5', '.h5'):
        if b.endswith(ext):
            b = b[:-len(ext)]
            break
    return b


def load_chunks(data_dir: str, chunks: Sequence[str],
                group_keys: Sequence[str] = GROUP_KEYS,
                only_earthquake: bool = True) -> pd.DataFrame:
    """把若干 chunk 的元数据合并成一张表，并加上 ``event_id`` / ``chunk`` 列。"""
    frames = []
    for c in chunks:
        csv = os.path.join(data_dir, f'{c}.csv')
        if not os.path.exists(csv):
            raise FileNotFoundError(f'找不到 {csv}')
        df = pd.read_csv(csv, low_memory=False)
        if only_earthquake and 'trace_category' in df.columns:
            n0 = len(df)
            df = df[df['trace_category'] == 'earthquake_local']
            print(f'  {c}: {n0} -> {len(df)} 条 (earthquake_local)')
        keep = ['trace_name'] + [k for k in group_keys if k in df.columns]
        df = df[keep].copy()
        df['chunk'] = _chunk_label(c)
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta['trace_name'] = meta['trace_name'].astype(str)
    meta['event_id'] = _event_key(meta, group_keys).astype(str)
    return meta


def build_split_table(meta: pd.DataFrame, ratios: Sequence[float] = (0.8, 0.1, 0.1),
                      seed: int = 42,
                      group_keys: Sequence[str] = GROUP_KEYS) -> pd.DataFrame:
    """按事件划分 train/val/test。

    做法：把事件随机打乱后，按 **trace 配额** 贪心填充 val/test（各占
    ``ratios[1]``/``ratios[2]``），其余进 train。这样既保证事件不跨集合，
    又让三个集合的 trace 数接近所需比例。
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f'ratios 必须和为 1，收到 {ratios}')
    if 'event_id' not in meta.columns:
        meta = meta.copy()
        meta['event_id'] = _event_key(meta, group_keys).astype(str)

    sizes = meta.groupby('event_id').size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.values)
    n = len(meta)
    quota = {'test': ratios[2] * n, 'val': ratios[1] * n}
    acc = {'test': 0, 'val': 0}
    assign: Dict[str, str] = {}
    for e in order:                      # 先填 test、再填 val，其余 train
        for s in ('test', 'val'):
            if acc[s] < quota[s]:
                assign[e] = s
                acc[s] += int(sizes[e])
                break
        else:
            assign[e] = 'train'
    out = meta.copy()
    out['split'] = out['event_id'].map(assign)
    return out


def split_caps(total_samples: int, ratios: Sequence[float] = (0.8, 0.15, 0.05),
               n_chunks: int = 2) -> Dict[str, int]:
    """把"总数据量"按比例折算成**每个 chunk、每个 split** 的 trace 上限。

    例：total_samples=100000, ratios=(0.8,0.15,0.05), n_chunks=2
        -> {'train': 40000, 'val': 7500, 'test': 2500, 'total': 100000}

    返回值里的 'total' 是折算回来的实际总量（= sum(cap * n_chunks)），
    方便在日志里核对"总量是否仍为 100000"。
    """
    import math as _math
    names = ('train', 'val', 'test')
    out: Dict[str, int] = {}
    for nm, r in zip(names, ratios):
        target = int(round(float(total_samples) * float(r)))
        out[nm] = int(_math.ceil(target / max(int(n_chunks), 1)))
    out['total'] = sum(out[nm] * max(int(n_chunks), 1) for nm in names)
    return out


def events_in(meta: pd.DataFrame, split: str) -> Set[str]:
    return set(meta.loc[meta.split == split, 'event_id'].astype(str))


def names_for_split(meta: pd.DataFrame, split: str, chunk: str,
                    max_samples: Optional[int] = None,
                    seed: int = 42) -> List[str]:
    """取某个 chunk 在某个 split 里的 trace_name（可选限量随机抽取）。

    注意：**先划分事件、再抽样**，否则抽样会改变各集合的事件构成。
    """
    sub = meta[(meta.split == split) & (meta.chunk == _chunk_label(chunk))]
    names = sub['trace_name'].tolist()
    if max_samples and len(names) > max_samples:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(names), size=int(max_samples), replace=False)
        names = [names[i] for i in sorted(pick)]
    return names


def pool_names(pool_meta: pd.DataFrame, exclude_events: Iterable[str],
               max_samples: Optional[int] = None, seed: int = 42) -> List[str]:
    """波形库（第二事件叠加增强）用的 trace：排除掉 val/test 的事件。

    否则测试集事件会通过增强"漏"进训练波形。
    """
    ex = set(map(str, exclude_events))
    sub = pool_meta[~pool_meta['event_id'].astype(str).isin(ex)]
    names = sub['trace_name'].tolist()
    if max_samples and len(names) > max_samples:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(names), size=int(max_samples), replace=False)
        names = [names[i] for i in sorted(pick)]
    return names


def describe_splits(meta: pd.DataFrame) -> str:
    lines = ['%-6s %10s %8s %10s %10s' % ('split', 'traces', 'trace%',
                                          'events', 'rec/event')]
    n = len(meta)
    for s in SPLITS:
        m = meta.split == s
        ne = meta.loc[m, 'event_id'].nunique()
        lines.append('%-6s %10d %7.1f%% %10d %10.2f'
                     % (s, int(m.sum()), m.mean() * 100, ne,
                        m.sum() / max(ne, 1)))
    lines.append('%-6s %10d %7.1f%% %10d' % ('total', n, 100.0,
                                             meta.event_id.nunique()))
    return '\n'.join(lines)


def assert_disjoint(meta: pd.DataFrame) -> Dict[str, int]:
    """校验三个集合的事件两两不相交；有交集直接抛错。"""
    ev = {s: events_in(meta, s) for s in SPLITS}
    ov = {f'{a}&{b}': len(ev[a] & ev[b])
          for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    trace_ov = {}
    tn = {s: set(meta.loc[meta.split == s, 'trace_name']) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            trace_ov[f'{a}&{b}'] = len(tn[a] & tn[b])
    bad = {k: v for k, v in ov.items() if v}
    if bad:
        raise AssertionError(f'事件级划分失败，集合间仍有共享事件: {bad}')
    if any(trace_ov.values()):
        raise AssertionError(f'trace 也出现跨集合重复: {trace_ov}')
    print('事件级划分校验通过：train/val/test 的事件与 trace 两两不相交')
    return ov


def save_split_table(meta: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta[['trace_name', 'event_id', 'chunk', 'split']].to_csv(path, index=False)
    print(f'划分表已保存: {path}')


def load_split_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'trace_name': str, 'event_id': str})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='事件级划分自检')
    ap.add_argument('--data-dir',
                    default=os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'STEAD_data', '1Dsignal'))
    ap.add_argument('--chunks', nargs='+', default=['chunk2', 'chunk3'])
    ap.add_argument('--pool-chunk', default='chunk4')
    ap.add_argument('--ratios', default='0.8,0.1,0.1')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='event_split_table.csv')
    a = ap.parse_args(argv)

    ratios = tuple(float(x) for x in a.ratios.split(','))
    print('加载元数据 ...')
    meta = load_chunks(a.data_dir, a.chunks)
    meta = build_split_table(meta, ratios, a.seed)
    print('\n' + describe_splits(meta))
    assert_disjoint(meta)
    ex = events_in(meta, 'val') | events_in(meta, 'test')
    pm = load_chunks(a.data_dir, [a.pool_chunk])
    pn_all = len(pm)
    pn_ok = len(pool_names(pm, ex))
    print(f'\n波形库 {a.pool_chunk}: {pn_all} 条 -> 排除 val/test 事件后 {pn_ok} 条')
    save_split_table(meta, a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

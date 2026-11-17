"""
PhaseNet Baseline Training Script
- 数据集划分方式与 BiLSTM_Linear.py 保持一致（多 chunk hdf5/csv，随机采样，train/val/test = 64/16/20）
- 输入：三分量波形 (E, N, Z)，长度 3000 采样点（原版 X_shape=[3000,1,3]）
- 输出：三通道逐点概率图 (P相, S相, 噪声)，形状 (batch, 3000, 3)
- 损失函数：原版 PhaseNet 交叉熵 (mean cross-entropy over time steps)
- 不使用纯噪声样本
- 模型架构严格对应原版 TF 实现（AI4EPS/PhaseNet，model.py）：
    depths=5, filters_root=8, kernel_size=7, pool_size=4
    每个 DownConv/UpConv 仅有单层卷积
    参数量 ~20 万
"""

import os
import glob
import datetime
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt, detrend
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm, gaussian_kde
from scipy import stats

# =========================================================================== #
# 事件级划分（回应审稿人：Please split by event）
# --------------------------------------------------------------------------- #
# 原实现用 train_test_split 按 trace 随机划分。STEAD 中同一地震事件会以多条记录
# （不同台站/不同 chunk）出现，实测在 chunk2+chunk3（40 万条、19.2 万个事件）上
# **65.1% 的测试 trace 其事件也出现在训练集中**。这里改为按事件划分：先把事件随机
# 分配到 train/val/test，再在各集合内部抽样；波形库也排除 val/test 的事件，避免
# 测试事件经"第二事件叠加"增强泄漏进训练波形。
# 三个模型用同一个 --split-table 可保证划分完全一致。
# =========================================================================== #
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
from event_split import (assert_disjoint, build_split_table, describe_splits,
                         events_in, load_chunks, load_split_table,
                         names_for_split, pool_names, save_split_table,
                         split_caps)

_1DSIGNAL = os.environ.get(
    'STEAD_DIR', os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'STEAD_data', '1Dsignal'))
CFG = {
    'data_dir': _1DSIGNAL,
    'train_chunks': ['chunk2', 'chunk3'],
    'pool_chunk': 'chunk4',
    # PhaseNet 是检测+拾取一体模型，默认加载 chunk1 纯噪声
    'noise_chunks': ['chunk1'],
    # 事件级划分比例：train/val/test = 0.8/0.15/0.05
    'ratios': (0.8, 0.15, 0.05),
    # 信号总量；由 split_caps() 按上面的比例折算成每个 chunk 的上限
    'total_samples': 100000,
    # 每个 chunk、每个 split 的 trace 上限。默认值刻意与原文训练量对齐：
    # 原文每 chunk 抽 50000 条（2 chunk = 100k）再按 trace 划分，训练集约 64k；
    # 现在按事件划分，train 占 80%，故每 chunk 取 32000 -> 训练集 32000*2*0.8*2
    # ≈ 64k（实际受各集合可用条数限制），保证"只改划分、不改数据量"。
    'max_samples': 32000,
    'val_samples': 5000,
    'test_samples': 5000,
    'pool_samples': 15000,
    'noise_samples': 20000,
    'split_table': '',         # 复用已有划分表 -> 三个模型完全同一划分
    'save_split_table': 'event_split_table.csv',
    'dry_run': False,          # 只装配数据、打印划分与形状，然后退出
}


def apply_cli(argv=None):
    """命令行覆盖 CFG，返回 seed。"""
    import argparse
    ap = argparse.ArgumentParser(description='事件级划分训练脚本（event-level split）')
    ap.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    ap.add_argument('--tag', default='', help='权重名后缀')
    ap.add_argument('--window-size', type=int, default=WINDOW_SIZE,
                    help='窗长（采样点）；默认 3000')
    ap.add_argument('--data-dir', default=CFG['data_dir'])
    ap.add_argument('--train-chunks', nargs='+', default=CFG['train_chunks'])
    ap.add_argument('--pool-chunk', default=CFG['pool_chunk'])
    ap.add_argument('--noise-chunks', nargs='*', default=CFG['noise_chunks'])
    ap.add_argument('--ratios', default='0.8,0.15,0.05')
    ap.add_argument('--total-samples', type=int, default=100000,
                    help='信号总量，按 --ratios 拆分；0 表示改用 '
                         '--max-samples/--val-samples/--test-samples')
    ap.add_argument('--max-samples', type=int, default=CFG['max_samples'])
    ap.add_argument('--val-samples', type=int, default=CFG['val_samples'])
    ap.add_argument('--test-samples', type=int, default=CFG['test_samples'])
    ap.add_argument('--pool-samples', type=int, default=CFG['pool_samples'])
    ap.add_argument('--noise-samples', type=int, default=CFG['noise_samples'],
                    help='噪声**总量**（默认 20000），按 --ratios 划分成 '
                         '16000/3000/1000')
    ap.add_argument('--split-table', default=CFG['split_table'],
                    help='复用已有划分表（多模型共用同一划分）')
    ap.add_argument('--save-split-table', default=CFG['save_split_table'])
    ap.add_argument('--dry-run', action='store_true',
                    help='只装配数据并打印划分/形状，不训练')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(argv)
    globals()['NUM_EPOCHS'] = int(a.epochs)
    if a.tag:
        globals()['MODEL_SAVE'] = f'best_phasenet_baseline{a.tag}.pth'
    print(f'[配置] epochs={NUM_EPOCHS}  权重输出={MODEL_SAVE}')
    globals()['WINDOW_SIZE'] = int(a.window_size)   # 让各调用点的 WINDOW_SIZE 生效
    print(f'[窗长] PhaseNet window_size = {WINDOW_SIZE} 采样点 ({WINDOW_SIZE/100:.1f}s)')
    CFG.update(data_dir=a.data_dir, train_chunks=a.train_chunks,
               pool_chunk=a.pool_chunk, noise_chunks=a.noise_chunks,
               ratios=tuple(float(x) for x in a.ratios.split(',')),
               max_samples=a.max_samples, val_samples=a.val_samples,
               test_samples=a.test_samples, pool_samples=a.pool_samples,
               noise_samples=a.noise_samples, split_table=a.split_table,
               save_split_table=a.save_split_table,
               total_samples=a.total_samples,
               dry_run=bool(a.dry_run))
    print('[事件级划分] data_dir=%s train_chunks=%s pool=%s ratios=%s '
          'max_samples=%s split_table=%r'
          % (CFG['data_dir'], CFG['train_chunks'], CFG['pool_chunk'],
             CFG['ratios'], CFG['max_samples'], CFG['split_table']))
    return a.seed


def _event_meta(seed):
    """返回 (meta, exclude_events)：优先复用划分表，保证多模型同一划分。"""
    reused = bool(CFG['split_table']) and os.path.exists(CFG['split_table'])
    if reused:
        print('使用已有事件划分表: %s' % CFG['split_table'])
        meta = load_split_table(CFG['split_table'])
    else:
        print('构建事件级划分 ...')
        meta = load_chunks(_data_dir(), CFG['train_chunks'])
        meta = build_split_table(meta, CFG['ratios'], seed)
    print('\n' + describe_splits(meta))
    assert_disjoint(meta)
    if CFG['save_split_table'] and not reused:
        save_split_table(meta, CFG['save_split_table'])
    return meta, (events_in(meta, 'val') | events_in(meta, 'test'))


def _dry_report(Xtr, ptr, str_, Xva, pva, sva, Xte, pte, ste, pool, meta):
    """--dry-run：打印划分与张量形状，确认无误后再正式训练。"""
    import numpy as _np
    print('\n================ dry-run 汇总 ================')
    print(describe_splits(meta))
    assert_disjoint(meta)
    for nm, X in (('train', Xtr), ('val', Xva), ('test', Xte)):
        print('  %-5s wave=%s' % (nm, _np.asarray(X).shape))
    if pool is not None:
        print('  pool  wave=%s  (已排除 val/test 事件)' % (_np.asarray(pool).shape,))
    else:
        print('  pool  未加载')
    print('提示：正式训练请复用同一划分表 --split-table event_split_table.csv，'
          '以保证三个模型划分完全一致。')
    print('=============================================\n')


def _data_dir():
    """data_dir 为相对路径时按脚本所在目录解析，避免受当前工作目录影响。"""
    d = CFG['data_dir']
    return d if os.path.isabs(d) else os.path.normpath(os.path.join(_HERE, d))


def _paths(chunk):
    return (os.path.join(_data_dir(), chunk + '.hdf5'),
            os.path.join(_data_dir(), chunk + '.csv'))

warnings.filterwarnings('ignore')

# ==================== 超参数 ====================
# 通道序统一为 Z,N,E（STEAD 原始 h5 列序是 E,N,Z）
CH_Z, CH_N, CH_E = 2, 1, 0
CHANNEL_INDICES  = [CH_Z, CH_N, CH_E]
WINDOW_SIZE  = 3000    # 原版 PhaseNet X_shape=[3000,1,3]
SIGMA        = 10      # 高斯标签标准差（采样点）
BATCH_SIZE   = 64
NUM_EPOCHS   = 100
LR           = 1e-3    # 统一为 1e-3（原版 PhaseNet 用 0.01；回退改回 1e-2 即可）
MAX_SAMPLES  = 300    # 每个 chunk 最大采样数（与原文件一致）
MODEL_SAVE   = 'best_phasenet_baseline.pth'

# ==================== 预处理 ====================
def preprocess_3c(data3c, fs=100):
    """
    三分量预处理：去趋势 + 1-40Hz 带通 + 逐通道标准化。
    data3c: (T, 3) 或 (3, T) → 返回 (T, 3) float32
    """
    if data3c.shape[0] == 3 and data3c.shape[1] != 3:
        data3c = data3c.T
    out = np.zeros_like(data3c, dtype=np.float32)
    nyq = fs / 2.0
    low, high = 1.0 / nyq, min(40.0 / nyq, 0.99)
    b, a = butter(2, [low, high], btype='band')
    for ch in range(data3c.shape[1]):
        x = detrend(data3c[:, ch].astype(np.float64), type='linear')
        x = filtfilt(b, a, x)
        std = np.std(x)
        if std > 0:
            x /= std
        out[:, ch] = x.astype(np.float32)
    return out  # (T, 3)


# ==================== 高斯标签生成 ====================
def make_label(p_idx, s_idx, length, sigma=SIGMA):
    """
    三通道软标签 (length, 3)：[P_prob, S_prob, noise_prob]
    p_idx / s_idx 为采样点索引，-1 表示不存在。
    三通道归一化后和为 1（对应原版 softmax 输出）。
    """
    t = np.arange(length, dtype=np.float32)
    p_g = (np.exp(-0.5 * ((t - p_idx) / sigma) ** 2)
           if 0 <= p_idx < length else np.zeros(length, np.float32))
    s_g = (np.exp(-0.5 * ((t - s_idx) / sigma) ** 2)
           if 0 <= s_idx < length else np.zeros(length, np.float32))
    noise = np.clip(1.0 - p_g - s_g, 0.0, 1.0)
    label = np.stack([p_g, s_g, noise], axis=-1)           # (T, 3)
    label /= label.sum(axis=-1, keepdims=True) + 1e-8      # 归一化
    return label.astype(np.float32)


# ==================== 数据加载 ====================
def load_noise_data(hdf5_file, csv_file, window_size=WINDOW_SIZE,
                    max_samples=MAX_SAMPLES, seed=42):
    """
    从 STEAD chunk1（纯噪声）加载噪声波形。

    PhaseNet 是「检测 + 拾取」一体模型，输出三通道 [P, S, noise]；必须见过
    纯噪声才能学会 noise 通道、把虚检压下去。原脚本没有加载噪声，这里补上
    （不想要噪声可传 --noise-chunks ""）。

    返回:
        waves:  ndarray (M, window_size, 3)
        labels: ndarray (M, window_size, 3) — P/S 全 0，noise 通道全 1
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}][Noise] Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Original rows: {len(df)}")

    if 'trace_category' in df.columns:
        sub = df[df['trace_category'] == 'noise']
        if len(sub) > 0:
            df = sub.copy()
        print(f"  After noise filter: {len(df)}")

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"  Sampled: {max_samples} (seed={seed})")

    ev_list = df['trace_name'].tolist()
    label_noise = make_label(-1, -1, window_size)      # [0, 0, 1]
    waves_list, labels_list = [], []
    rng = np.random.default_rng(seed)
    err = 0

    try:
        with h5py.File(hdf5_file, 'r') as f:
            if 'data' not in f:
                print(f"[Noise] ERROR: No 'data' group in {hdf5_file}")
                return np.array([]), np.array([])

            for name in tqdm(ev_list,
                             desc=f"  [Noise] {os.path.basename(hdf5_file)}"):
                try:
                    dset = f.get('data/' + str(name))
                    if dset is None:
                        err += 1; continue

                    data = np.array(dset)
                    if data.ndim != 2:
                        err += 1; continue
                    if data.shape[0] == 3 and data.shape[1] != 3:
                        data = data.T
                    if data.shape[1] < 3:
                        err += 1; continue

                    T = data.shape[0]
                    if T < window_size:
                        err += 1; continue

                    # 在长记录里随机取窗，噪声样本更有代表性
                    start = int(rng.integers(0, T - window_size + 1))
                    win = data[start:start + window_size, :][:, CHANNEL_INDICES]
                    win = preprocess_3c(win)
                    if np.isnan(win).any() or np.isinf(win).any():
                        err += 1; continue

                    waves_list.append(win.astype(np.float32))
                    labels_list.append(label_noise.copy())
                except Exception as e:
                    err += 1
                    if err < 10:
                        print(f"  [Noise] Warning [{name}]: {e}")
                    continue
    except Exception as e:
        print(f"[Noise] ERROR opening {hdf5_file}: {e}")
        return np.array([]), np.array([])

    print(f"  [Noise] Loaded {len(waves_list)} samples | errors: {err}")
    if not waves_list:
        return np.array([]), np.array([])
    return (np.stack(waves_list).astype(np.float32),
            np.stack(labels_list).astype(np.float32))


def load_phasenet_data(hdf5_file, csv_file, window_size=WINDOW_SIZE, max_samples=MAX_SAMPLES,
                       seed=42, trace_names=None):
    """
    从 STEAD 格式 hdf5/csv 加载三分量波形及 P/S 到时。
    不加载纯噪声（p_arrival_sample <= 0 的样本跳过）。
    返回:
        waves:  ndarray (N, window_size, 3)
        labels: ndarray (N, window_size, 3)
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Original rows: {len(df)}")

    # 仅保留地震样本（排除纯噪声）
    if 'trace_category' in df.columns:
        df = df[df['trace_category'] == 'earthquake_local'].copy()
        print(f"  After earthquake filter: {len(df)}")

    if trace_names is not None:
        # 事件级划分：只加载指定 trace
        want = [str(t) for t in trace_names]
        df['trace_name'] = df['trace_name'].astype(str)
        df = df.set_index('trace_name').reindex(want).dropna(
            how='all').reset_index()
        print(f"  Event-split trace list: {len(df)} rows")
    elif len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"  Sampled: {max_samples} (seed={seed})")

    ev_list = df['trace_name'].tolist()
    waves_list, labels_list = [], []

    try:
        with h5py.File(hdf5_file, 'r') as f:
            if 'data' not in f:
                print("ERROR: No 'data' group in HDF5.")
                return np.array([]), np.array([])

            err = 0
            for name in tqdm(ev_list, desc=f"  {os.path.basename(hdf5_file)}"):
                try:
                    dset = f.get('data/' + str(name))
                    if dset is None:
                        err += 1; continue

                    data = np.array(dset)

                    # 形状处理 → (T, 3)
                    if data.ndim == 1:
                        err += 1; continue
                    if data.shape[0] == 3 and data.shape[1] != 3:
                        data = data.T
                    if data.shape[1] < 3:
                        err += 1; continue

                    T = data.shape[0]
                    if T < window_size:
                        err += 1; continue

                    # P/S 到时
                    p_arrival = int(dset.attrs.get('p_arrival_sample', -1))
                    s_arrival = int(dset.attrs.get('s_arrival_sample', -1))

                    # 排除纯噪声：P 到时必须有效
                    if p_arrival <= 0 or p_arrival >= T:
                        continue

                    # 固定取前 window_size 点（与原文件窗口策略一致）
                    win = data[:window_size, CHANNEL_INDICES]   # (T, 3) 统一为 Z,N,E

                    if p_arrival >= window_size:
                        continue
                    s_idx = s_arrival if (0 <= s_arrival < window_size) else -1

                    win = preprocess_3c(win)
                    if np.isnan(win).any() or np.isinf(win).any():
                        err += 1; continue

                    label = make_label(p_arrival, s_idx, window_size)
                    waves_list.append(win)
                    labels_list.append(label)

                except Exception as e:
                    err += 1
                    if err < 10:
                        print(f"  Warning [{name}]: {e}")
                    continue

            print(f"  Loaded {len(waves_list)} samples | errors: {err}")

    except Exception as e:
        print(f"ERROR opening {hdf5_file}: {e}")
        return np.array([]), np.array([])

    if len(waves_list) == 0:
        return np.array([]), np.array([])

    return (np.array(waves_list, dtype=np.float32),
            np.array(labels_list, dtype=np.float32))


# ==================== 数据增强（来自 BiLSTM_Linear_3C_SIMP.py）====================
class SimpleAugmentation:
    """
    三分量精简数据增强，作用于波形数据 x (T, 3)。

    参数
    ----
    noise_std_range : (float, float)
        高斯噪声 std 的对数均匀采样范围。
        std=0.01 ≈ SNR 40dB，std=0.30 ≈ SNR 10dB（信号标准化后 RMS≈1）。
    scale_range : (float, float)
        幅度缩放因子范围，对数均匀采样。
    noise_prob : float
        施加高斯噪声的概率。
    scale_prob : float
        施加幅度缩放的概率。
    dropout_prob : float
        施加分量 Dropout 的概率。
    dropout_n_range : (int, int)
        每次随机 Dropout 的分量数范围（闭区间），默认 (1, 2)。
    second_event_prob : float
        叠加第二地震事件的概率，默认 0.30（与 EQTransformer 一致）。
    second_event_pool : np.ndarray or None
        用于采样第二事件的波形库，shape (M, T, 3)，与主样本维度相同。
        传入 None 时退化为叠加随机带限噪声。
    second_event_amp_range : (float, float)
        第二事件的相对幅度范围（相对于主信号 RMS），对数均匀采样。
        默认 (0.1, 1.0)：第二事件幅度为主信号的 10%–100%。
    """

    def __init__(
        self,
        noise_std_range=(0.01, 0.30),
        scale_range=(0.5, 2.0),
        noise_prob=0.70,
        scale_prob=0.50,
        dropout_prob=0.30,
        dropout_n_range=(1, 2),
        second_event_prob=0.30,
        second_event_pool=None,
        second_event_amp_range=(0.1, 1.0),
    ):
        self.noise_std_range       = noise_std_range
        self.scale_range           = scale_range
        self.noise_prob            = noise_prob
        self.scale_prob            = scale_prob
        self.dropout_prob          = dropout_prob
        self.dropout_n_range       = dropout_n_range
        self.second_event_prob     = second_event_prob
        self.second_event_pool     = second_event_pool   # (M, T, 3) or None
        self.second_event_amp_range = second_event_amp_range

    def __call__(self, wave):
        """
        wave : (T, 3) float32
        返回增强后的 (T, 3) float32
        """
        x   = wave.copy()
        rng = np.random.default_rng()
        T   = x.shape[0]

        # ── 1. 随机高斯噪声 ─────────────────
        if rng.random() < self.noise_prob:
            log_lo = np.log(self.noise_std_range[0])
            log_hi = np.log(self.noise_std_range[1])
            std = np.exp(rng.uniform(log_lo, log_hi))
            x += rng.standard_normal(x.shape).astype(np.float32) * std

        # ── 2. 随机幅度缩放 ─────────────────
        if rng.random() < self.scale_prob:
            scale = np.exp(rng.uniform(
                np.log(self.scale_range[0]),
                np.log(self.scale_range[1])
            ))
            x *= scale

        # ── 3. 随机分量 Dropout ──────────────
        if rng.random() < self.dropout_prob:
            n_drop = int(rng.integers(self.dropout_n_range[0],
                                      self.dropout_n_range[1] + 1))
            drop_chs = rng.choice(3, size=n_drop, replace=False)
            for ch in drop_chs:
                x[:, ch] = 0.0

        # ── 4. 第二地震事件叠加 ─────────────
        if rng.random() < self.second_event_prob:
            if self.second_event_pool is not None:
                # 从真实波形库中随机取一条
                idx2 = int(rng.integers(0, len(self.second_event_pool)))
                x2 = self.second_event_pool[idx2].copy()   # (T, 3)
                # 若长度不匹配则随机裁剪/填充对齐
                if x2.shape[0] != T:
                    if x2.shape[0] > T:
                        st = int(rng.integers(0, x2.shape[0] - T))
                        x2 = x2[st: st + T]
                    else:
                        pad = T - x2.shape[0]
                        x2 = np.pad(x2, ((0, pad), (0, 0)), mode='constant')
            else:
                # 无波形库时：用带限高斯噪声合成伪地震信号
                x2 = np.zeros_like(x)
                for ch in range(3):
                    burst = rng.standard_normal(T).astype(np.float32)
                    # 简单平滑模拟地震包络
                    onset = int(rng.integers(0, max(1, T - T // 4)))
                    decay = np.exp(-np.arange(T - onset) / (T * 0.1)).astype(np.float32)
                    burst[onset:] *= decay
                    burst[:onset]  = 0.0
                    x2[:, ch] = burst

            # 目标相对幅度：对数均匀采样
            main_rms = np.sqrt(np.mean(x ** 2)) + 1e-8
            sec_rms  = np.sqrt(np.mean(x2 ** 2)) + 1e-8
            amp = np.exp(rng.uniform(
                np.log(self.second_event_amp_range[0]),
                np.log(self.second_event_amp_range[1])
            ))
            # 将第二事件缩放到目标相对幅度后叠加
            x2_scaled = x2 * (main_rms / sec_rms * amp)
            x += x2_scaled.astype(np.float32)

        return x.astype(np.float32)


# ==================== Dataset ====================
class PhaseNetDataset(Dataset):
    """waves: (N, T, 3) | labels: (N, T, 3)"""
    def __init__(self, waves, labels, is_train=True, augmentor=None,
                 noise_augmentor=None):
        self.waves    = waves
        self.labels   = labels
        self.is_train = is_train
        self.augmentor = augmentor
        # 纯噪声样本单独用一套增强器：第二事件叠加会把地震塞进噪声窗，
        # 而标签仍是 [0,0,1]，等于教模型「像地震的是噪声」。
        self.noise_augmentor = noise_augmentor

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave  = self.waves[idx].copy()    # (T, 3)
        label = self.labels[idx].copy()   # (T, 3)
        if self.is_train:
            # 噪声样本判据：noise 通道几乎全 1（注意不能用 EQT 的 label.max()<1e-6，
            # 因为 PhaseNet 的噪声标签是 [0,0,1]，max 恒为 1）
            is_noise = float(label[:, 2].min()) > 0.5
            aug = self.noise_augmentor if (is_noise and self.noise_augmentor is not None) \
                else self.augmentor
            if aug is not None:
                wave = aug(wave)
        wave_t  = torch.from_numpy(wave.T).float()   # (3, T) for Conv1d
        label_t = torch.from_numpy(label).float()    # (T, 3)
        return wave_t, label_t


# ==================== PhaseNet 模型 ====================
# 严格对应原版 TF 实现（AI4EPS/PhaseNet，model.py）：
#   depths=5, filters_root=8, kernel_size=7, pool_size=4
#   Input 层: in_ch → filters_root
#   DownConv_d: ch_in → filters_root * 2^d（单层卷积）
#   MaxPool(4) 下采样
#   UpConv_d: ConvTranspose(4) 上采样 + skip-cat + 单层卷积
#   输出: 1×1 conv → softmax
#   参数量 ~20 万
class PhaseNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=3,
                 filters_root=8, depths=5, kernel_size=7, pool_size=4):
        super().__init__()
        self.depths    = depths
        self.pool_size = pool_size
        fs = filters_root
        k  = kernel_size
        p  = k // 2

        # ---- Input layer (原版 "Input" scope) ----
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, fs, k, padding=p, bias=False),
            nn.BatchNorm1d(fs),
            nn.ReLU(inplace=True),
        )

        # ---- Encoder (原版 DownConv_0 .. DownConv_{depths-1}) ----
        # DownConv_d 的 filters = filters_root * 2^d
        self.down_convs = nn.ModuleList()
        self.pools      = nn.ModuleList()
        ch = fs
        for d in range(depths):
            oc = fs * (2 ** d)
            self.down_convs.append(nn.Sequential(
                nn.Conv1d(ch, oc, k, padding=p, bias=False),
                nn.BatchNorm1d(oc),
                nn.ReLU(inplace=True),
            ))
            if d < depths - 1:
                self.pools.append(nn.MaxPool1d(pool_size))
            ch = oc

        # ---- Decoder (原版 UpConv_{depths-2} .. UpConv_0) ----
        self.up_trans = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for d in range(depths - 2, -1, -1):
            ec = fs * (2 ** d)
            self.up_trans.append(
                nn.ConvTranspose1d(ch, ec, pool_size, stride=pool_size)
            )
            self.up_convs.append(nn.Sequential(
                nn.Conv1d(ec * 2, ec, k, padding=p, bias=False),
                nn.BatchNorm1d(ec),
                nn.ReLU(inplace=True),
            ))
            ch = ec

        # ---- 1×1 output conv ----
        self.out_conv = nn.Conv1d(ch, n_classes, 1)

    def forward(self, x):
        # x: (B, 3, T)
        x = self.input_conv(x)

        skips = []
        for d in range(self.depths):
            x = self.down_convs[d](x)
            if d < self.depths - 1:
                skips.append(x)
                x = self.pools[d](x)

        for i, (up, conv) in enumerate(zip(self.up_trans, self.up_convs)):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1],
                                  mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        x = self.out_conv(x)         # (B, n_classes, T)
        x = x.permute(0, 2, 1)       # (B, T, n_classes)
        x = F.softmax(x, dim=-1)     # 原版使用 softmax
        return x


# ==================== 损失函数 ====================
class PhaseNetLoss(nn.Module):
    """
    原版 PhaseNet 交叉熵损失（loss_type="cross_entropy"）：
        L = -mean_{b,t} [ sum_c  target_c(t) * log(pred_c(t)) ]
    pred:   (B, T, 3) — softmax 后的概率
    target: (B, T, 3) — 归一化软标签（和为 1）
    """
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7)
        ce = -(target * torch.log(pred)).sum(dim=-1)   # (B, T)
        return ce.mean()


# ==================== 批量结果可视化 ====================
def plot_batch_results(y_true_p, y_pred_p, y_true_s, y_pred_s,
                       save_path: str = 'test_results.pdf',
                       window_size: int = WINDOW_SIZE):
    """
    批量推断结果综合可视化（P波和S波的残差分析 + 误差统计）。
    """
    # P波分析
    y_true_p_s = y_true_p * window_size
    y_pred_p_s = y_pred_p * window_size
    residuals_p = y_pred_p_s - y_true_p_s
    errors_p = np.abs(residuals_p)
    mu_p, std_p = np.mean(residuals_p), np.std(residuals_p)

    mae_p = np.mean(errors_p)
    rmse_p = np.sqrt(np.mean(residuals_p ** 2))
    acc10_p = np.mean(errors_p <= 10) * 100
    acc20_p = np.mean(errors_p <= 20) * 100
    acc50_p = np.mean(errors_p <= 50) * 100

    # S波分析
    y_true_s_s = y_true_s * window_size
    y_pred_s_s = y_pred_s * window_size
    residuals_s = y_pred_s_s - y_true_s_s
    errors_s = np.abs(residuals_s)
    mu_s, std_s = np.mean(residuals_s), np.std(residuals_s)

    mae_s = np.mean(errors_s)
    rmse_s = np.sqrt(np.mean(residuals_s ** 2))
    acc10_s = np.mean(errors_s <= 10) * 100
    acc20_s = np.mean(errors_s <= 20) * 100
    acc50_s = np.mean(errors_s <= 50) * 100

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.32)

    # P波可视化
    # 1. P波预测值 vs 真实值散点
    ax1 = fig.add_subplot(gs[0, 0])
    try:
        xy = np.vstack([y_true_p_s, y_pred_p_s])
        z = gaussian_kde(xy)(xy)
        ax1.scatter(y_true_p_s, y_pred_p_s, c=z, s=4, cmap='viridis',
                    alpha=0.6, rasterized=True)
    except Exception:
        ax1.scatter(y_true_p_s, y_pred_p_s, s=4, alpha=0.3, color='steelblue')
    lim = [0, window_size]
    ax1.plot(lim, lim, 'r--', lw=2, label='Perfect')
    ax1.set_xlabel('True Position (samples)', fontsize=11)
    ax1.set_ylabel('Predicted Position (samples)', fontsize=11)
    ax1.set_title('P-wave: Predicted vs True', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. P波残差散点图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_true_p_s, residuals_p, s=3, alpha=0.3, color='purple')
    ax2.axhline(0, color='red', lw=2, linestyle='-', label='Zero')
    ax2.axhline(20, color='orange', lw=1.5, linestyle='--', alpha=0.7, label='±20 samp')
    ax2.axhline(-20, color='orange', lw=1.5, linestyle='--', alpha=0.7)
    ax2.set_xlabel('True Position (samples)', fontsize=11)
    ax2.set_ylabel('Residual (Pred − True)', fontsize=11)
    ax2.set_title('P-wave: Residual Plot', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-150, 150)

    # 3. P波残差直方图 + 正态拟合
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals_p, bins=80, density=True, alpha=0.7,
             color='skyblue', edgecolor='black', range=(-150, 150),
             label='Residuals')
    x_fit = np.linspace(-150, 150, 300)
    ax3.plot(x_fit, norm.pdf(x_fit, mu_p, std_p), 'r-', lw=2,
             label=f'Normal  μ={mu_p:.1f}, σ={std_p:.1f}')
    ax3.axvline(0, color='black', lw=2, alpha=0.8, label='Zero')
    ax3.axvline(20, color='orange', lw=1.5, linestyle='--', alpha=0.6, label='±20')
    ax3.axvline(-20, color='orange', lw=1.5, linestyle='--', alpha=0.6)
    ax3.set_xlabel('Residual (samples)', fontsize=11)
    ax3.set_ylabel('Probability Density', fontsize=11)
    ax3.set_title('P-wave: Residual Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.2)

    # 4. P波误差累积分布（CDF）
    ax4 = fig.add_subplot(gs[0, 3])
    sorted_err = np.sort(errors_p)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax4.plot(sorted_err, cdf, lw=2.5, color='navy', label='Empirical CDF')
    ax4.axvline(10, color='g', lw=2, linestyle='--', alpha=0.8,
                label=f'10 samp: {acc10_p:.1f}%')
    ax4.axvline(20, color='orange', lw=2, linestyle='--', alpha=0.8,
                label=f'20 samp: {acc20_p:.1f}%')
    ax4.axvline(50, color='r', lw=2, linestyle='--', alpha=0.8,
                label=f'50 samp: {acc50_p:.1f}%')
    ax4.set_xlabel('Absolute Error (samples)', fontsize=11)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax4.set_title('P-wave: CDF of Absolute Error', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 200)

    # S波可视化
    # 5. S波预测值 vs 真实值散点
    ax5 = fig.add_subplot(gs[1, 0])
    try:
        xy = np.vstack([y_true_s_s, y_pred_s_s])
        z = gaussian_kde(xy)(xy)
        ax5.scatter(y_true_s_s, y_pred_s_s, c=z, s=4, cmap='viridis',
                    alpha=0.6, rasterized=True)
    except Exception:
        ax5.scatter(y_true_s_s, y_pred_s_s, s=4, alpha=0.3, color='steelblue')
    lim = [0, window_size]
    ax5.plot(lim, lim, 'r--', lw=2, label='Perfect')
    ax5.set_xlabel('True Position (samples)', fontsize=11)
    ax5.set_ylabel('Predicted Position (samples)', fontsize=11)
    ax5.set_title('S-wave: Predicted vs True', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. S波残差散点图
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.scatter(y_true_s_s, residuals_s, s=3, alpha=0.3, color='purple')
    ax6.axhline(0, color='red', lw=2, linestyle='-', label='Zero')
    ax6.axhline(20, color='orange', lw=1.5, linestyle='--', alpha=0.7, label='±20 samp')
    ax6.axhline(-20, color='orange', lw=1.5, linestyle='--', alpha=0.7)
    ax6.set_xlabel('True Position (samples)', fontsize=11)
    ax6.set_ylabel('Residual (Pred − True)', fontsize=11)
    ax6.set_title('S-wave: Residual Plot', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-150, 150)

    # 7. S波残差直方图 + 正态拟合
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(residuals_s, bins=80, density=True, alpha=0.7,
             color='skyblue', edgecolor='black', range=(-150, 150),
             label='Residuals')
    x_fit = np.linspace(-150, 150, 300)
    ax7.plot(x_fit, norm.pdf(x_fit, mu_s, std_s), 'r-', lw=2,
             label=f'Normal  μ={mu_s:.1f}, σ={std_s:.1f}')
    ax7.axvline(0, color='black', lw=2, alpha=0.8, label='Zero')
    ax7.axvline(20, color='orange', lw=1.5, linestyle='--', alpha=0.6, label='±20')
    ax7.axvline(-20, color='orange', lw=1.5, linestyle='--', alpha=0.6)
    ax7.set_xlabel('Residual (samples)', fontsize=11)
    ax7.set_ylabel('Probability Density', fontsize=11)
    ax7.set_title('S-wave: Residual Distribution', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8, loc='upper right')
    ax7.grid(True, alpha=0.2)

    # 8. S波误差累积分布（CDF）
    ax8 = fig.add_subplot(gs[1, 3])
    sorted_err = np.sort(errors_s)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax8.plot(sorted_err, cdf, lw=2.5, color='navy', label='Empirical CDF')
    ax8.axvline(10, color='g', lw=2, linestyle='--', alpha=0.8,
                label=f'10 samp: {acc10_s:.1f}%')
    ax8.axvline(20, color='orange', lw=2, linestyle='--', alpha=0.8,
                label=f'20 samp: {acc20_s:.1f}%')
    ax8.axvline(50, color='r', lw=2, linestyle='--', alpha=0.8,
                label=f'50 samp: {acc50_s:.1f}%')
    ax8.set_xlabel('Absolute Error (samples)', fontsize=11)
    ax8.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax8.set_title('S-wave: CDF of Absolute Error', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9, loc='lower right')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, 200)

    # 9. P波密度热图
    ax9 = fig.add_subplot(gs[2, 0])
    h, xe, ye = np.histogram2d(y_true_p_s, y_pred_p_s,
                               bins=60, range=[[0, window_size], [0, window_size]])
    im = ax9.imshow(np.log10(h.T + 1), origin='lower', aspect='auto',
                    extent=[0, window_size, 0, window_size],
                    cmap='hot', interpolation='gaussian')
    ax9.plot([0, window_size], [0, window_size], 'cyan', lw=2, label='Perfect')
    ax9.set_xlabel('True Position (samples)', fontsize=11)
    ax9.set_ylabel('Predicted Position (samples)', fontsize=11)
    ax9.set_title('P-wave: Density Heatmap (log scale)', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9, loc='upper left')
    plt.colorbar(im, ax=ax9, label='log10(Count+1)')

    # 10. S波密度热图
    ax10 = fig.add_subplot(gs[2, 1])
    h, xe, ye = np.histogram2d(y_true_s_s, y_pred_s_s,
                               bins=60, range=[[0, window_size], [0, window_size]])
    im = ax10.imshow(np.log10(h.T + 1), origin='lower', aspect='auto',
                     extent=[0, window_size, 0, window_size],
                     cmap='hot', interpolation='gaussian')
    ax10.plot([0, window_size], [0, window_size], 'cyan', lw=2, label='Perfect')
    ax10.set_xlabel('True Position (samples)', fontsize=11)
    ax10.set_ylabel('Predicted Position (samples)', fontsize=11)
    ax10.set_title('S-wave: Density Heatmap (log scale)', fontsize=12, fontweight='bold')
    ax10.legend(fontsize=9, loc='upper left')
    plt.colorbar(im, ax=ax10, label='log10(Count+1)')

    # 11. 统计摘要文本框（P波）
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.axis('off')

    skewness_p = stats.skew(residuals_p)
    kurtosis_p = stats.kurtosis(residuals_p)

    n_sample = min(5000, len(residuals_p))
    rng = np.random.default_rng(0)
    sample_res = rng.choice(residuals_p, n_sample, replace=False)
    _, shapiro_p = stats.shapiro(sample_res)

    summary_p = f"""
╔══════════════════════════════════════╗
║      P-wave Statistics                ║
╚══════════════════════════════════════╝

Samples  : {len(y_true_p):,}

Residual
  Mean   : {mu_p:.2f} samp
  Std    : {std_p:.2f} samp
  Skew   : {skewness_p:.3f}
  Kurt   : {kurtosis_p:.3f}

Normality (Shapiro-Wilk)
  p-value: {shapiro_p:.4f}  {'✓ Normal' if shapiro_p > 0.05 else '✗ Non-normal'}

Error
  MAE    : {mae_p:.2f} samp
  RMSE   : {rmse_p:.2f} samp
  Median : {np.median(errors_p):.2f} samp

Accuracy
  ≤10 samp : {acc10_p:.2f}%
  ≤20 samp : {acc20_p:.2f}%
  ≤50 samp : {acc50_p:.2f}%

Percentiles (|error|)
  50th : {np.percentile(errors_p, 50):.1f}
  75th : {np.percentile(errors_p, 75):.1f}
  90th : {np.percentile(errors_p, 90):.1f}
  95th : {np.percentile(errors_p, 95):.1f}
"""
    ax11.text(0.05, 0.5, summary_p, fontsize=8.5, family='monospace',
              verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 12. 统计摘要文本框（S波）
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    skewness_s = stats.skew(residuals_s)
    kurtosis_s = stats.kurtosis(residuals_s)

    n_sample = min(5000, len(residuals_s))
    rng = np.random.default_rng(0)
    sample_res = rng.choice(residuals_s, n_sample, replace=False)
    _, shapiro_s = stats.shapiro(sample_res)

    summary_s = f"""
╔══════════════════════════════════════╗
║      S-wave Statistics                ║
╚══════════════════════════════════════╝

Samples  : {len(y_true_s):,}

Residual
  Mean   : {mu_s:.2f} samp
  Std    : {std_s:.2f} samp
  Skew   : {skewness_s:.3f}
  Kurt   : {kurtosis_s:.3f}

Normality (Shapiro-Wilk)
  p-value: {shapiro_s:.4f}  {'✓ Normal' if shapiro_s > 0.05 else '✗ Non-normal'}

Error
  MAE    : {mae_s:.2f} samp
  RMSE   : {rmse_s:.2f} samp
  Median : {np.median(errors_s):.2f} samp

Accuracy
  ≤10 samp : {acc10_s:.2f}%
  ≤20 samp : {acc20_s:.2f}%
  ≤50 samp : {acc50_s:.2f}%

Percentiles (|error|)
  50th : {np.percentile(errors_s, 50):.1f}
  75th : {np.percentile(errors_s, 75):.1f}
  90th : {np.percentile(errors_s, 90):.1f}
  95th : {np.percentile(errors_s, 95):.1f}
"""
    ax12.text(0.05, 0.5, summary_s, fontsize=8.5, family='monospace',
              verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    fig.suptitle('PhaseNet Baseline — Test Set Results (P & S Waves)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  测试结果图已保存: {save_path}")

    return (mae_p, rmse_p, acc10_p, acc20_p, acc50_p,
            mae_s, rmse_s, acc10_s, acc20_s, acc50_s)


# ==================== 训练 / 验证 ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for wave, label in loader:
        wave, label = wave.to(device), label.to(device)
        pred = model(wave)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    p_errors, s_errors = [], []
    p_true_list, p_pred_list = [], []
    s_true_list, s_pred_list = [], []

    for wave, label in loader:
        wave, label = wave.to(device), label.to(device)
        pred = model(wave)
        total_loss += criterion(pred, label).item()

        pred_np  = pred.cpu().numpy()
        label_np = label.cpu().numpy()
        B = pred_np.shape[0]
        for b in range(B):
            true_p = np.argmax(label_np[b, :, 0])
            pred_p = np.argmax(pred_np[b, :, 0])
            if label_np[b, true_p, 0] > 0.01:
                p_errors.append(abs(pred_p - true_p))
                p_true_list.append(true_p / WINDOW_SIZE)
                p_pred_list.append(pred_p / WINDOW_SIZE)
            true_s = np.argmax(label_np[b, :, 1])
            pred_s = np.argmax(pred_np[b, :, 1])
            if label_np[b, true_s, 1] > 0.01:
                s_errors.append(abs(pred_s - true_s))
                s_true_list.append(true_s / WINDOW_SIZE)
                s_pred_list.append(pred_s / WINDOW_SIZE)

    avg_loss = total_loss / max(1, len(loader))
    p_mae = np.mean(p_errors) if p_errors else float('nan')
    s_mae = np.mean(s_errors) if s_errors else float('nan')
    return avg_loss, p_mae, s_mae, p_true_list, p_pred_list, s_true_list, s_pred_list


# ==================== 主函数 ====================
def main(seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("PhaseNet Baseline  (depths=5, filters_root=8, pool_size=4)")
    print(f"Random seed: {seed}")
    print("=" * 60)

    # ===================== 事件级划分（回应审稿人） =====================
    print(f"训练 chunk: {CFG['train_chunks']}  波形库: {CFG['pool_chunk']}")
    meta, exclude = _event_meta(seed)

    # ---- 总量 => 每个 chunk 的上限（默认 100000 条按 0.8/0.15/0.05 分）----
    if CFG['total_samples'] > 0:
        _c = split_caps(CFG['total_samples'], CFG['ratios'],
                        len(CFG['train_chunks']))
        CFG['max_samples'], CFG['val_samples'], CFG['test_samples'] = (
            _c['train'], _c['val'], _c['test'])
        print("[事件级划分] 信号总量=%d 比例=%s -> 每个 chunk 上限 "
              "train=%d, val=%d, test=%d；合计=%d 条"
              % (CFG['total_samples'], CFG['ratios'], _c['train'], _c['val'],
                 _c['test'], _c['total']))
        if _c['total'] != CFG['total_samples']:
            print("[事件级划分] 注意：因向上取整，实际总量为 %d（目标 %d）"
                  % (_c['total'], CFG['total_samples']))

    def _load_split(split, max_samples):
        w_l, y_l = [], []
        for c in CFG['train_chunks']:
            names = names_for_split(meta, split, c, max_samples, seed)
            if not names:
                continue
            hf, cf = _paths(c)
            print(f"\n[{split}] {c}: 指定 {len(names)} 条")
            w, y = load_phasenet_data(hf, cf, window_size=WINDOW_SIZE,
                                      max_samples=max_samples, seed=seed,
                                      trace_names=names)
            if len(w) > 0:
                w_l.append(w); y_l.append(y)
        if not w_l:
            return (np.zeros((0, WINDOW_SIZE, 3), np.float32),
                    np.zeros((0, WINDOW_SIZE, 3), np.float32))
        return np.concatenate(w_l, axis=0), np.concatenate(y_l, axis=0)

    X_tr, y_tr = _load_split('train', CFG['max_samples'])
    X_va, y_va = _load_split('val',   CFG['val_samples'])
    X_te, y_te = _load_split('test',  CFG['test_samples'])

    # ---- 噪声（chunk1）----
    # PhaseNet 是「检测+拾取」一体模型（输出 [P,S,noise] 三通道），必须见过纯噪声，
    # 否则 noise 通道学不会、虚检率偏高（原脚本未加载噪声，这里补上）。
    if CFG['noise_chunks']:
        print("\nLoading noise data ...")
        n_noise = max(len(CFG['noise_chunks']), 1)
        NC = split_caps(CFG['noise_samples'], CFG['ratios'], n_noise)
        noise_cap = NC['train'] + NC['val'] + NC['test']
        print(f"[噪声] 总量={CFG['noise_samples']} 比例={CFG['ratios']} -> "
              f"每个 chunk 上限 {noise_cap}（train/val/test = {NC['train']}/"
              f"{NC['val']}/{NC['test']}）；合计 {NC['total']} 条")
        for c in CFG['noise_chunks']:
            hf, cf = _paths(c)
            if not (os.path.exists(hf) and os.path.exists(cf)):
                print(f"[skip] 不存在: {hf}")
                continue
            nw, ny = load_noise_data(hf, cf, window_size=WINDOW_SIZE,
                                     max_samples=noise_cap,
                                     seed=seed)
            if len(nw) == 0:
                continue
            idx = np.random.default_rng(seed).permutation(len(nw))
            n_tr, n_va = NC['train'], NC['val']
            parts = {'train': idx[:n_tr], 'val': idx[n_tr:n_tr + n_va],
                     'test': idx[n_tr + n_va:]}
            print(f"  {c}: {len(nw)} 条噪声 -> train/val/test = "
                  f"{len(parts['train'])}/{len(parts['val'])}/"
                  f"{len(parts['test'])}")
            X_tr = np.concatenate([X_tr, nw[parts['train']]], 0)
            y_tr = np.concatenate([y_tr, ny[parts['train']]], 0)
            X_va = np.concatenate([X_va, nw[parts['val']]], 0)
            y_va = np.concatenate([y_va, ny[parts['val']]], 0)
            X_te = np.concatenate([X_te, nw[parts['test']]], 0)
            y_te = np.concatenate([y_te, ny[parts['test']]], 0)

    waves, labels = X_tr, y_tr          # 兼容原代码后续引用
    print(f"\nTrain: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

    # ---- 波形库（排除 val/test 事件）----
    wave_pool = None
    if CFG['pool_chunk']:
        print("\nLoading waveform pool for augmentation...")
        pm = load_chunks(_data_dir(), [CFG['pool_chunk']])
        pn = pool_names(pm, exclude, CFG['pool_samples'], seed)
        phf, pcf = _paths(CFG['pool_chunk'])
        pw, _ = load_phasenet_data(phf, pcf, window_size=WINDOW_SIZE,
                                   max_samples=CFG['pool_samples'], seed=seed,
                                   trace_names=pn)
        if len(pw) > 0:
            wave_pool = np.concatenate([pw], axis=0)
            print(f"Waveform pool: {len(wave_pool)} samples "
                  f"(已排除 val/test 事件)")

    # ---- 创建数据增强器 ----
    if CFG.get('dry_run'):
        _dry_report(X_tr, None, None, X_va, None, None, X_te, None,
                    None, wave_pool, meta)
        return

    augmentor = SimpleAugmentation(
        noise_std_range=(0.01, 0.30),
        scale_range=(0.5, 2.0),
        noise_prob=0.70,
        scale_prob=0.50,
        dropout_prob=0.30,
        dropout_n_range=(1, 2),
        second_event_prob=0.30,
        second_event_pool=wave_pool,
        second_event_amp_range=(0.1, 1.0),
    )

    # 噪声专用增强：保留加噪/缩放/丢通道，但**不做第二事件叠加**
    noise_augmentor = SimpleAugmentation(
        noise_std_range=(0.01, 0.30),
        scale_range=(0.5, 2.0),
        noise_prob=0.70,
        scale_prob=0.50,
        dropout_prob=0.30,
        dropout_n_range=(1, 2),
        second_event_prob=0.0,
        second_event_pool=None,
        second_event_amp_range=(0.1, 1.0),
    )
    train_ds = PhaseNetDataset(X_tr, y_tr, is_train=True, augmentor=augmentor,
                               noise_augmentor=noise_augmentor)
    val_ds   = PhaseNetDataset(X_va, y_va, is_train=False)
    test_ds  = PhaseNetDataset(X_te, y_te, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,  shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128,         shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128,         shuffle=False,
                              num_workers=4, pin_memory=True)

    # ---- 模型 ----
    model = PhaseNet(in_channels=3, n_classes=3,
                     filters_root=8, depths=5,
                     kernel_size=7, pool_size=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")   # 应约为 ~20 万

    criterion = PhaseNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                 weight_decay=1e-5)   # 与 BiMamba/EQT 统一
    # 原版 TF 使用指数衰减；这里用 MultiStepLR 近似
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 70, 90], gamma=0.3)

    # ---- 训练循环 ----
    best_val = float('inf')
    tr_losses, va_losses = [], []
    va_p_maes, va_s_maes = [], []
    
    import time
    start_time = time.time()

    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, vp, vs, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        tr_losses.append(tr); va_losses.append(vl)
        va_p_maes.append(vp); va_s_maes.append(vs)

        # 与 BiMamba 统一：按验证集 P 拾取误差保存（vp 为 NaN 时跳过）
        if vp == vp and vp < best_val:
            best_val = vp
            torch.save(model.state_dict(), MODEL_SAVE)

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            lr_now = optimizer.param_groups[0]['lr']
            
            # 计算剩余时间
            elapsed_time = time.time() - start_time
            epochs_completed = epoch + 1
            avg_time_per_epoch = elapsed_time / epochs_completed
            remaining_epochs = NUM_EPOCHS - epochs_completed
            remaining_time = avg_time_per_epoch * remaining_epochs
            
            # 格式化时间显示
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}h {minutes}m {secs}s"
                elif minutes > 0:
                    return f"{minutes}m {secs}s"
                else:
                    return f"{secs}s"
            
            elapsed_str = format_time(elapsed_time)
            remaining_str = format_time(remaining_time)
            
            print(f"Epoch {epoch:03d}/{NUM_EPOCHS}  tr={tr:.6f}  val={vl:.6f}  "
                  f"P_MAE={vp:.1f}samp  S_MAE={vs:.1f}samp  "
                  f"lr={lr_now:.2e}  best={best_val:.6f}  "
                  f"elapsed: {elapsed_str}  remaining: {remaining_str}")

    # ---- 最终测试 ----
    print("\nFinal test evaluation:")
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    te_loss, te_p, te_s, p_true_list, p_pred_list, s_true_list, s_pred_list = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss : {te_loss:.6f}")
    print(f"  P-wave MAE: {te_p:.2f} samples  ({te_p / 100 * 1000:.1f} ms @ 100 Hz)")
    print(f"  S-wave MAE: {te_s:.2f} samples  ({te_s / 100 * 1000:.1f} ms @ 100 Hz)")

    # 使用BiMamba_Linear_3C_LFEM_gauss.py中的图像展示功能
    if p_true_list and p_pred_list and s_true_list and s_pred_list:
        p_true_arr = np.array(p_true_list)
        p_pred_arr = np.array(p_pred_list)
        s_true_arr = np.array(s_true_list)
        s_pred_arr = np.array(s_pred_list)
        plot_batch_results(p_true_arr, p_pred_arr, s_true_arr, s_pred_arr,
                          save_path='test_set_results.pdf', window_size=WINDOW_SIZE)

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = np.arange(NUM_EPOCHS)

    axes[0].plot(ep, tr_losses, label='Train')
    axes[0].plot(ep, va_losses, label='Val')
    axes[0].set_title('Cross-Entropy Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(ep, va_p_maes, label='P-wave')
    axes[1].plot(ep, va_s_maes, label='S-wave')
    axes[1].set_title('Val Pick MAE (samples)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    model.eval()
    with torch.no_grad():
        sw, sl = test_ds[0]
        po = model(sw.unsqueeze(0).to(device))[0].cpu().numpy()
        lo = sl.numpy()
        t  = np.arange(WINDOW_SIZE) / 100.0
    axes[2].plot(t, lo[:, 0], 'b--', lw=1,   label='Label P')
    axes[2].plot(t, lo[:, 1], 'r--', lw=1,   label='Label S')
    axes[2].plot(t, po[:, 0], 'b-',  lw=1.5, label='Pred P')
    axes[2].plot(t, po[:, 1], 'r-',  lw=1.5, label='Pred S')
    axes[2].set_title('Sample Prediction')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('phasenet_results.pdf', dpi=150)
    plt.close()
    print("Plot saved to 'phasenet_results.pdf'")
    print(f"Model saved to '{MODEL_SAVE}'")


if __name__ == '__main__':
    main(seed=apply_cli())

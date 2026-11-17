"""
EQTransformer Baseline Training Script
=======================================
数据加载逻辑与 phasenet_baseline_cld.py 完全一致：
  - STEAD 格式 hdf5/csv，多 chunk，随机采样
  - 排除纯噪声（p_arrival_sample <= 0）
  - train/val/test = 64/16/20（先 80/20，再对 train 做 80/20）
  - 相同数据增强（SimpleAugmentation）
  - 输入：三分量波形 (Z, N, E)，长度 3000 采样点（已统一为 30 s；原版 EQT 为 60 s）
  - 输出：三通道逐点概率图 (检测/detector, P相, S相)，形状 (batch, 3000, 3)

模型架构：严格对应原版 Mousavi et al. 2020 (EQTransformer)
  编码器：7 层 Conv 残差块 → 每层后 MaxPool(2)，通道 [8,16,16,32,32,64,64]
  解码器：三个独立分支（Detector / P-picker / S-picker）
           每分支：反卷积上采样 → LSTM × 1 → Multi-Head Self-Attention × 1 → 1×1 Conv → Sigmoid
  参数量约 ~370 K

损失函数：三分支 Binary Cross-Entropy 之和
  L = BCE(det_pred, det_label) + BCE(p_pred, p_label) + BCE(s_pred, s_label)

参考：
  Mousavi, S. M., et al. (2020). Earthquake transformer—an attentive
  deep-learning model for simultaneous earthquake detection and phase picking.
  Nature Communications, 11(1), 3952.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from scipy.stats import norm
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
    'STEAD_DIR', './')
CFG = {
    'data_dir': _1DSIGNAL,
    'train_chunks': ['chunk2', 'chunk3'],
    'pool_chunk': 'chunk4',
    # EQTransformer 同样是一体模型；原 autodl 脚本默认就用 chunk1
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
    ap.add_argument('--window-size', type=int, default=WINDOW_SIZE,
                    help='窗长（采样点）；默认 3000')
    ap.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    ap.add_argument('--tag', default='', help='权重名后缀')
    a = ap.parse_args(argv)
    globals()['WINDOW_SIZE'] = int(a.window_size)
    globals()['NUM_EPOCHS'] = int(a.epochs)
    if a.tag:
        globals()['MODEL_SAVE'] = f'best_eqtransformer_baseline{a.tag}.pth'
    print(f'[窗长] EQTransformer window_size = {WINDOW_SIZE} 采样点 ({WINDOW_SIZE/100:.1f}s)')
    print(f'[配置] epochs={NUM_EPOCHS}  权重输出={MODEL_SAVE}')
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
# 统一为 3000（30 s）：与 BiMamba 拾取窗口、PhaseNet 一致。
# 原版 EQTransformer 用 60 s；改成 30 s 会减少 S 到时的有效覆盖
# （标签要求 s_idx < window_size），但对三个模型是同等影响。
WINDOW_SIZE  = 3000
# 通道序统一为 Z,N,E（STEAD 原始 h5 列序是 E,N,Z）
CH_Z, CH_N, CH_E = 2, 1, 0
CHANNEL_INDICES  = [CH_Z, CH_N, CH_E]
SIGMA        = 10      # 高斯标签标准差（采样点）
BATCH_SIZE   = 64       # 统一为 64（T=3000 显存约为 T=6000 的一半）
NUM_EPOCHS   = 100
LR           = 1e-3
MAX_SAMPLES  = 300     # 每个 chunk 最大采样数
MODEL_SAVE   = 'best_eqtransformer_baseline.pth'


# ==================== 预处理（与 PhaseNet 版本完全一致）====================
def preprocess_3c(data3c, fs=100):
    """
    三分量预处理：去趋势 + 1-40 Hz 带通 + 逐通道标准化。
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
def make_label_eqt(p_idx, s_idx, length, sigma=SIGMA):
    """
    EQTransformer 三通道软标签 (length, 3)：
      ch0: Detector  — 覆盖 P/S 之间的方波（P 前 buffer ~ S 后 buffer）
      ch1: P 相高斯
      ch2: S 相高斯

    detector 标签：在 [p_idx - 2σ, s_idx + 2σ] 区间置 1，其余 0。
    若 p_idx 或 s_idx 不存在则 detector 全 0。
    """
    t = np.arange(length, dtype=np.float32)

    # P / S 高斯
    p_g = (np.exp(-0.5 * ((t - p_idx) / sigma) ** 2)
           if 0 <= p_idx < length else np.zeros(length, np.float32))
    s_g = (np.exp(-0.5 * ((t - s_idx) / sigma) ** 2)
           if 0 <= s_idx < length else np.zeros(length, np.float32))

    # Detector：矩形窗覆盖地震信号区间
    det = np.zeros(length, np.float32)
    if 0 <= p_idx < length:
        start = max(0, p_idx - int(2 * sigma))
        end   = min(length, (s_idx + int(2 * sigma) + 1)
                    if 0 <= s_idx < length else p_idx + int(4 * sigma) + 1)
        det[start:end] = 1.0

    label = np.stack([det, p_g, s_g], axis=-1)   # (T, 3)
    return label.astype(np.float32)


# ==================== 噪声数据加载 ====================
def load_noise_data(hdf5_file, csv_file,
                    window_size=WINDOW_SIZE, max_samples=MAX_SAMPLES, seed=42,
                    trace_names=None):
    """
    从 STEAD chunk1（纯噪声）加载噪声波形。
    CSV 中 trace_category == 'noise'，无 P/S 到时标注。
    返回:
        waves_noise:  ndarray (M, window_size, 3)
        labels_noise: ndarray (M, window_size, 3)  — 全零（无检测、无P、无S）
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}][Noise] Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Original rows: {len(df)}")

    if 'trace_category' in df.columns:
        df = df[df['trace_category'] == 'noise'].copy()
        print(f"  After noise filter: {len(df)}")

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
                print(f"[Noise] ERROR: No 'data' group in {hdf5_file}")
                return np.array([]), np.array([])

            err = 0
            for name in tqdm(ev_list, desc=f"  [Noise] {os.path.basename(hdf5_file)}"):
                try:
                    dset = f.get('data/' + str(name))
                    if dset is None:
                        err += 1; continue

                    data = np.array(dset)
                    if data.ndim == 1:
                        err += 1; continue
                    if data.shape[0] == 3 and data.shape[1] != 3:
                        data = data.T
                    if data.shape[1] < 3:
                        err += 1; continue

                    T = data.shape[0]
                    if T < window_size:
                        err += 1; continue

                    win = data[:window_size][:, CHANNEL_INDICES]  # 统一为 Z,N,E
                    win = preprocess_3c(win)
                    if np.isnan(win).any() or np.isinf(win).any():
                        err += 1; continue

                    # 噪声标签：全零
                    label = np.zeros((window_size, 3), dtype=np.float32)
                    waves_list.append(win)
                    labels_list.append(label)

                except Exception as e:
                    err += 1
                    if err < 10:
                        print(f"  [Noise] Warning [{name}]: {e}")
                    continue

            print(f"  [Noise] Loaded {len(waves_list)} samples | errors: {err}")

    except Exception as e:
        print(f"[Noise] ERROR opening {hdf5_file}: {e}")
        return np.array([]), np.array([])

    if len(waves_list) == 0:
        return np.array([]), np.array([])

    return (np.array(waves_list, dtype=np.float32),
            np.array(labels_list, dtype=np.float32))


# ==================== 数据加载（与 PhaseNet 版本完全一致）====================
def load_eqt_data(hdf5_file, csv_file,
                  window_size=WINDOW_SIZE, max_samples=MAX_SAMPLES, seed=42,
                  trace_names=None):
    """
    从 STEAD 格式 hdf5/csv 加载三分量波形及 P/S 到时。
    不加载纯噪声（p_arrival_sample <= 0 的样本跳过）。

    EQT 原版使用 6000 点窗口（60 s）；若数据不足则跳过。
    返回:
        waves:  ndarray (N, window_size, 3)
        labels: ndarray (N, window_size, 3)
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Original rows: {len(df)}")

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

                    p_arrival = int(dset.attrs.get('p_arrival_sample', -1))
                    s_arrival = int(dset.attrs.get('s_arrival_sample', -1))

                    if p_arrival <= 0 or p_arrival >= T:
                        continue

                    win = data[:window_size, CHANNEL_INDICES]   # 统一为 Z,N,E

                    if p_arrival >= window_size:
                        continue
                    s_idx = s_arrival if (0 <= s_arrival < window_size) else -1

                    win = preprocess_3c(win)
                    if np.isnan(win).any() or np.isinf(win).any():
                        err += 1; continue

                    label = make_label_eqt(p_arrival, s_idx, window_size)
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


# ==================== 数据增强（与 PhaseNet 版本完全一致）====================
class SimpleAugmentation:
    """
    三分量精简数据增强，作用于波形 x (T, 3)。
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
        self.noise_std_range        = noise_std_range
        self.scale_range            = scale_range
        self.noise_prob             = noise_prob
        self.scale_prob             = scale_prob
        self.dropout_prob           = dropout_prob
        self.dropout_n_range        = dropout_n_range
        self.second_event_prob      = second_event_prob
        self.second_event_pool      = second_event_pool
        self.second_event_amp_range = second_event_amp_range

    def __call__(self, wave):
        x   = wave.copy()
        rng = np.random.default_rng()
        T   = x.shape[0]

        # 1. 随机高斯噪声
        if rng.random() < self.noise_prob:
            log_lo = np.log(self.noise_std_range[0])
            log_hi = np.log(self.noise_std_range[1])
            std = np.exp(rng.uniform(log_lo, log_hi))
            x += rng.standard_normal(x.shape).astype(np.float32) * std

        # 2. 随机幅度缩放
        if rng.random() < self.scale_prob:
            scale = np.exp(rng.uniform(
                np.log(self.scale_range[0]),
                np.log(self.scale_range[1])
            ))
            x *= scale

        # 3. 随机分量 Dropout
        if rng.random() < self.dropout_prob:
            n_drop = int(rng.integers(self.dropout_n_range[0],
                                      self.dropout_n_range[1] + 1))
            drop_chs = rng.choice(3, size=n_drop, replace=False)
            for ch in drop_chs:
                x[:, ch] = 0.0

        # 4. 第二地震事件叠加
        if rng.random() < self.second_event_prob:
            if self.second_event_pool is not None:
                idx2 = int(rng.integers(0, len(self.second_event_pool)))
                x2 = self.second_event_pool[idx2].copy()
                if x2.shape[0] != T:
                    if x2.shape[0] > T:
                        st = int(rng.integers(0, x2.shape[0] - T))
                        x2 = x2[st: st + T]
                    else:
                        pad = T - x2.shape[0]
                        x2 = np.pad(x2, ((0, pad), (0, 0)), mode='constant')
            else:
                x2 = np.zeros_like(x)
                for ch in range(3):
                    burst = rng.standard_normal(T).astype(np.float32)
                    onset = int(rng.integers(0, max(1, T - T // 4)))
                    decay = np.exp(-np.arange(T - onset) / (T * 0.1)).astype(np.float32)
                    burst[onset:] *= decay
                    burst[:onset]  = 0.0
                    x2[:, ch] = burst

            main_rms = np.sqrt(np.mean(x ** 2)) + 1e-8
            sec_rms  = np.sqrt(np.mean(x2 ** 2)) + 1e-8
            amp = np.exp(rng.uniform(
                np.log(self.second_event_amp_range[0]),
                np.log(self.second_event_amp_range[1])
            ))
            x2_scaled = x2 * (main_rms / sec_rms * amp)
            x += x2_scaled.astype(np.float32)

        return x.astype(np.float32)


# 噪声增强：不加第二事件，避免在纯噪声中注入地震信号
class NoiseAugmentation(SimpleAugmentation):
    def __call__(self, wave):
        x = wave.copy()
        T = x.shape[0]
        rng = np.random.default_rng()
        if rng.random() < self.noise_prob:
            log_lo = np.log(self.noise_std_range[0])
            log_hi = np.log(self.noise_std_range[1])
            std = np.exp(rng.uniform(log_lo, log_hi))
            x += rng.standard_normal(x.shape).astype(np.float32) * std
        if rng.random() < self.scale_prob:
            scale = np.exp(rng.uniform(np.log(self.scale_range[0]),
                                        np.log(self.scale_range[1])))
            x *= scale
        if rng.random() < self.dropout_prob:
            n_drop = int(rng.integers(self.dropout_n_range[0],
                                      self.dropout_n_range[1] + 1))
            drop_chs = rng.choice(3, size=n_drop, replace=False)
            for ch in drop_chs:
                x[:, ch] = 0.0
        return x.astype(np.float32)


# ==================== Dataset ====================
class EQTDataset(Dataset):
    """waves: (N, T, 3) | labels: (N, T, 3)"""

    def __init__(self, waves, labels, is_train=True, augmentor=None):
        self.waves    = waves
        self.labels   = labels
        self.is_train = is_train
        self.augmentor = augmentor

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave  = self.waves[idx].copy()   # (T, 3)
        label = self.labels[idx].copy()  # (T, 3)
        if self.is_train and self.augmentor is not None:
            wave = self.augmentor(wave)
        wave_t  = torch.from_numpy(wave.T).float()   # (3, T) for Conv1d
        label_t = torch.from_numpy(label).float()    # (T, 3)
        return wave_t, label_t


# ==================== EQTransformer 模型 ====================
#
# 架构摘要（严格对应 Mousavi et al. 2020 Figure 1）：
#
#  Input (B, 3, T)
#    └─ Encoder: 7 × (ResConv1D → MaxPool2)
#                通道序列: [8,16,16,32,32,64,64]
#                T=3000 → 1500→750→375→187→93→46→23  (≈T/128)
#
#    └─ 【Transformer块在 bottleneck 处】序列长度 ~23，O(23²)可忽略
#         Multi-Head Self-Attention(8 heads) + Add&Norm
#
#  共享 bottleneck 特征 → 三个独立解码分支
#    每个分支：
#      7 × (ConvTranspose2 + ResConvBlock + skip-cat)  上采样回 T
#      BiLSTM(hidden=64)  — T=3000 时序列长度合理，T=6000 时需降 batch
#      1×1 Conv → Sigmoid  输出 (B, T, 1)
#
#  关键：Attention 在 bottleneck（原论文位置），不在解码器末端
#

# ── 残差卷积块 ──────────────────────────────────────────────

class ResConvBlock(nn.Module):
    """单层残差卷积块：Conv-BN-ReLU-Dropout-Conv-BN + skip"""

    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


# ── Multi-Head Self-Attention 块（用于 bottleneck）───────────

class SelfAttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention + Add & Norm。
    在 encoder bottleneck 处调用，序列长度 T/128 ≈ 23（T=3000），
    显存开销 O(B × (T/128)²) ≈ 可完全忽略。
    输入/输出: (B, L, d_model)，L = T/128
    """

    def __init__(self, d_model=64, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


# ── 编码器（含 bottleneck Attention）────────────────────────

class EQTEncoder(nn.Module):
    """
    7 层 ResConv → MaxPool(2)，最后在 bottleneck 做 Self-Attention。
    通道: [3→8, 8→16, 16→16, 16→32, 32→32, 32→64, 64→64]
    T=3000 → bottleneck 长度 ≈ 23
    """

    CHANNELS = [8, 16, 16, 32, 32, 64, 64]

    def __init__(self, in_ch=3, kernel_size=3, n_heads=8, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()
        ch = in_ch
        for oc in self.CHANNELS:
            self.blocks.append(ResConvBlock(ch, oc, kernel_size, dropout))
            self.pools.append(nn.MaxPool1d(2))
            ch = oc
        self.out_ch = ch  # 64

        # Attention 在 bottleneck（序列长度 ~23），显存可忽略
        self.bottleneck_attn = SelfAttentionBlock(ch, n_heads, dropout)

    def forward(self, x):
        skips = [x]           # 原始输入作为第 0 个 skip（3 通道）
        for blk, pool in zip(self.blocks, self.pools):
            x = blk(x)
            skips.append(x)   # 保存 skip，池化前
            x = pool(x)
        # x: (B, 64, T/128)  ← bottleneck
        # Self-Attention 在此处，序列极短
        x = x.permute(0, 2, 1)          # (B, L, 64)
        x = self.bottleneck_attn(x)     # (B, L, 64)  O(L²), L≈23
        x = x.permute(0, 2, 1)          # (B, 64, L)
        return x, skips


# ── 单个解码分支（上采样 + 池化-LSTM-插值）────────────────────

class EQTDecoderBranch(nn.Module):
    """
    EQTransformer 单输出分支：
      7 × (ConvTranspose1d(stride=2) + ResConvBlock + skip-cat)
      Pool(lstm_pool_factor) → BiLSTM → 线性插值还原
      1×1 Conv → Sigmoid

    lstm_pool_factor 是解决 T=6000 OOM 的核心参数：
      LSTM 实际处理序列长度 = T / lstm_pool_factor
      factor=1  → 原始长度，T=6000 时约需 3-4 GB/分支（显存充足时可用）
      factor=8  → 750 点，显存降 64×，精度损失极小（推荐）
      factor=16 → 375 点，显存最省，适合 <8 GB 显卡

    显存估算（T=6000, B=16, hidden=64, factor=8）：
      LSTM 序列 = 6000/8 = 750
      forward+backward ≈ 2 × 16 × 750 × 128 × 4 bytes ≈ 12 MB/分支  ✓
    """

    CHANNELS = [64, 32, 32, 16, 16, 8, 8]

    def __init__(self, enc_channels, kernel_size=3,
                 lstm_hidden=64, lstm_pool_factor=8, dropout=0.1):
        super().__init__()
        self.lstm_pool_factor = lstm_pool_factor

        enc_ch_rev = list(reversed(enc_channels))  # [64,64,32,32,16,16,8]
        enc_ch_rev.append(3)                        # + 原始输入 (3ch)

        self.up_trans = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        in_ch = enc_ch_rev[0]  # 64
        for i, out_ch in enumerate(self.CHANNELS):
            skip_ch = enc_ch_rev[i + 1] if i + 1 < len(enc_ch_rev) else 0
            self.up_trans.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.up_convs.append(
                ResConvBlock(out_ch + skip_ch, out_ch, kernel_size, dropout)
            )
            in_ch = out_ch
        # in_ch == 8（最浅层通道数）

        # Pool → LSTM → interpolate 三件套
        # LSTM 输入维度 = in_ch（上采样后的通道数）
        if lstm_pool_factor > 1:
            self.pre_lstm_pool = nn.AvgPool1d(lstm_pool_factor,
                                              stride=lstm_pool_factor)
        else:
            self.pre_lstm_pool = nn.Identity()

        self.lstm = nn.LSTM(in_ch, lstm_hidden, batch_first=True,
                            bidirectional=True)
        lstm_out = lstm_hidden * 2

        self.out_conv = nn.Conv1d(lstm_out, 1, 1)

    def forward(self, x, skips):
        """
        x    : (B, 64, L)  — encoder bottleneck（已过 Attention）
        skips: 编码器各层 skip，skips[0] 最浅，skips[-1] 最深
        """
        skips_rev = list(reversed(skips))

        for i, (up, conv) in enumerate(zip(self.up_trans, self.up_convs)):
            x = up(x)
            skip = skips_rev[i + 1] if i + 1 < len(skips_rev) else None
            if skip is not None:
                if x.shape[-1] != skip.shape[-1]:
                    x = F.interpolate(x, size=skip.shape[-1],
                                      mode='linear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = conv(x)

        # x: (B, 8, T)
        T_full = x.shape[-1]

        if self.lstm_pool_factor > 1:
            # ① 池化压缩：T → T/factor
            x_short = self.pre_lstm_pool(x)          # (B, 8, T/f)
            x_short = x_short.permute(0, 2, 1)       # (B, T/f, 8)
            x_short, _ = self.lstm(x_short)           # (B, T/f, hidden*2)
            x_short = x_short.permute(0, 2, 1)       # (B, hidden*2, T/f)
            # ② 线性插值还原：T/factor → T
            x = F.interpolate(x_short, size=T_full,
                              mode='linear', align_corners=False)
        else:
            # factor=1：直接处理全长序列（显存充足时使用）
            x = x.permute(0, 2, 1)                   # (B, T, 8)
            x, _ = self.lstm(x)                       # (B, T, hidden*2)
            x = x.permute(0, 2, 1)                   # (B, hidden*2, T)

        x = self.out_conv(x)             # (B, 1, T)
        x = torch.sigmoid(x).squeeze(1) # (B, T)
        return x


# ── EQTransformer 主模型 ──────────────────────────────────────

class EQTransformer(nn.Module):
    """
    EQTransformer baseline（PyTorch 实现，对应原论文架构）。
    输入 : (B, 3, T)
    输出 : (B, T, 3)   通道顺序 [detector, P-prob, S-prob]

    lstm_pool_factor : LSTM 前的序列压缩倍数，用于控制 T=6000 时的显存
      factor=1  → 原始长度，显存最大（仅适合 >24 GB 显卡或 T≤3000）
      factor=8  → 推荐，T=6000 压到 750，精度损失极小
      factor=16 → 最省显存，适合 <8 GB 显卡
    """

    ENC_CHANNELS = [8, 16, 16, 32, 32, 64, 64]

    def __init__(self, in_channels=3, kernel_size=3, lstm_hidden=64,
                 lstm_pool_factor=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.encoder = EQTEncoder(in_channels, kernel_size, n_heads, dropout)

        branch_kwargs = dict(enc_channels=self.ENC_CHANNELS,
                             kernel_size=kernel_size,
                             lstm_hidden=lstm_hidden,
                             lstm_pool_factor=lstm_pool_factor,
                             dropout=dropout)
        self.det_branch = EQTDecoderBranch(**branch_kwargs)
        self.p_branch   = EQTDecoderBranch(**branch_kwargs)
        self.s_branch   = EQTDecoderBranch(**branch_kwargs)

    def forward(self, x):
        feat, skips = self.encoder(x)
        det = self.det_branch(feat, skips)
        p   = self.p_branch(feat, skips)
        s   = self.s_branch(feat, skips)
        return torch.stack([det, p, s], dim=-1)  # (B, T, 3)


# ==================== 损失函数 ====================
class EQTLoss(nn.Module):
    """
    三分支独立 Binary Cross-Entropy 之和。
      pred, target: (B, T, 3)  — [detector, P, S]
    权重可调节三分支相对重要性。
    """

    def __init__(self, w_det=0.05, w_p=0.40, w_s=0.55):
        super().__init__()
        self.w_det = w_det
        self.w_p   = w_p
        self.w_s   = w_s

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        bce = -(target * torch.log(pred) +
                (1 - target) * torch.log(1 - pred))   # (B, T, 3)
        loss_det = bce[..., 0].mean()
        loss_p   = bce[..., 1].mean()
        loss_s   = bce[..., 2].mean()
        return self.w_det * loss_det + self.w_p * loss_p + self.w_s * loss_s


# ==================== 批量结果可视化 ====================
def plot_batch_results(y_true_p, y_pred_p, y_true_s, y_pred_s,
                       save_path='test_results_eqt.pdf',
                       window_size=WINDOW_SIZE):
    """
    批量推断结果综合可视化（P/S 波残差分析）。
    与 PhaseNet 版本完全一致，仅标题不同。
    """
    # P 波
    y_true_p_s  = y_true_p * window_size
    y_pred_p_s  = y_pred_p * window_size
    residuals_p = y_pred_p_s - y_true_p_s
    errors_p    = np.abs(residuals_p)
    mu_p, std_p = np.mean(residuals_p), np.std(residuals_p)
    mae_p    = np.mean(errors_p)
    rmse_p   = np.sqrt(np.mean(residuals_p ** 2))
    acc10_p  = np.mean(errors_p <= 10) * 100
    acc20_p  = np.mean(errors_p <= 20) * 100
    acc50_p  = np.mean(errors_p <= 50) * 100

    # S 波
    y_true_s_s  = y_true_s * window_size
    y_pred_s_s  = y_pred_s * window_size
    residuals_s = y_pred_s_s - y_true_s_s
    errors_s    = np.abs(residuals_s)
    mu_s, std_s = np.mean(residuals_s), np.std(residuals_s)
    mae_s    = np.mean(errors_s)
    rmse_s   = np.sqrt(np.mean(residuals_s ** 2))
    acc10_s  = np.mean(errors_s <= 10) * 100
    acc20_s  = np.mean(errors_s <= 20) * 100
    acc50_s  = np.mean(errors_s <= 50) * 100

    fig = plt.figure(figsize=(24, 18))
    gs  = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # ── P 波散点图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true_p_s, y_pred_p_s, s=1, alpha=0.3, c='steelblue')
    ax1.plot([0, window_size], [0, window_size], 'r--', lw=1.5)
    ax1.set_xlabel('True (samples)'); ax1.set_ylabel('Pred (samples)')
    ax1.set_title('P-wave: True vs Pred', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ── S 波散点图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_true_s_s, y_pred_s_s, s=1, alpha=0.3, c='tomato')
    ax2.plot([0, window_size], [0, window_size], 'r--', lw=1.5)
    ax2.set_xlabel('True (samples)'); ax2.set_ylabel('Pred (samples)')
    ax2.set_title('S-wave: True vs Pred', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ── P 波残差分布
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals_p, bins=80, density=True, alpha=0.7, color='steelblue',
             edgecolor='black', range=(-150, 150))
    x_fit = np.linspace(-150, 150, 300)
    ax3.plot(x_fit, norm.pdf(x_fit, mu_p, std_p), 'r-', lw=2,
             label=f'μ={mu_p:.1f}, σ={std_p:.1f}')
    ax3.axvline(0, color='k', lw=2, alpha=0.8)
    ax3.set_title('P-wave: Residual Dist.', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ── S 波残差分布
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(residuals_s, bins=80, density=True, alpha=0.7, color='tomato',
             edgecolor='black', range=(-150, 150))
    ax4.plot(x_fit, norm.pdf(x_fit, mu_s, std_s), 'navy', lw=2,
             label=f'μ={mu_s:.1f}, σ={std_s:.1f}')
    ax4.axvline(0, color='k', lw=2, alpha=0.8)
    ax4.set_title('S-wave: Residual Dist.', fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    # ── P CDF
    ax5 = fig.add_subplot(gs[1, 0])
    sorted_err = np.sort(errors_p)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax5.plot(sorted_err, cdf, lw=2, color='steelblue')
    ax5.axvline(10, color='g',      lw=2, ls='--', label=f'10: {acc10_p:.1f}%')
    ax5.axvline(20, color='orange', lw=2, ls='--', label=f'20: {acc20_p:.1f}%')
    ax5.axvline(50, color='r',      lw=2, ls='--', label=f'50: {acc50_p:.1f}%')
    ax5.set_title('P-wave: CDF', fontweight='bold')
    ax5.set_xlim(0, 200); ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # ── S CDF
    ax6 = fig.add_subplot(gs[1, 1])
    sorted_err = np.sort(errors_s)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax6.plot(sorted_err, cdf, lw=2, color='tomato')
    ax6.axvline(10, color='g',      lw=2, ls='--', label=f'10: {acc10_s:.1f}%')
    ax6.axvline(20, color='orange', lw=2, ls='--', label=f'20: {acc20_s:.1f}%')
    ax6.axvline(50, color='r',      lw=2, ls='--', label=f'50: {acc50_s:.1f}%')
    ax6.set_title('S-wave: CDF', fontweight='bold')
    ax6.set_xlim(0, 200); ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    # ── P 密度热图
    ax7 = fig.add_subplot(gs[1, 2])
    h, xe, ye = np.histogram2d(y_true_p_s, y_pred_p_s,
                               bins=60, range=[[0, window_size], [0, window_size]])
    im = ax7.imshow(np.log10(h.T + 1), origin='lower', aspect='auto',
                    extent=[0, window_size, 0, window_size], cmap='hot')
    ax7.plot([0, window_size], [0, window_size], 'cyan', lw=2)
    ax7.set_title('P-wave: Density Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax7, label='log10(N+1)')

    # ── S 密度热图
    ax8 = fig.add_subplot(gs[1, 3])
    h, xe, ye = np.histogram2d(y_true_s_s, y_pred_s_s,
                               bins=60, range=[[0, window_size], [0, window_size]])
    im = ax8.imshow(np.log10(h.T + 1), origin='lower', aspect='auto',
                    extent=[0, window_size, 0, window_size], cmap='hot')
    ax8.plot([0, window_size], [0, window_size], 'cyan', lw=2)
    ax8.set_title('S-wave: Density Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax8, label='log10(N+1)')

    # ── P 统计摘要
    ax9 = fig.add_subplot(gs[2, 0:2]); ax9.axis('off')
    skew_p = stats.skew(residuals_p);  kurt_p = stats.kurtosis(residuals_p)
    n_samp = min(5000, len(residuals_p))
    rng    = np.random.default_rng(0)
    _, shapiro_p = stats.shapiro(rng.choice(residuals_p, n_samp, replace=False))
    summary_p = (
        f"P-wave Statistics  |  N={len(y_true_p):,}\n"
        f"Residual: μ={mu_p:.2f}  σ={std_p:.2f}  Skew={skew_p:.3f}  Kurt={kurt_p:.3f}\n"
        f"MAE={mae_p:.2f}  RMSE={rmse_p:.2f}  Median={np.median(errors_p):.2f} (samples)\n"
        f"Accuracy ≤10={acc10_p:.1f}%  ≤20={acc20_p:.1f}%  ≤50={acc50_p:.1f}%\n"
        f"Shapiro-Wilk p={shapiro_p:.4f}  "
        f"{'✓ Normal' if shapiro_p > 0.05 else '✗ Non-normal'}"
    )
    ax9.text(0.05, 0.5, summary_p, fontsize=11, family='monospace',
             va='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # ── S 统计摘要
    ax10 = fig.add_subplot(gs[2, 2:4]); ax10.axis('off')
    skew_s = stats.skew(residuals_s);  kurt_s = stats.kurtosis(residuals_s)
    n_samp = min(5000, len(residuals_s))
    _, shapiro_s = stats.shapiro(rng.choice(residuals_s, n_samp, replace=False))
    summary_s = (
        f"S-wave Statistics  |  N={len(y_true_s):,}\n"
        f"Residual: μ={mu_s:.2f}  σ={std_s:.2f}  Skew={skew_s:.3f}  Kurt={kurt_s:.3f}\n"
        f"MAE={mae_s:.2f}  RMSE={rmse_s:.2f}  Median={np.median(errors_s):.2f} (samples)\n"
        f"Accuracy ≤10={acc10_s:.1f}%  ≤20={acc20_s:.1f}%  ≤50={acc50_s:.1f}%\n"
        f"Shapiro-Wilk p={shapiro_s:.4f}  "
        f"{'✓ Normal' if shapiro_s > 0.05 else '✗ Non-normal'}"
    )
    ax10.text(0.05, 0.5, summary_s, fontsize=11, family='monospace',
              va='center',
              bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    fig.suptitle('EQTransformer Baseline — Test Set Results',
                 fontsize=16, fontweight='bold', y=0.998)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  测试结果图已保存: {save_path}")


# ==================== 训练 / 验证 ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for wave, label in loader:
        wave, label = wave.to(device), label.to(device)
        pred = model(wave)                        # (B, T, 3)
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
        pred = model(wave)                        # (B, T, 3)
        total_loss += criterion(pred, label).item()

        pred_np  = pred.cpu().numpy()
        label_np = label.cpu().numpy()
        B = pred_np.shape[0]

        for b in range(B):
            # P 拾取（通道 1）
            true_p = np.argmax(label_np[b, :, 1])
            pred_p = np.argmax(pred_np[b, :, 1])
            if label_np[b, true_p, 1] > 0.01:
                p_errors.append(abs(pred_p - true_p))
                p_true_list.append(true_p / WINDOW_SIZE)
                p_pred_list.append(pred_p / WINDOW_SIZE)

            # S 拾取（通道 2）
            true_s = np.argmax(label_np[b, :, 2])
            pred_s = np.argmax(pred_np[b, :, 2])
            if label_np[b, true_s, 2] > 0.01:
                s_errors.append(abs(pred_s - true_s))
                s_true_list.append(true_s / WINDOW_SIZE)
                s_pred_list.append(pred_s / WINDOW_SIZE)

    avg_loss = total_loss / max(1, len(loader))
    p_mae = np.mean(p_errors) if p_errors else float('nan')
    s_mae = np.mean(s_errors) if s_errors else float('nan')
    return avg_loss, p_mae, s_mae, p_true_list, p_pred_list, s_true_list, s_pred_list


# ==================== 主函数 ====================
def main(seed=42):
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("EQTransformer Baseline  (7-layer ResConv Enc + 3 LSTM-Attn Dec)")
    print(f"Window size: {WINDOW_SIZE} samples  |  Random seed: {seed}")
    print("=" * 60)

    # ===================== 事件级划分（回应审稿人） =====================
    print(f"Signal chunks: {CFG['train_chunks']}  "
          f"Noise: {CFG['noise_chunks']}  Pool: {CFG['pool_chunk']}")
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

    def _load_sig(split, max_samples):
        w_l, y_l = [], []
        for c in CFG['train_chunks']:
            names = names_for_split(meta, split, c, max_samples, seed)
            if not names:
                continue
            hf, cf = _paths(c)
            print(f"\n[{split}] {c}: 指定 {len(names)} 条")
            w, y = load_eqt_data(hf, cf, window_size=WINDOW_SIZE,
                                 max_samples=max_samples, seed=seed,
                                 trace_names=names)
            if len(w) > 0:
                w_l.append(w); y_l.append(y)
        if not w_l:
            return (np.zeros((0, WINDOW_SIZE, 3), np.float32),
                    np.zeros((0, WINDOW_SIZE, 3), np.float32))
        return np.concatenate(w_l, axis=0), np.concatenate(y_l, axis=0)

    sig_tr_w, sig_tr_l = _load_sig('train', CFG['max_samples'])
    sig_va_w, sig_va_l = _load_sig('val',   CFG['val_samples'])
    sig_te_w, sig_te_l = _load_sig('test',  CFG['test_samples'])
    print(f"\nSignal  Train: {len(sig_tr_w)}  Val: {len(sig_va_w)}  "
          f"Test: {len(sig_te_w)}")

    # ---- 噪声（chunk1）：无事件标签，且与信号 trace 完全不重叠，仍按 trace 划分 ----
    noise_waves, noise_labels = np.array([]), np.array([])
    NC = None
    if CFG['noise_chunks']:
        # 噪声总量（默认 20000）按与信号相同的 80/15/5 划分 -> 16000/3000/1000
        n_noise = max(len(CFG['noise_chunks']), 1)
        NC = split_caps(CFG['noise_samples'], CFG['ratios'], n_noise)
        noise_cap = NC['train'] + NC['val'] + NC['test']
        print(f"[噪声] 总量={CFG['noise_samples']} 比例={CFG['ratios']} -> "
              f"每个 chunk 上限 {noise_cap}（train/val/test = {NC['train']}/"
              f"{NC['val']}/{NC['test']}）；合计 {NC['total']} 条")
        nw_l, nl_l = [], []
        for c in CFG['noise_chunks']:
            hf, cf = _paths(c)
            print(f"\nLoading noise: {hf} & {cf}")
            nw, nl = load_noise_data(hf, cf, window_size=WINDOW_SIZE,
                                     max_samples=noise_cap,
                                     seed=seed)
            if len(nw) > 0:
                nw_l.append(nw); nl_l.append(nl)
        if nw_l:
            noise_waves = np.concatenate(nw_l, axis=0)
            noise_labels = np.concatenate(nl_l, axis=0)
            print(f"Total noise: {len(noise_waves)}")

    # ---- 波形库（排除 val/test 事件）----
    wave_pool = None
    if CFG['pool_chunk']:
        pm = load_chunks(_data_dir(), [CFG['pool_chunk']])
        pn = pool_names(pm, exclude, CFG['pool_samples'], seed)
        phf, pcf = _paths(CFG['pool_chunk'])
        pw, _ = load_eqt_data(phf, pcf, window_size=WINDOW_SIZE,
                              max_samples=CFG['pool_samples'], seed=seed,
                              trace_names=pn)
        if len(pw) > 0:
            wave_pool = pw
            print(f"Waveform pool: {len(wave_pool)} samples "
                  f"(已排除 val/test 事件)")

    # ---- 组装三个集合 ----
    X_tr, y_tr = sig_tr_w, sig_tr_l
    X_va, y_va = sig_va_w, sig_va_l
    X_te, y_te = sig_te_w, sig_te_l
    if len(noise_waves) > 0:
        # 原实现用 train_test_split(0.2/0.2) = 64/16/20；这里改成与信号一致的 80/15/5
        _perm = np.random.default_rng(seed).permutation(len(noise_waves))
        n_tr, n_va = NC['train'], NC['val']
        _pi = {'train': _perm[:n_tr], 'val': _perm[n_tr:n_tr + n_va],
               'test': _perm[n_tr + n_va:]}
        n_tr_w, ny_tr_w = noise_waves[_pi['train']], noise_labels[_pi['train']]
        n_va_w, ny_va_w = noise_waves[_pi['val']], noise_labels[_pi['val']]
        n_te_w, ny_te_w = noise_waves[_pi['test']], noise_labels[_pi['test']]
        X_tr = np.concatenate([X_tr, n_tr_w], axis=0)
        y_tr = np.concatenate([y_tr, ny_tr_w], axis=0)
        X_va = np.concatenate([X_va, n_va_w], axis=0)
        y_va = np.concatenate([y_va, ny_va_w], axis=0)
        X_te = np.concatenate([X_te, n_te_w], axis=0)
        y_te = np.concatenate([y_te, ny_te_w], axis=0)
        print(f"  噪声并入: train={len(n_tr_w)} val={len(n_va_w)} "
              f"test={len(n_te_w)}")
        print(f"\nAfter noise mix — Train: {len(X_tr)} "
              f"(sig={len(sig_tr_w)}, noi={len(n_tr_w)})  Val: {len(X_va)}  "
              f"Test: {len(X_te)}")
    else:
        print(f"\nNo noise loaded. Train: {len(X_tr)}  Val: {len(X_va)}  "
              f"Test: {len(X_te)}")

    # ---- 数据增强（信号用全量，噪声只用基础增强）----
    if CFG.get('dry_run'):
        _dry_report(X_tr, None, None, X_va, None, None, X_te, None,
                    None, wave_pool, meta)
        return

    signal_aug = SimpleAugmentation(
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
    noise_aug = NoiseAugmentation(
        noise_std_range=(0.01, 0.30),
        scale_range=(0.5, 2.0),
        noise_prob=0.50,
        scale_prob=0.30,
        dropout_prob=0.30,
    )

    # 混搭 Dataset：对每条样本动态判断用哪个增强器
    class MixedDataset(Dataset):
        def __init__(self, waves, labels, is_train, signal_aug, noise_aug):
            self.waves = waves
            self.labels = labels
            self.is_train = is_train
            self.signal_aug = signal_aug
            self.noise_aug = noise_aug

        def __len__(self):
            return len(self.waves)

        def __getitem__(self, idx):
            wave = self.waves[idx].copy()
            label = self.labels[idx].copy()
            is_noise = (label.max() < 1e-6)  # 全零标签 = 噪声
            if self.is_train:
                if is_noise:
                    wave = self.noise_aug(wave)
                else:
                    wave = self.signal_aug(wave)
            wave_t = torch.from_numpy(wave.T).float()
            return wave_t, torch.from_numpy(label).float()

    train_ds = MixedDataset(X_tr, y_tr, is_train=True,
                            signal_aug=signal_aug, noise_aug=noise_aug)
    val_ds   = EQTDataset(X_va, y_va, is_train=False)
    test_ds  = EQTDataset(X_te, y_te, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128,        shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128,        shuffle=False,
                              num_workers=0, pin_memory=True)

    # ---- 模型 ----
    model = EQTransformer(in_channels=3, kernel_size=3,
                          lstm_hidden=64, lstm_pool_factor=1,
                          n_heads=8, dropout=0.1).to(device)    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = EQTLoss(w_det=0.05, w_p=0.40, w_s=0.55)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                 weight_decay=1e-5)   # 与 BiMamba/PhaseNet 统一
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 70, 90], gamma=0.3)

    # ---- 训练循环 ----
    best_val = float('inf')
    tr_losses, va_losses = [], []
    va_p_maes, va_s_maes = [], []
    start_time = time.time()

    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
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
            elapsed = time.time() - start_time
            avg_ep  = elapsed / (epoch + 1)
            remain  = avg_ep * (NUM_EPOCHS - epoch - 1)

            def fmt(s):
                h, m, s = int(s // 3600), int(s % 3600 // 60), int(s % 60)
                return (f"{h}h {m}m {s}s" if h else
                        f"{m}m {s}s"      if m else f"{s}s")

            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
                  f"tr={tr:.6f}  val={vl:.6f}  "
                  f"P_MAE={vp:.1f}samp  S_MAE={vs:.1f}samp  "
                  f"lr={lr_now:.2e}  best={best_val:.6f}  "
                  f"elapsed={fmt(elapsed)}  remaining={fmt(remain)}")

    # ---- 最终测试 ----
    print("\nFinal test evaluation:")
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    te_loss, te_p, te_s, p_true_list, p_pred_list, s_true_list, s_pred_list = \
        evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss : {te_loss:.6f}")
    print(f"  P-wave MAE: {te_p:.2f} samples  ({te_p / 100 * 1000:.1f} ms @ 100 Hz)")
    print(f"  S-wave MAE: {te_s:.2f} samples  ({te_s / 100 * 1000:.1f} ms @ 100 Hz)")

    if p_true_list and s_true_list:
        plot_batch_results(
            np.array(p_true_list), np.array(p_pred_list),
            np.array(s_true_list), np.array(s_pred_list),
            save_path='test_set_results_eqt.pdf',
            window_size=WINDOW_SIZE
        )

    # ---- 训练曲线 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = np.arange(NUM_EPOCHS)

    axes[0].plot(ep, tr_losses, label='Train')
    axes[0].plot(ep, va_losses, label='Val')
    axes[0].set_title('BCE Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(ep, va_p_maes, label='P-wave')
    axes[1].plot(ep, va_s_maes, label='S-wave')
    axes[1].set_title('Val Pick MAE (samples)'); axes[1].set_xlabel('Epoch')
    axes[1].legend()

    model.eval()
    with torch.no_grad():
        sw, sl = test_ds[0]
        po = model(sw.unsqueeze(0).to(device))[0].cpu().numpy()  # (T, 3)
        lo = sl.numpy()
        t  = np.arange(WINDOW_SIZE) / 100.0

    axes[2].plot(t, lo[:, 1], 'b--', lw=1,   label='Label P')
    axes[2].plot(t, lo[:, 2], 'r--', lw=1,   label='Label S')
    axes[2].plot(t, po[:, 1], 'b-',  lw=1.5, label='Pred P')
    axes[2].plot(t, po[:, 2], 'r-',  lw=1.5, label='Pred S')
    axes[2].set_title('Sample Prediction'); axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('eqtransformer_results.pdf', dpi=150)
    plt.close()
    print("Plot saved to 'eqtransformer_results.pdf'")
    print(f"Model saved to '{MODEL_SAVE}'")


if __name__ == '__main__':
    main(seed=apply_cli())

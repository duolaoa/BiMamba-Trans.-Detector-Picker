# 纯地震检测模型 - 三分量版本（双向Mamba版）
# 1. 纯检测模型：仅输出P/S检测（有/无地震），不回归到时
# 2. 检测标签：从P波开始，1.4×(S-P) 长度标记为地震相位
# 3. 输入：6000点 raw 三分量波形（不提取特征）
# 4. 去掉所有到时回归相关的损失函数和模型头
import pickle
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
import warnings
import time
import os
import glob
from scipy.stats import norm, gaussian_kde
from scipy import stats
import math

# =========================================================================== #
# 事件级划分版（回应审稿人：Please split by event）
# 本文件是 PS_Detection.py 的副本，仅改造**训练入口 main() 的数据划分**：
#   原实现按 trace 随机划分，实测 65.1% 的测试 trace 其事件也出现在训练集；
#   现改为按事件划分，并与拾取模型/双头检测器共用同一张 --split-table。
# 模型结构、损失、增强、训练循环均未改动。
# =========================================================================== #
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from event_split import (assert_disjoint, build_split_table, describe_splits,
                         events_in, load_chunks, load_split_table,
                         names_for_split, pool_names, save_split_table,
                         split_caps)

_1DSIGNAL = os.environ.get(
    'STEAD_DIR', './')
PRESETS = {
    'det_pn197k':      dict(proj_dim=16,  hidden_dim=40),
    'det_pm322k':      dict(proj_dim=32,  hidden_dim=48),
    'det_eqt466k':     dict(proj_dim=56,  hidden_dim=48),
    'det_ctrl813k':    dict(proj_dim=96,  hidden_dim=48),
    'det_abl_nomamba': dict(proj_dim=454, hidden_dim=48, mamba_mode='none'),
    'det_abl_notrans': dict(proj_dim=70,  hidden_dim=48, use_transformer=False),
    'det_abl_lfem':    dict(proj_dim=32,  hidden_dim=48, use_lfem=True),
}

CFG = {
    'data_dir': _1DSIGNAL,
    'train_chunks': ['chunk2', 'chunk3'],
    'pool_chunk': 'chunk4',
    'noise_chunks': ['chunk1'],
    # 事件级划分比例：train/val/test = 0.8/0.15/0.05
    'ratios': (0.8, 0.15, 0.05),
    # 信号总量；由 split_caps() 按上面的比例折算成每个 chunk 的上限
    'total_samples': 100000,
    'max_samples': 32000,
    'val_samples': 5000,
    'test_samples': 5000,
    'pool_samples': 15000,
    'noise_samples': 20000,
    'split_table': '',
    'save_split_table': 'event_split_table.csv',
    'dry_run': False,
}


def apply_cli(argv=None):
    """命令行覆盖 CFG，返回 seed。"""
    global BAND_LOW, BAND_HIGH, USE_SCALER
    global POS_WEIGHT, FOCAL_GAMMA, FOCAL_ON_NOISE
    import argparse
    ap = argparse.ArgumentParser(description='PS_Detection 事件级划分训练')
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
    ap.add_argument('--noise-samples', type=int, default=CFG['noise_samples'])
    ap.add_argument('--split-table', default=CFG['split_table'])
    ap.add_argument('--save-split-table', default=CFG['save_split_table'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--window-size', type=int, default=6000,
                    help='检测窗长（采样点）；原实现 6000=60s @100Hz，'
                         '3000=30s（与 PhaseNet/EQT 及拾取器一致）')
    ap.add_argument('--proj-dim', type=int, default=96)
    ap.add_argument('--preset', default=None,
                    help='参数量/模块命名预设；与显式宽度参数同用时以预设为准')
    # ── 模块消融开关（检测器默认 use_lfem=False，前端是普通 Conv1d）──
    ap.add_argument('--mamba-mode', choices=['none', 'unidirectional', 'bidirectional'],
                    default='bidirectional', help='none=去掉 BiMamba')
    ap.add_argument('--no-transformer', dest='use_transformer', action='store_false',
                    help='去掉 Transformer 编码器')
    ap.add_argument('--use-lfem', dest='use_lfem', action='store_true',
                    help='把普通 Conv1d 前端换成 LFEM1D（默认关）')
    ap.set_defaults(use_transformer=True, use_lfem=False)
    ap.add_argument('--hidden-dim', type=int, default=48)
    ap.add_argument('--mamba-layers', type=int, default=3)
    ap.add_argument('--d-state', type=int, default=16)
    ap.add_argument('--expand', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=100,
                    help='训练轮数（主路径原为硬编码 100）')
    ap.add_argument('--tag', default='',
                    help='权重名后缀: best_detection_model_3C<tag>.pth')
    # ── 优先级1：检测损失的正样本加权 / focal ──
    ap.add_argument('--pos-weight', type=float, default=3.0,
                    help='软化正样本权重 w=1+(pw-1)*target（1.0 即原纯 BCE）；默认 3.0')
    ap.add_argument('--focal-gamma', type=float, default=0.0,
                    help='focal 调制指数 (1-p_t)^gamma；0=关闭（默认），建议 1.0~2.0')
    ap.add_argument('--focal-on-noise', action='store_true',
                    help='focal 也作用于噪声抑制项（默认只作用于 P/S 信号项）')
    # ── 优先级2：预处理频带与 scaler ──
    ap.add_argument('--band-low', type=float, default=BAND_LOW,
                    help='带通下限 Hz（与拾取器统一为 1.0；原实现为 2.0）')
    ap.add_argument('--band-high', type=float, default=BAND_HIGH,
                    help='带通上限 Hz（与拾取器统一为 40.0；原实现为 30.0）')
    ap.add_argument('--scaler', action='store_true',
                    help='保留 StandardScaler（默认关闭，与三个拾取器统一为逐通道 /std）')
    a = ap.parse_args(argv)
    if getattr(a, 'preset', None):
        _p = PRESETS[a.preset]
        for _k, _v in _p.items():
            setattr(a, _k, _v)
        print(f'[preset] {a.preset} -> ' +
              ' '.join(f'{k}={v}' for k, v in _p.items()))
    CFG.update(window_size=a.window_size,
               mamba_mode=a.mamba_mode, use_transformer=a.use_transformer,
               use_lfem=a.use_lfem,
               proj_dim=a.proj_dim, hidden_dim=a.hidden_dim,
               mamba_layers=a.mamba_layers, d_state=a.d_state, expand=a.expand,
               epochs=a.epochs, tag=a.tag,
               data_dir=a.data_dir, train_chunks=a.train_chunks,
               pool_chunk=a.pool_chunk, noise_chunks=a.noise_chunks,
               ratios=tuple(float(x) for x in a.ratios.split(',')),
               max_samples=a.max_samples, val_samples=a.val_samples,
               test_samples=a.test_samples, pool_samples=a.pool_samples,
               noise_samples=a.noise_samples, split_table=a.split_table,
               save_split_table=a.save_split_table,
               total_samples=a.total_samples, dry_run=bool(a.dry_run))
    print('[事件级划分] data_dir=%s chunks=%s pool=%s ratios=%s split_table=%r'
          % (CFG['data_dir'], CFG['train_chunks'], CFG['pool_chunk'],
             CFG['ratios'], CFG['split_table']))
    # 优先级2：把频带/scaler 开关写进模块全局（simple_preprocessing / Dataset 读它们）
    BAND_LOW, BAND_HIGH = float(a.band_low), float(a.band_high)
    USE_SCALER = bool(a.scaler)
    # 优先级1：损失参数写进全局，供 train_model 内的 criterion 读取
    POS_WEIGHT, FOCAL_GAMMA, FOCAL_ON_NOISE = (
        float(a.pos_weight), float(a.focal_gamma), bool(a.focal_on_noise))
    print('[优先级1] 检测损失 pos_weight=%.2f  focal_gamma=%.2f  focal_on_noise=%s'
          % (POS_WEIGHT, FOCAL_GAMMA, FOCAL_ON_NOISE))
    print('[优先级2] 预处理频带 %.1f-%.1f Hz  StandardScaler=%s'
          % (BAND_LOW, BAND_HIGH, 'ON' if USE_SCALER else 'OFF(统一为逐通道/std)'))
    return a.seed


def _event_meta(seed):
    """返回 (meta, exclude_events)：优先复用划分表，保证各模型同一划分。"""
    reused = bool(CFG['split_table']) and os.path.exists(CFG['split_table'])
    if reused:
        print('使用已有事件划分表: %s' % CFG['split_table'])
        meta = load_split_table(CFG['split_table'])
    else:
        print('构建事件级划分 ...')
        meta = build_split_table(load_chunks(_data_dir(),
                                             CFG['train_chunks']),
                                 CFG['ratios'], seed)
    print('\n' + describe_splits(meta))
    assert_disjoint(meta)
    if CFG['save_split_table'] and not reused:
        save_split_table(meta, CFG['save_split_table'])
    return meta, (events_in(meta, 'val') | events_in(meta, 'test'))


def _data_dir():
    """data_dir 为相对路径时按脚本所在目录解析，避免受当前工作目录影响。"""
    d = CFG['data_dir']
    return d if os.path.isabs(d) else os.path.normpath(os.path.join(_HERE, d))


def _paths(chunk):
    return (os.path.join(_data_dir(), chunk + '.hdf5'),
            os.path.join(_data_dir(), chunk + '.csv'))

warnings.filterwarnings('ignore')

# 三个分量的列索引（STEAD数据集约定）
# STEAD: columns 0=E, 1=N, 2=Z （可根据实际数据集调整）
CH_Z = 2   # 垂直分量
CH_N = 1   # 南北分量
CH_E = 0   # 东西分量
CHANNEL_INDICES = [CH_Z, CH_N, CH_E]   # 提取顺序 Z, N, E
N_CHANNELS = len(CHANNEL_INDICES)       # = 3


# =================== 简化的数据预处理 ===================
# ============ 预处理频带与标准化（优先级2：与三个拾取器统一）============
# 原实现：2-30 Hz 带通 + 额外的 StandardScaler('scaler_3C.pkl')。
# 三个拾取器统一为：1-40 Hz + 逐通道 /std。30-40 Hz 恰是尖锐初动（低信噪比事件
# 最可见的特征）所在，去掉它会削弱可检测性；scaler 叠加也会改变分数尺度。
# 由 apply_cli 的 --band-low/--band-high 与 --scaler 控制。
POS_WEIGHT, FOCAL_GAMMA, FOCAL_ON_NOISE = 3.0, 0.0, False   # 优先级1（CLI 可覆盖）
BAND_LOW   = 1.0        # Hz
BAND_HIGH  = 40.0       # Hz
USE_SCALER = False      # False = 不再叠加 StandardScaler（与拾取器统一）


def simple_preprocessing(data, fs=100):
    """简化预处理：去趋势 + 带通(BAND_LOW-BAND_HIGH Hz) + 逐通道 /std。"""
    # 去趋势
    data = detrend(data, type='linear')

    # 带通滤波（默认 1-40 Hz，与 PhaseNet/EQTransformer/BiMamba 拾取器一致）
    nyq = fs / 2
    low, high = BAND_LOW / nyq, min(BAND_HIGH / nyq, 0.99)
    b, a = butter(2, [low, high], btype='band')
    data = filtfilt(b, a, data)

    # 标准归一化
    std = np.std(data)
    if std > 0:
        data = data / std

    return data.astype(np.float32)


def preprocess_3c(data_3c, fs=100):
    """
    对三分量数据分别进行预处理。
    输入: data_3c  shape (T, 3)  — 三列对应 [Z, N, E]
    输出: processed shape (T, 3)
    """
    processed = np.zeros_like(data_3c, dtype=np.float32)
    for c in range(data_3c.shape[1]):
        processed[:, c] = simple_preprocessing(data_3c[:, c], fs=fs)
    return processed

# =================== 地震相位检测标签生成 ===================
def generate_seismic_phase_label(window_size, p_center, s_center, scale=1.4):
    """
    生成地震相位矩形标签（纯检测用）。

    标签语义：从P波到时开始，持续 scale×(S-P) 长度的区间标记为正样本（=1.0），
    其余为0。该设计使得检测模型学习识别地震相位的存在区域。

    输入:
        window_size : 窗口大小
        p_center    : P波到时相对位置 [0, 1]
        s_center    : S波到时相对位置 [0, 1]，< 0 表示无效S到时
        scale       : 相位长度系数（默认1.4，即地震相位窗口覆盖P到超过S）
    输出:
        label : shape (window_size,)，正样本区内值=1.0，其余0.0
    """
    p_idx = int(round(p_center * window_size))
    if s_center >= 0:
        s_idx = int(round(s_center * window_size))
        phase_len = max(200, int(round(scale * (s_idx - p_idx))))
    else:
        phase_len = 1000   # 无S到时时默认长度（10s @ 100Hz）
    hi = min(window_size, p_idx + phase_len)
    label = np.zeros(window_size, dtype=np.float32)
    if hi > p_idx:
        label[p_idx:hi] = 1.0
    return label


def generate_s_phase_label(window_size, p_center, s_center, scale=0.5):
    """
    生成S波相位矩形标签。

    标签语义：从S波到时前0.15*(S-P)开始，持续 scale*(S-P) 长度。

    输入:
        window_size : 窗口大小
        p_center    : P波到时相对位置 [0, 1]
        s_center    : S波到时相对位置 [0, 1]
        scale       : 相位长度系数（默认0.5）
    输出:
        label : shape (window_size,)，正样本区内值=1.0，其余0.0
    """
    if s_center < 0:
        return np.zeros(window_size, dtype=np.float32)

    s_idx = int(round(s_center * window_size))
    p_idx = int(round(p_center * window_size))
    sp_gap = max(1, s_idx - p_idx)
    phase_len = max(100, int(round(scale * sp_gap)))
    start = max(0, s_idx - int(0.15 * sp_gap))
    hi = min(window_size, start + phase_len)
    label = np.zeros(window_size, dtype=np.float32)
    if hi > start:
        label[start:hi] = 1.0
    return label


# =================== 简化的数据增强 ===================
class SimpleAugmentation:
    """
    三分量精简数据增强，作用于特征矩阵 x (T, F)。

    参数
    ----
    per_ch : int
        每个分量占用的特征列数（如原始+差分=2）。
        原始波形列索引 = [0, per_ch, 2*per_ch]。
    n_ch : int
        分量数，固定为 3（Z, N, E）。
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
        用于采样第二事件的波形库，shape (M, T, F)，与主样本特征维度相同。
        传入 None 时退化为叠加随机带限噪声（仍有效果，但不如真实波形）。
    second_event_amp_range : (float, float)
        第二事件的相对幅度范围（相对于主信号 RMS），对数均匀采样。
        默认 (0.1, 1.0)：第二事件幅度为主信号的 10%–100%，
        覆盖远震微弱叠加到近震强叠加的全部情形。
    """

    def __init__(
        self,
        per_ch=1,
        n_ch=3,
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
        self.raw_cols              = [c * per_ch for c in range(n_ch)]
        self.all_ch_slices         = [slice(c * per_ch, (c + 1) * per_ch)
                                      for c in range(n_ch)]
        self.noise_std_range       = noise_std_range
        self.scale_range           = scale_range
        self.noise_prob            = noise_prob
        self.scale_prob            = scale_prob
        self.dropout_prob          = dropout_prob
        self.dropout_n_range       = dropout_n_range
        self.second_event_prob     = second_event_prob
        self.second_event_pool     = second_event_pool   # (M, T, F) or None
        self.second_event_amp_range = second_event_amp_range
        self.n_ch                  = n_ch
        self.per_ch                = per_ch

    def __call__(self, x):
        """
        x : (T, F) float32
        返回增强后的 (T, F) float32
        """
        x   = x.copy()
        rng = np.random.default_rng()
        T   = x.shape[0]

        # ── 1. 随机高斯噪声（只加在原始波形列）────────────────
        if rng.random() < self.noise_prob:
            log_lo = np.log(self.noise_std_range[0])
            log_hi = np.log(self.noise_std_range[1])
            std = np.exp(rng.uniform(log_lo, log_hi))
            for col in self.raw_cols:
                x[:, col] += rng.standard_normal(T).astype(np.float32) * std

        # ── 2. 随机幅度缩放（作用于所有特征列）────────────────
        if rng.random() < self.scale_prob:
            scale = np.exp(rng.uniform(
                np.log(self.scale_range[0]),
                np.log(self.scale_range[1])
            ))
            x *= scale

        # ── 3. 随机分量 Dropout（将整个分量的所有特征列置零）──
        if rng.random() < self.dropout_prob:
            n_drop = int(rng.integers(self.dropout_n_range[0],
                                      self.dropout_n_range[1] + 1))
            drop_chs = rng.choice(self.n_ch, size=n_drop, replace=False)
            for ch in drop_chs:
                x[:, self.all_ch_slices[ch]] = 0.0

        # ── 4. 第二地震事件叠加 ─────────────────────────────────
        if rng.random() < self.second_event_prob:
            if self.second_event_pool is not None:
                idx2 = int(rng.integers(0, len(self.second_event_pool)))
                x2 = self.second_event_pool[idx2].copy()   # (T, F)
                if x2.shape[0] != T:
                    if x2.shape[0] > T:
                        st = int(rng.integers(0, x2.shape[0] - T))
                        x2 = x2[st: st + T]
                    else:
                        pad = T - x2.shape[0]
                        x2 = np.pad(x2, ((0, pad), (0, 0)), mode='constant')
            else:
                x2 = np.zeros_like(x)
                for col in self.raw_cols:
                    burst = rng.standard_normal(T).astype(np.float32)
                    onset = int(rng.integers(0, max(1, T - T // 4)))
                    decay = np.exp(-np.arange(T - onset) / (T * 0.1)).astype(np.float32)
                    burst[onset:] *= decay
                    burst[:onset]  = 0.0
                    x2[:, col] = burst

            main_rms = np.sqrt(np.mean(x[:, self.raw_cols] ** 2)) + 1e-8
            sec_rms  = np.sqrt(np.mean(x2[:, self.raw_cols] ** 2)) + 1e-8
            amp = np.exp(rng.uniform(
                np.log(self.second_event_amp_range[0]),
                np.log(self.second_event_amp_range[1])
            ))
            x2_scaled = x2 * (main_rms / sec_rms * amp)
            x += x2_scaled.astype(np.float32)

        return x.astype(np.float32)


# =============== 数据集 =================
class LightweightWaveDataset(Dataset):
    def __init__(self, waves, p_targets, s_targets=None,
                 is_train=True, scaler=None, transforms=None,
                 window_size=6000, phase_scale=1.4, s_phase_scale=0.5):
        """
        waves           : (N, T, F)
        p_targets       : (N,)  P波相对位置；-2.0 = 噪声样本
        s_targets       : (N,)  S波相对位置；-1.0 = 无效；传 None 全部置 -1

        每条样本生成两种检测标签：
          ├── P相位标签 (phase_scale) → P检测头
          └── S相位标签 (s_phase_scale) → S检测头

        噪声样本（p_target == -2.0）两种标签均为全零。
        """
        if waves.ndim == 2:
            waves = waves[..., None]

        self.waves          = waves.astype(np.float32)
        self.p_targets      = p_targets.astype(np.float32)
        self.s_targets      = (s_targets.astype(np.float32)
                               if s_targets is not None
                               else np.full(len(p_targets), -1.0, dtype=np.float32))
        self.is_train       = is_train
        self.transforms     = transforms
        self.window_size    = window_size
        self.phase_scale    = phase_scale
        self.s_phase_scale  = s_phase_scale

        # 标准化
        if scaler is None:
            w2 = self.waves.reshape(-1, self.waves.shape[-1])
            if USE_SCALER:
                self.scaler = StandardScaler()
                self.scaler.fit(w2)
            else:
                # 优先级2：波形已在 simple_preprocessing 里逐通道 /std，
                # 这里不再叠加 StandardScaler（等价恒等变换，保留接口兼容）
                self.scaler = StandardScaler(with_mean=False, with_std=False)
                self.scaler.fit(w2)
        else:
            self.scaler = scaler

        w2  = self.waves.reshape(-1, self.waves.shape[-1])
        w2n = self.scaler.transform(w2)
        self.waves = w2n.reshape(self.waves.shape)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        x     = self.waves[idx].copy()
        p_pos = self.p_targets[idx]
        s_pos = self.s_targets[idx]

        if self.is_train and self.transforms:
            x = self.transforms(x)

        # ── 噪声样本（p_pos == -2.0）────────────────────────────────────
        is_noise_flag = 1.0 if p_pos <= -1.5 else 0.0

        if is_noise_flag:
            return (torch.from_numpy(x).float(),
                    torch.zeros(self.window_size),   # p_phase (全零)
                    torch.zeros(self.window_size),   # s_phase (全零)
                    torch.tensor(0.0),               # s_valid
                    torch.tensor(1.0))               # is_noise

        # ── 地震信号样本 ────────────────────────────────────────────────
        # P 波相位标签：从P波开始，1.4×(S-P) 长度
        p_phase = generate_seismic_phase_label(self.window_size, p_pos, s_pos,
                                                self.phase_scale)

        # S 波相位标签：S波附近窗口
        s_phase = generate_s_phase_label(self.window_size, p_pos, s_pos,
                                          self.s_phase_scale)
        s_valid = 1.0 if s_pos >= 0.0 else 0.0

        return (torch.from_numpy(x).float(),
                torch.tensor(p_phase,  dtype=torch.float32),   # P检测标签
                torch.tensor(s_phase,  dtype=torch.float32),   # S检测标签
                torch.tensor(s_valid,  dtype=torch.float32),
                torch.tensor(is_noise_flag, dtype=torch.float32))


# =============== 双向 Mamba 模块（基于官方 mamba_ssm 库）=================
from mamba_ssm import Mamba


class BidirectionalMambaLayer(nn.Module):
    """
    单层双向 Mamba：
      - 前向 Mamba（原始时间顺序）
      - 反向 Mamba（时间轴翻转后扫描，再翻转回来）
      - 两路输出相加后经 LayerNorm，保持维度 d_model 不变（残差结构）

    输入 : (B, T, d_model)
    输出 : (B, T, d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.fwd_mamba = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_out = self.fwd_mamba(x)                    # (B, T, d_model)
        bwd_out = self.bwd_mamba(x.flip(1)).flip(1)    # (B, T, d_model)
        return self.norm(x + fwd_out + bwd_out)        # 残差 + 双向融合


class BidirectionalMamba(nn.Module):
    """
    多层双向 Mamba 序列编码器。

    输入 : (B, T, d_model)         ← 此处 d_model = 96（proj 层输出）
    输出 : (B, T, d_model * 2)     ← 末层将前/后向特征拼接，维度翻倍

    参数
    ----
    d_model    : 输入特征维度
    d_state    : Mamba SSM 状态维度（默认 16）
    d_conv     : 局部卷积核大小（默认 4）
    expand     : 内部扩展比（默认 2）
    num_layers : 堆叠层数（默认 3）
    dropout    : 层间 Dropout 概率
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaLayer(d_model, d_state=d_state,
                                    d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.final_fwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.final_bwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = self.dropout(layer(out))          # (B, T, d_model)

        fwd_final = self.final_fwd(out)             # (B, T, d_model)
        bwd_final = self.final_bwd(out.flip(1)).flip(1)  # (B, T, d_model)
        return torch.cat([fwd_final, bwd_final], dim=-1)  # (B, T, d_model*2)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.register_buffer('pos_encoder',
                             self._build_sinusoidal(10000, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_sinusoidal(max_len, d_model):
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:d_model//2])
        return pe

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)
        return self.transformer_encoder(x)


class CrossChannelAttention(nn.Module):
    """轻量跨分量注意力。"""
    def __init__(self, n_feat=3, n_ch=N_CHANNELS):
        super().__init__()
        per_ch = n_feat // n_ch
        self.gate = nn.Sequential(
            nn.Linear(n_feat, n_ch),
            nn.Sigmoid()
        )
        self.per_ch = per_ch
        self.n_ch   = n_ch

    def forward(self, x):
        g = self.gate(x)                          # (B, T, 3)
        B, T, F = x.shape
        xr = x.view(B, T, self.n_ch, self.per_ch)
        gw = g.unsqueeze(-1)
        out = (xr * gw).view(B, T, F)
        return out + x


class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.dwconv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels
        )
        self.pwconv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x


class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, padding=2):
        super().__init__()
        self.dconv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

    def forward(self, x):
        return self.dconv(x)


class ChannelShuffle(nn.Module):
    """通道重排模块。"""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, time_steps = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, time_steps)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_channels, time_steps)
        return x


class LFEM1D(nn.Module):
    """1D 时序专用 LFEM（局部特征增强模块）。"""
    def __init__(self, dim, expansion=0.5, kernel_size=3):
        super().__init__()
        hidden_dim = int(dim * expansion)

        self.conv1x1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()

        branch_out_dim = hidden_dim // 3
        branch0_out = branch_out_dim
        branch1_out = branch_out_dim
        branch2_out = hidden_dim - branch0_out - branch1_out
        self.branch0 = nn.Conv1d(hidden_dim, branch0_out, kernel_size, padding=kernel_size//2)
        self.branch1 = DilatedConv1d(hidden_dim, branch1_out, kernel_size=3, dilation=2, padding=2)
        self.branch2 = DepthWiseConv1d(hidden_dim, branch2_out, kernel_size=3, padding=1)

        self.use_shuffle = (hidden_dim % 3 == 0)
        if self.use_shuffle:
            self.shuffle = nn.ChannelShuffle(groups=3)

        self.conv_out = nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.act(self.bn(self.conv1x1(x)))

        b0 = self.branch0(out)
        b1 = self.branch1(out)
        b2 = self.branch2(out)

        out = torch.cat([b0, b1, b2], dim=1)
        if self.use_shuffle:
            out = self.shuffle(out)

        out = self.bn_out(self.conv_out(out))
        return self.act(out + residual)


# =============== 纯检测模型架构（三分量版）=================
class LightweightPWaveNet(nn.Module):
    """
    三分量地震检测网络（纯检测，不到时回归）。

    参数
    ----
    input_dim     : 输入特征通道数，三分量版默认为 3
    hidden_dim    : 隐层维度，双向Mamba输出为 hidden_dim * 2
    use_transformer: True  → Mamba + TransformerEncoder 并联后融合
                    False → 仅使用 Mamba，去掉Transformer分支（更轻量）
    mamba_mode    : 'none' → 不使用 Mamba
                    'unidirectional' → 使用单向 Mamba
                    'bidirectional' → 使用双向 Mamba
    """

    def __init__(self, input_dim=3, hidden_dim=48,
                 use_transformer: bool = True, use_lfem: bool = False,
                 mamba_mode: str = 'bidirectional', use_cross_channel_attention: bool = False,
                 proj_dim: int = 96, mamba_layers: int = 3, d_state: int = 16,
                 expand: int = 2):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_lfem = use_lfem
        self.mamba_mode = mamba_mode
        self.use_cross_channel_attention = use_cross_channel_attention

        if use_cross_channel_attention:
            self.cross_ch_attn = CrossChannelAttention(n_feat=input_dim)
        else:
            self.cross_ch_attn = nn.Identity()

        # 特征提取器：LFEM1D 或普通 1D 卷积
        if use_lfem:
            self.lfem_block1 = LFEM1D(input_dim, expansion=1.0, kernel_size=7)
            self.pool1 = nn.MaxPool1d(2)

            self.lfem_block2 = LFEM1D(input_dim, expansion=1.0, kernel_size=5)
            self.pool2 = nn.MaxPool1d(2)

            self.lfem_block3 = LFEM1D(input_dim, expansion=1.5, kernel_size=3)
        else:
            self.lfem_block1 = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(input_dim),
                nn.ReLU()
            )
            self.pool1 = nn.MaxPool1d(2)

            self.lfem_block2 = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(input_dim),
                nn.ReLU()
            )
            self.pool2 = nn.MaxPool1d(2)

            self.lfem_block3 = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(input_dim),
                nn.ReLU()
            )

        # 第三级池化：进一步压缩时间维度，减少Mamba/Transformer计算量
        self.pool3 = nn.MaxPool1d(2)

        # 特征投影层（proj_dim 可配：默认 96 = 原实现）
        self.proj = nn.Conv1d(input_dim, proj_dim, kernel_size=1)

        # Mamba：三种模式
        if mamba_mode == 'bidirectional':
            self.mamba = BidirectionalMamba(proj_dim, d_state=d_state, d_conv=4,
                                           expand=expand,
                                           num_layers=mamba_layers, dropout=0.1)
            mamba_out_dim = proj_dim * 2   # 192
        elif mamba_mode == 'unidirectional':
            from mamba_ssm import Mamba
            self.mamba = nn.Sequential(*[
                Mamba(d_model=proj_dim, d_state=d_state, d_conv=4, expand=expand)
                for _ in range(mamba_layers)])
            mamba_out_dim = proj_dim   # 96
        elif mamba_mode == 'none':
            self.mamba = nn.Identity()
            mamba_out_dim = proj_dim   # 96
        else:
            raise ValueError(f"Invalid mamba_mode: {mamba_mode}")

        # Transformer 分支（可选）
        transformer_dim = hidden_dim * 2  # 128
        if use_transformer:
            self.transformer = TransformerEncoder(
                proj_dim, transformer_dim, num_layers=2, nhead=4, dropout=0.1)
            fusion_in_dim = mamba_out_dim + transformer_dim
        else:
            self.transformer = None
            fusion_in_dim = mamba_out_dim

        # 融合输出
        fused_dim = hidden_dim * 2   # 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ── P 波检测头（矩形相位标签，BCEWithLogitsLoss）─────────────────
        self.p_det_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # ── S 波检测头（矩形相位标签，BCEWithLogitsLoss）─────────────────
        self.s_det_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        返回 (p_det, s_det)：
          p_det : (batch, seq_len) logits → 送入 BCEWithLogitsLoss（P检测）
          s_det : (batch, seq_len) logits → 送入 BCEWithLogitsLoss（S检测）
        """
        batch_size, seq_len, _ = x.shape

        x = x.transpose(1, 2)           # (batch, features, time)

        # LFEM1D 特征提取
        x = self.lfem_block1(x)          # (batch, input_dim, time)
        x = self.pool1(x)                 # (batch, input_dim, time/2)
        x = self.lfem_block2(x)          # (batch, input_dim, time/2)
        x = self.pool2(x)                 # (batch, input_dim, time/4)
        x = self.lfem_block3(x)          # (batch, input_dim, time/4)
        x = self.pool3(x)                 # (batch, input_dim, time/8)
        x = self.proj(x)                  # (batch, 96, time/8)
        x = x.transpose(1, 2)           # (batch, time', 96)

        # Mamba
        mamba_out = self.mamba(x)     # (batch, time', hidden*2 或 hidden)

        # Transformer（可选）
        if self.use_transformer:
            transformer_out = self.transformer(x)
            combined = torch.cat([mamba_out, transformer_out], dim=-1)
        else:
            combined = mamba_out

        # 融合 → (batch, time', fused_dim)
        fused = self.fusion(combined)

        def _interp(logits):
            """将 time' 插值回 seq_len。"""
            return F.interpolate(
                logits.unsqueeze(1), size=seq_len,
                mode='linear', align_corners=False
            ).squeeze(1)

        # P 波检测头
        p_det = _interp(self.p_det_head(fused).squeeze(-1))     # (batch, seq_len)

        # S 波检测头
        s_det = _interp(self.s_det_head(fused).squeeze(-1))     # (batch, seq_len)

        return p_det, s_det


# =============== 纯检测损失函数 =================

class DetectionOnlyLoss(nn.Module):
    """
    纯检测损失函数（P+S 检测头）。

    组合三项：
    1. P 检测 BCE 损失（信号样本）
    2. S 检测 BCE 损失（有效S标签的信号样本）
    3. 噪声抑制损失（噪声样本上压制检测激活）

    参数
    ----
    p_weight : P 波检测损失权重（默认 1.0）
    s_weight : S 波检测损失权重（默认 0.6）
    noise_weight : 噪声抑制损失权重（默认 0.5）
    """

    def __init__(self, p_weight: float = 1.0, s_weight: float = 0.6,
                 noise_weight: float = 0.5, pos_weight: float = 3.0,
                 focal_gamma: float = 0.0, focal_on_noise: bool = False):
        super().__init__()
        self.p_weight = p_weight
        self.s_weight = s_weight
        self.noise_weight = noise_weight
        # ── 优先级1：原实现为纯 BCE，既无 pos_weight 也无 focal，导致难正样本
        #    （低信噪比事件）得不到额外梯度，分数长低尾（实测 EQS-D 上正样本
        #    P5=1.6e-6 ≈ 负样本中位数 1.66e-6）。 ──
        self.pos_weight = float(pos_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_on_noise = bool(focal_on_noise)

    def _bce(self, pred, target, focal: bool = True):
        """软化正样本加权 + 可选 focal 调制。

        pos_weight   : w = 1 + (pw-1)*target —— 正样本按标签强度加权
        focal_gamma  : 再乘 (1-p_t)^gamma —— 把梯度集中到难样本（0 表示不启用）
        噪声项（target 全 0）上 pos_weight 自动退化为 1，不影响。
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        if self.pos_weight != 1.0:
            bce = bce * (1.0 + (self.pos_weight - 1.0) * target)
        if focal and self.focal_gamma > 0.0:
            p = torch.sigmoid(pred)
            pt = p * target + (1.0 - p) * (1.0 - target)
            bce = bce * (1.0 - pt).clamp(min=1e-6) ** self.focal_gamma
        return bce.mean()

    def forward(self, p_det, s_det, p_rect, s_rect, s_valid, is_noise):
        """
        参数
        ----
        p_det   : (B, T) logits  — P 检测头
        s_det   : (B, T) logits  — S 检测头
        p_rect  : (B, T) P相位矩形标签
        s_rect  : (B, T) S相位矩形标签
        s_valid : (B,)   1=有效 S，0=无效
        is_noise: (B,)   1=噪声，0=信号

        返回
        ----
        total_loss, p_det_loss, s_det_loss
        """
        signal_mask = ~is_noise.bool()
        noise_mask  = is_noise.bool()

        # ── P 波检测损失（仅信号样本）────────────────────────────────
        if signal_mask.any():
            p_det_loss = self._bce(p_det[signal_mask], p_rect[signal_mask])
        else:
            p_det_loss = torch.tensor(0.0, device=p_det.device)

        # ── S 波检测损失（仅有效S标签的信号样本）──────────────────────
        s_signal_mask = signal_mask & s_valid.bool()
        if s_signal_mask.any():
            s_det_loss = self._bce(s_det[s_signal_mask], s_rect[s_signal_mask])
        else:
            s_det_loss = torch.tensor(0.0, device=p_det.device)

        # ── 噪声样本：检测头抑制（期望输出全为0）─────────────────────
        if noise_mask.any():
            noise_zeros = torch.zeros_like(p_det[noise_mask])
            p_noise_sup = self._bce(p_det[noise_mask], noise_zeros,
                                    focal=self.focal_on_noise)
            s_noise_sup = self._bce(s_det[noise_mask], noise_zeros,
                                    focal=self.focal_on_noise)
            noise_loss = 0.5 * (p_noise_sup + s_noise_sup)
        else:
            noise_loss = torch.tensor(0.0, device=p_det.device)

        total_loss = (self.p_weight * p_det_loss
                      + self.s_weight * s_det_loss
                      + self.noise_weight * noise_loss)
        return total_loss, p_det_loss, s_det_loss


# ================ 数据加载（三分量版，P+S双到时）=====================
def load_lightweight_data(file_name, csv_file, window_size=6000, max_samples=50000,
                          seed=42, trace_names=None):
    """
    三分量数据加载，同时读取 P 波和 S 波到时。
    读取每条波形的三个分量 (Z, N, E)，分别预处理后拼接为 (window_size, 3)。

    返回:
        sequences  : np.ndarray, shape (M, window_size, 3)
        p_targets  : np.ndarray, shape (M,)   P波相对位置 [0, 1]
        s_targets  : np.ndarray, shape (M,)   S波相对位置 [0, 1]，
                     无有效S到时的样本填 -1.0
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Original dataset size: {len(df)}")

    df_filtered = df.copy()

    if trace_names is not None:
        # 事件级划分：只加载指定 trace（先划分事件、再抽样）
        want = [str(t) for t in trace_names]
        df_filtered['trace_name'] = df_filtered['trace_name'].astype(str)
        df_filtered = (df_filtered.set_index('trace_name')
                       .reindex(want).dropna(how='all').reset_index())
        print(f"Event-split trace list: {len(df_filtered)} rows")
    elif len(df_filtered) > max_samples:
        df_filtered = df_filtered.sample(n=max_samples, random_state=seed)
        print(f"Sampled to {max_samples} samples (seed={seed})")

    ev_list = df_filtered['trace_name'].to_list()
    print(f"[{timestamp}] Processing {len(ev_list)} trace names...")

    sequences = []
    p_targets = []
    s_targets = []

    print(f"Loading HDF5 file: {file_name}")

    try:
        with h5py.File(file_name, 'r') as f:
            print("HDF5 file opened successfully")
            print(f"HDF5 keys: {list(f.keys())}")

            if 'data' in f.keys():
                print(f"Found 'data' group with {len(f['data'].keys())} entries")
                sample_keys = list(f['data'].keys())[:5]
                print(f"Sample keys: {sample_keys}")
            else:
                print("ERROR: No 'data' group found in HDF5 file!")
                return np.array([]), np.array([]), np.array([])

            valid_count = 0
            s_valid_count = 0
            error_count = 0

            for i, name in enumerate(tqdm(ev_list, desc="Loading data")):
                try:
                    dset = f.get('data/' + str(name))
                    if dset is None:
                        error_count += 1
                        if error_count < 5:
                            print(f"Warning: Dataset 'data/{name}' not found")
                        continue

                    data = np.array(dset)  # shape: (T, C), C >= 3

                    if data.ndim < 2:
                        print(f"Warning: Data shape {data.shape} for {name} is not 2D")
                        continue

                    if data.shape[1] < 3:
                        print(f"Warning: Data has only {data.shape[1]} channels for {name}")
                        continue

                    # 提取三个分量 (T, 3)，顺序 Z, N, E
                    data_3c = data[:, CHANNEL_INDICES]  # (T, 3)

                    # P波到时
                    p_arrival = dset.attrs.get('p_arrival_sample', -1)
                    if p_arrival == -1:
                        for attr_name in ['p_arrival', 'P_arrival_sample', 'p_pick']:
                            p_arrival = dset.attrs.get(attr_name, -1)
                            if p_arrival != -1:
                                break

                    if p_arrival <= 0 or p_arrival >= data_3c.shape[0]:
                        continue

                    # S波到时（可选：无效则记录 -1）
                    s_arrival = dset.attrs.get('s_arrival_sample', -1)
                    if s_arrival == -1:
                        for attr_name in ['s_arrival', 'S_arrival_sample', 's_pick']:
                            s_arrival = dset.attrs.get(attr_name, -1)
                            if s_arrival != -1:
                                break

                    if data_3c.shape[0] < window_size:
                        continue

                    # 三分量独立预处理
                    try:
                        data_3c = preprocess_3c(data_3c)   # (T, 3) float32
                    except Exception as e:
                        if error_count < 5:
                            print(f"Preprocessing error for {name}: {e}")
                        error_count += 1
                        continue

                    # 截取固定窗口
                    ws = 0
                    we = window_size
                    win = data_3c[ws:we, :]   # (window_size, 3)

                    if win.shape[0] < window_size:
                        continue

                    # 归一化相对位置标签
                    p_rel = float(p_arrival) / window_size
                    p_rel = np.clip(p_rel, 0.0, 1.0)

                    # S波：在窗口内且大于P波才有效
                    if (s_arrival > p_arrival) and (s_arrival < window_size):
                        s_rel = float(s_arrival) / window_size
                        s_rel = np.clip(s_rel, 0.0, 1.0)
                        s_valid_count += 1
                    else:
                        s_rel = -1.0   # 无效标记

                    # 有效性检查
                    if np.isnan(win).any() or np.isinf(win).any():
                        continue

                    sequences.append(win)
                    p_targets.append(p_rel)
                    s_targets.append(s_rel)
                    valid_count += 1

                    if valid_count % 10000 == 0:
                        print(f"Loaded {valid_count} valid samples...")

                except Exception as e:
                    error_count += 1
                    if error_count < 10:
                        print(f"Error processing {name}: {e}")
                    continue

            print(f"[{timestamp}] Successfully loaded {valid_count} samples "
                  f"(S-wave valid: {s_valid_count})")
            print(f"Total errors encountered: {error_count}")

    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        return np.array([]), np.array([]), np.array([])

    if len(sequences) == 0:
        print("ERROR: No valid sequences found!")
        return np.array([]), np.array([]), np.array([])

    sequences = np.array(sequences)   # (M, window_size, 3)
    p_targets = np.array(p_targets)   # (M,)
    s_targets = np.array(s_targets)   # (M,)
    print(f"Final dataset: {len(sequences)} sequences, shape: {sequences.shape}")
    print(f"P-target range: {p_targets.min():.3f} – {p_targets.max():.3f}")
    s_valid_mask = s_targets >= 0
    if s_valid_mask.any():
        print(f"S-target range: {s_targets[s_valid_mask].min():.3f} – "
              f"{s_targets[s_valid_mask].max():.3f}  "
              f"({s_valid_mask.sum()} valid / {len(s_targets)} total)")
    else:
        print("S-target: no valid S arrivals found in this file")

    return sequences, p_targets, s_targets


# ================== 噪声数据加载 ====================
def load_noise_data(file_name, csv_file, window_size=6000, max_samples=10000, seed=42):
    """
    从 STEAD chunk1（噪声分区）加载纯噪声波形。

    chunk1 的 CSV 中 trace_category == 'noise'，无 P/S 到时标注。
    返回的 p_targets 全部填 -2.0（噪声标记），s_targets 全部填 -2.0。

    返回:
        sequences : np.ndarray, shape (M, window_size, 3)
        p_targets : np.ndarray, shape (M,) — 全部为 -2.0
        s_targets : np.ndarray, shape (M,) — 全部为 -2.0
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"[Noise] Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"[Noise] Original dataset size: {len(df)}")

    if 'trace_category' in df.columns:
        df = df[df['trace_category'] == 'noise']
        print(f"[Noise] Filtered to noise category: {len(df)} samples")

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"[Noise] Sampled to {max_samples} samples")

    ev_list = df['trace_name'].to_list()
    print(f"[{timestamp}][Noise] Processing {len(ev_list)} trace names...")

    sequences = []
    valid_count = 0
    error_count = 0

    try:
        with h5py.File(file_name, 'r') as f:
            if 'data' not in f.keys():
                print("[Noise] ERROR: No 'data' group found in HDF5 file!")
                return np.array([]), np.array([]), np.array([])

            for name in tqdm(ev_list, desc="Loading noise"):
                try:
                    dset = f.get('data/' + str(name))
                    if dset is None:
                        error_count += 1
                        continue

                    data = np.array(dset)   # (T, C)
                    if data.ndim < 2 or data.shape[1] < 3:
                        continue
                    if data.shape[0] < window_size:
                        continue

                    data_3c = data[:, CHANNEL_INDICES]   # (T, 3)
                    try:
                        data_3c = preprocess_3c(data_3c)
                    except Exception:
                        error_count += 1
                        continue

                    win = data_3c[:window_size, :]       # (window_size, 3)
                    if win.shape[0] < window_size:
                        continue
                    if np.isnan(win).any() or np.isinf(win).any():
                        continue

                    sequences.append(win)
                    valid_count += 1

                    if valid_count % 5000 == 0:
                        print(f"[Noise] Loaded {valid_count} valid noise samples...")

                except Exception as e:
                    error_count += 1
                    if error_count < 5:
                        print(f"[Noise] Error processing {name}: {e}")
                    continue

    except Exception as e:
        print(f"[Noise] Error opening HDF5 file: {e}")
        return np.array([]), np.array([]), np.array([])

    if len(sequences) == 0:
        print("[Noise] WARNING: No valid noise sequences found!")
        return np.array([]), np.array([]), np.array([])

    sequences = np.array(sequences)                          # (M, T, 3)
    p_targets = np.full(len(sequences), -2.0, dtype=np.float32)   # 噪声标记
    s_targets = np.full(len(sequences), -2.0, dtype=np.float32)

    print(f"[Noise] Loaded {len(sequences)} noise sequences, shape: {sequences.shape}")
    return sequences, p_targets, s_targets


# ==================== 检测混淆矩阵评估 ====================
def compute_detection_metrics(p_det_logits, s_det_logits,
                              is_noise, s_valid,
                              threshold=0.5):
    """
    基于检测头输出（logits）计算每样本检测混淆矩阵及衍生指标。

    策略：
      - 对每样本的检测 logits 做 sigmoid → 取时间轴最大值
      - max >= threshold → 预测为正（检测到事件）
      - max <  threshold → 预测为负（未检测到事件）
      - Ground Truth：信号样本为正，噪声样本为负

    参数
    ----
    p_det_logits : (B, T)  P 波检测 logits
    s_det_logits : (B, T)  S 波检测 logits
    is_noise     : (B,)    1=噪声，0=信号
    s_valid      : (B,)    1=S 有效，0=S 无效
    threshold    : float   判定阈值（默认 0.5）

    返回
    ----
    p_metrics : dict — P 波混淆矩阵及指标
    s_metrics : dict — S 波混淆矩阵及指标
    """
    B = p_det_logits.shape[0]
    is_noise_np = is_noise.cpu().numpy().flatten() if torch.is_tensor(is_noise) else is_noise.flatten()
    s_valid_np  = s_valid.cpu().numpy().flatten()  if torch.is_tensor(s_valid)  else s_valid.flatten()

    def _per_sample_metrics(det_logits, gt_signal_mask, name):
        det_logits_np = det_logits.cpu().numpy() if torch.is_tensor(det_logits) else det_logits
        pred_prob = 1.0 / (1.0 + np.exp(-det_logits_np))   # sigmoid
        pred_max  = pred_prob.max(axis=1)                   # (B,) 每样本最大值

        pred_pos = (pred_max >= threshold).astype(int)
        gt_pos   = gt_signal_mask.astype(int)

        TP = int(np.sum((pred_pos == 1) & (gt_pos == 1)))
        TN = int(np.sum((pred_pos == 0) & (gt_pos == 0)))
        FP = int(np.sum((pred_pos == 1) & (gt_pos == 0)))
        FN = int(np.sum((pred_pos == 0) & (gt_pos == 1)))

        precision = TP / (TP + FP + 1e-10) * 100
        recall    = TP / (TP + FN + 1e-10) * 100
        f1        = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            'name'      : name,
            'threshold' : threshold,
            'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
            'n_total'   : B,
            'n_signal'  : int(gt_pos.sum()),
            'n_noise'   : B - int(gt_pos.sum()),
            'precision' : precision,
            'recall'    : recall,
            'f1'        : f1,
            'accuracy'  : (TP + TN) / (TP + TN + FP + FN + 1e-10) * 100,
        }

    # P 波：所有信号 vs 噪声
    gt_signal = (is_noise_np == 0).astype(int)
    p_metrics = _per_sample_metrics(p_det_logits, gt_signal, 'P')

    # S 波：仅对 S 有效信号 vs 噪声
    gt_s_signal = ((is_noise_np == 0) & (s_valid_np > 0.5)).astype(int)
    s_metrics = _per_sample_metrics(s_det_logits, gt_s_signal, 'S')

    return p_metrics, s_metrics


def print_detection_report(p_metrics, s_metrics):
    """打印检测指标报告（含混淆矩阵）。"""
    for met in [p_metrics, s_metrics]:
        cm = met['confusion_matrix']
        print(f"\n  [{met['name']}-wave Detection]  threshold={met['threshold']}")
        print(f"    Confusion Matrix:")
        print(f"                      Pred Pos    Pred Neg")
        print(f"      True Pos  (sig)  {cm['TP']:>8d}    {cm['FN']:>8d}")
        print(f"      True Neg  (noi)  {cm['FP']:>8d}    {cm['TN']:>8d}")
        print(f"    Total: {met['n_total']}  (signal={met['n_signal']}, noise={met['n_noise']})")
        print(f"    Precision: {met['precision']:.2f}%")
        print(f"    Recall   : {met['recall']:.2f}%")
        print(f"    F1       : {met['f1']:.2f}%")
        print(f"    Accuracy : {met['accuracy']:.2f}%")


def plot_confusion_matrix(p_metrics, s_metrics, save_path='confusion_matrix.png'):
    """绘制 P/S 检测混淆矩阵热力图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, met in zip(axes, [p_metrics, s_metrics]):
        cm = met['confusion_matrix']
        matrix = np.array([[cm['TP'], cm['FN']],
                           [cm['FP'], cm['TN']]], dtype=float)

        im = ax.imshow(matrix, cmap='Blues', vmin=0)

        # 标注数值和百分比
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                val = int(matrix[i, j])
                pct = val / total * 100 if total > 0 else 0
                color = 'white' if matrix[i, j] > matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pos (signal)', 'Neg (noise)'], fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['True Pos\n(signal)', 'True Neg\n(noise)'], fontsize=11)
        ax.set_title(f"{met['name']}-wave\n"
                     f"Prec={met['precision']:.1f}%  Recall={met['recall']:.1f}%  "
                     f"F1={met['f1']:.1f}%",
                     fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Detection Confusion Matrix — Test Set',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved: {save_path}")


# ================ 训练函数 =====================
def train_lightweight_model(model, train_loader, val_loader, device,
                            num_epochs=200, lr=1e-3,
                            p_weight=1.0, s_weight=0.6,
                            use_amp=False):
    """
    训练纯P+S波检测模型。

    参数
    ----
    p_weight : P 波检测损失权重（默认 1.0）
    s_weight : S 波检测损失权重（默认 0.6）
    use_amp  : 是否使用混合精度训练（默认 False，Mamba fp16 不稳定）
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.3)

    criterion = DetectionOnlyLoss(
        p_weight=p_weight,
        s_weight=s_weight,
        noise_weight=0.5,
        pos_weight=POS_WEIGHT,
        focal_gamma=FOCAL_GAMMA,
        focal_on_noise=FOCAL_ON_NOISE,
    )
    print(f"DetectionOnlyLoss — P_weight: {p_weight:.3f}, "
          f"S_weight: {s_weight:.3f}, pos_weight: {POS_WEIGHT:.2f}, "
          f"focal_gamma: {FOCAL_GAMMA:.2f}")

    # 混合精度（Mamba 在 fp16 下易溢出，默认关闭）
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("AMP mixed precision: ON (float16) — 注意：Mamba 可能不稳定")
    else:
        print("AMP: OFF (fp32)")

    best_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        nan_batches = 0

        for batch in train_loader:
            x, p_rect, s_rect, s_valid, is_noise = batch
            x        = x.to(device, non_blocking=True)
            p_rect   = p_rect.to(device, non_blocking=True)
            s_rect   = s_rect.to(device, non_blocking=True)
            s_valid  = s_valid.to(device, non_blocking=True)
            is_noise = is_noise.to(device, non_blocking=True)

            if torch.isnan(x).any() or torch.isnan(p_rect).any():
                nan_batches += 1
                continue

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                p_det, s_det = model(x)

            # loss 在 fp32 下计算，避免 BCEWithLogitsLoss 在 fp16 下数值不稳定
            total_loss, p_det_l, s_det_l = criterion(
                p_det.float(), s_det.float(),
                p_rect, s_rect,
                s_valid, is_noise
            )

            if not torch.isfinite(total_loss):
                nan_batches += 1
                continue

            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()

        if nan_batches > 0:
            print(f"  Warning: {nan_batches} NaN batches skipped this epoch")

        val_p_f1, val_s_f1 = evaluate_detection(
            model, val_loader, device)

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        # 用 P-F1 作为主验证指标
        val_metric = val_p_f1 if val_p_f1 > 0 else 0.0
        val_losses.append(val_metric)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'cfg': dict(band_low=BAND_LOW, band_high=BAND_HIGH,
                                    use_scaler=USE_SCALER,
                                    pos_weight=POS_WEIGHT,
                                    focal_gamma=FOCAL_GAMMA,
                                    window_size=int(CFG.get('window_size', 6000)),
                                    proj_dim=CFG.get('proj_dim', 96),
                                    hidden_dim=CFG.get('hidden_dim', 48),
                                    mamba_layers=CFG.get('mamba_layers', 3),
                                    d_state=CFG.get('d_state', 16),
                                    expand=CFG.get('expand', 2),
                                    mamba_mode=CFG.get('mamba_mode', 'bidirectional'),
                                    use_transformer=CFG.get('use_transformer', True),
                                    use_lfem=CFG.get('use_lfem', False),
                                    tag=CFG.get('tag', ''))}, 
                       f'best_detection_model_3C{CFG.get("tag", "")}.pth')

        epoch_time     = time.time() - epoch_start
        remaining_time = epoch_time * (num_epochs - epoch - 1)
        hrs, rem       = divmod(remaining_time, 3600)
        mins, secs     = divmod(rem, 60)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            s_info = f", S_F1={val_s_f1:.2f}%" if val_s_f1 >= 0 else ""
            print(f"Epoch {epoch:02d}: Loss={avg_train_loss:.6f}, "
                  f"P_F1={val_p_f1:.2f}%{s_info}, Best={best_loss:.6f}, "
                  f"ETA: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}")

    return train_losses, val_losses


def evaluate_detection(model, loader, device, threshold=0.5):
    """
    验证集检测评估：返回 P_F1, S_F1（百分比）。
    S_F1 = -1.0 表示验证集内无有效 S 标签。
    """
    model.eval()
    all_p_det, all_s_det = [], []
    all_is_noise, all_s_valid = [], []

    with torch.no_grad():
        for batch in loader:
            x, p_rect, s_rect, s_valid, is_noise = batch
            x        = x.to(device)
            s_valid  = s_valid.to(device)
            is_noise = is_noise.to(device)

            p_det, s_det = model(x)
            all_p_det.append(p_det.cpu())
            all_s_det.append(s_det.cpu())
            all_is_noise.append(is_noise.cpu())
            all_s_valid.append(s_valid.cpu())

    p_det_all    = torch.cat(all_p_det, dim=0)
    s_det_all    = torch.cat(all_s_det, dim=0)
    is_noise_all = torch.cat(all_is_noise, dim=0)
    s_valid_all  = torch.cat(all_s_valid, dim=0)

    p_metrics, s_metrics = compute_detection_metrics(
        p_det_all, s_det_all, is_noise_all, s_valid_all,
        threshold=threshold)

    p_f1 = p_metrics['f1']
    s_f1 = s_metrics['f1'] if s_metrics['n_signal'] > 0 else -1.0
    return p_f1, s_f1


# ==================== 可视化函数（检测版） ====================
def plot_detection_example(model, dataset, device, idx=0, save_path='detection_example.png',
                           window_size=6000):
    """绘制检测样例：输入波形 + 预测概率 vs 真实标签。"""
    x, p_rect, s_rect, s_valid, is_noise = dataset[idx]
    x_tensor = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        p_det, s_det = model(x_tensor)

    p_prob = torch.sigmoid(p_det).cpu().numpy().flatten()
    s_prob = torch.sigmoid(s_det).cpu().numpy().flatten()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # 三个分量波形
    x_np = x.numpy()
    for c, (ax, label, color) in enumerate(zip(
        axes[:3], ['Z', 'N', 'E'], ['tab:blue', 'tab:orange', 'tab:green'])):
        ax.plot(x_np[:, c], color=color, lw=0.5)
        ax.set_ylabel(f'{label}')
        ax.grid(True, alpha=0.3)

    # 检测概率
    ax = axes[3]
    ax.plot(p_prob, 'b-', lw=1.5, label='P detection prob')
    ax.plot(s_prob, 'r-', lw=1.5, label='S detection prob')
    ax.plot(p_rect.numpy(), 'b--', lw=1, alpha=0.5, label='P label')
    ax.plot(s_rect.numpy(), 'r--', lw=1, alpha=0.5, label='S label')
    ax.axhline(0.5, color='gray', lw=0.5, linestyle=':')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Detection Prob')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    noise_str = 'NOISE' if is_noise.item() > 0.5 else 'SIGNAL'
    s_str = f', S_valid={s_valid.item():.0f}' if is_noise.item() < 0.5 else ''
    fig.suptitle(f'Detection Example (idx={idx}, {noise_str}{s_str})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detection example saved: {save_path}")


# ==================== 主函数 ====================
def main(seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加速配置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark = True")

    print("=" * 50)
    print("纯地震检测模型 — 三分量版本（双向Mamba）")
    print(f"输入分量: Z(col{CH_Z}), N(col{CH_N}), E(col{CH_E})")
    print(f"Random seed: {seed}")
    print(f"输入窗口: 6000 点 (60s @ 100Hz)")
    print("=" * 50)

    window_size = int(CFG.get('window_size', 6000))
    print(f'[窗长] window_size = {window_size} 采样点 ({window_size/100:.1f}s @100Hz)')

    # ===================== 事件级划分（回应审稿人） =====================
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
        sq_l, p_l, s_l = [], [], []
        for c in CFG['train_chunks']:
            names = names_for_split(meta, split, c, max_samples, seed)
            if not names:
                continue
            hf, cf = _paths(c)
            print(f"\n[{split}] {c}: 指定 {len(names)} 条")
            sq, pt, st = load_lightweight_data(
                hf, cf, window_size=window_size, max_samples=max_samples,
                seed=seed, trace_names=names)
            if len(sq) > 0:
                sq_l.append(sq); p_l.append(pt); s_l.append(st)
        if not sq_l:
            return (np.zeros((0, window_size, 3), np.float32),
                    np.zeros(0, np.float32), np.zeros(0, np.float32))
        return (np.concatenate(sq_l, axis=0), np.concatenate(p_l, axis=0),
                np.concatenate(s_l, axis=0))

    X_train, p_train, s_train = _load_split('train', CFG['max_samples'])
    X_val,   p_val,   s_val   = _load_split('val',   CFG['val_samples'])
    X_test,  p_test,  s_test  = _load_split('test',  CFG['test_samples'])

    # 噪声（chunk1）：无事件归属，按 trace 并入三个集合（原实现也是并入后再划分）
    if CFG['noise_chunks']:
        print("\nLoading noise data ...")
        for c in CFG['noise_chunks']:
            hf, cf = _paths(c)
            if not (os.path.exists(hf) and os.path.exists(cf)):
                print(f"[skip] 不存在: {hf}")
                continue
            nseq, npp, nss = load_noise_data(
                hf, cf, window_size=window_size,
                max_samples=CFG['noise_samples'], seed=seed)
            if len(nseq) == 0:
                continue
            idx = np.random.default_rng(seed).permutation(len(nseq))
            n_tr = int(round(len(idx) * CFG['ratios'][0]))
            n_va = int(round(len(idx) * CFG['ratios'][1]))
            parts = {'train': idx[:n_tr], 'val': idx[n_tr:n_tr + n_va],
                     'test': idx[n_tr + n_va:]}
            print(f"  {c}: {len(nseq)} 条噪声 -> train/val/test = "
                  f"{len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}")
            X_train = np.concatenate([X_train, nseq[parts['train']]], 0)
            p_train = np.concatenate([p_train, npp[parts['train']]], 0)
            s_train = np.concatenate([s_train, nss[parts['train']]], 0)
            X_val = np.concatenate([X_val, nseq[parts['val']]], 0)
            p_val = np.concatenate([p_val, npp[parts['val']]], 0)
            s_val = np.concatenate([s_val, nss[parts['val']]], 0)
            X_test = np.concatenate([X_test, nseq[parts['test']]], 0)
            p_test = np.concatenate([p_test, npp[parts['test']]], 0)
            s_test = np.concatenate([s_test, nss[parts['test']]], 0)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 波形库（第二事件增强）：排除 val/test 事件，避免测试事件经增强泄漏
    wave_pool = None
    if CFG['pool_chunk']:
        pm = load_chunks(_data_dir(), [CFG['pool_chunk']])
        pn = pool_names(pm, exclude, CFG['pool_samples'], seed)
        hf, cf = _paths(CFG['pool_chunk'])
        print(f"\nLoading waveform pool from: {hf} & {cf}"
              f"（{len(pn)} 条，已排除 val/test 事件）")
        ps, _, _ = load_lightweight_data(
            hf, cf, window_size=window_size, max_samples=CFG['pool_samples'],
            seed=seed, trace_names=pn)
        if len(ps) > 0:
            wave_pool = np.concatenate([ps], axis=0)
            wave_pool = wave_pool.astype(np.float32, copy=False)
            print(f"Waveform pool shape: {wave_pool.shape}")

    # 兼容原代码后续对 waves.shape 的使用
    waves = X_train if len(X_train) else np.zeros((0, window_size, 3), np.float32)
    print(f"\nTraining data shape: {waves.shape}  (raw 3C waveforms)")

    if CFG.get('dry_run'):
        print('\n[dry-run] 数据装配完成，跳过训练。'
              '正式训练请用 --split-table 复用同一张划分表。')
        return

    # 初始化增强器
    augmentation = SimpleAugmentation(
        per_ch=1,
        second_event_pool=wave_pool
    )

    # 相位标签参数
    phase_scale = 1.4       # P相位长度 = 1.4 × (S-P)
    s_phase_scale = 0.5     # S相位长度 = 0.5 × (S-P)
    print(f"Phase label: P-phase = {phase_scale}×(S-P), S-phase = {s_phase_scale}×(S-P)")

    train_ds = LightweightWaveDataset(
        X_train, p_train, s_targets=s_train,
        is_train=True, transforms=augmentation,
        window_size=window_size,
        phase_scale=phase_scale, s_phase_scale=s_phase_scale)
    val_ds = LightweightWaveDataset(
        X_val, p_val, s_targets=s_val,
        is_train=False, scaler=train_ds.scaler,
        window_size=window_size,
        phase_scale=phase_scale, s_phase_scale=s_phase_scale)
    test_ds = LightweightWaveDataset(
        X_test, p_test, s_targets=s_test,
        is_train=False, scaler=train_ds.scaler,
        window_size=window_size,
        phase_scale=phase_scale, s_phase_scale=s_phase_scale)

    with open('scaler_3C.pkl', 'wb') as f:
        pickle.dump(train_ds.scaler, f)
    print("Scaler saved to 'scaler_3C.pkl'")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 模型初始化
    input_dim = waves.shape[-1]
    print(f"\nModel input_dim = {input_dim}  (raw 3-component waveforms)")

    use_transformer = CFG.get('use_transformer', True)
    print(f"use_transformer = {use_transformer}")

    mamba_mode = CFG.get('mamba_mode', 'bidirectional')
    print(f"mamba_mode = {mamba_mode}")

    model = LightweightPWaveNet(
        input_dim=input_dim,
        hidden_dim=CFG.get('hidden_dim', 48),
        use_transformer=use_transformer,
        mamba_mode=mamba_mode,
        proj_dim=CFG.get('proj_dim', 96),
        mamba_layers=CFG.get('mamba_layers', 3),
        d_state=CFG.get('d_state', 16),
        expand=CFG.get('expand', 2),
        use_lfem=CFG.get('use_lfem', False),
    ).to(device)
    print(f"[模块] mamba_mode={mamba_mode} use_transformer={use_transformer} "
          f"use_lfem={CFG.get('use_lfem', False)}")
    print(f"[宽度] proj_dim={CFG.get('proj_dim',96)} hidden_dim={CFG.get('hidden_dim',48)} "
          f"mamba_layers={CFG.get('mamba_layers',3)} d_state={CFG.get('d_state',16)}")
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # 损失权重
    p_weight = 1.0
    s_weight = 0.6
    print(f"Loss weights — P: {p_weight:.3f}, S: {s_weight:.3f}")

    # 训练
    print("\nTraining (pure detection)...")
    _ep = CFG.get('epochs', 100)
    print(f"[主路径] 训练轮数 = {_ep}")
    train_losses, val_metrics = train_lightweight_model(
        model, train_loader, val_loader, device,
        num_epochs=_ep, lr=1e-3,
        p_weight=p_weight, s_weight=s_weight
    )

    # ─────────────── 训练后评估 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("=" * 60)
    _best = f'best_detection_model_3C{CFG.get("tag", "")}.pth'
    try:
        _ck = torch.load(_best, map_location='cpu', weights_only=False)
        _sd = (_ck['model_state_dict']
               if isinstance(_ck, dict) and 'model_state_dict' in _ck else _ck)
        model.load_state_dict(_sd)
        print(f"[训练后评估] 已加载本次权重: {_best}")
    except Exception as _e:
        print(f"[训练后评估] 加载 {_best} 失败（{_e}），跳过该评估")
        return model
    model.eval()

    all_p_det, all_s_det = [], []
    all_is_noise, all_s_valid = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, p_rect, s_rect, s_valid, is_noise = batch
            x        = x.to(device)
            s_valid  = s_valid.to(device)
            is_noise = is_noise.to(device)

            p_det, s_det = model(x)

            all_p_det.append(p_det.cpu())
            all_s_det.append(s_det.cpu())
            all_is_noise.append(is_noise.cpu())
            all_s_valid.append(s_valid.cpu())

    p_det_all    = torch.cat(all_p_det, dim=0)
    s_det_all    = torch.cat(all_s_det, dim=0)
    is_noise_all = torch.cat(all_is_noise, dim=0)
    s_valid_all  = torch.cat(all_s_valid, dim=0)

    # 检测评估
    print(f"\n{'='*60}")
    print("DETECTION EVALUATION (Confusion Matrix)")
    print(f"{'='*60}")

    p_det_metrics, s_det_metrics = compute_detection_metrics(
        p_det_all, s_det_all, is_noise_all, s_valid_all,
        threshold=0.5)
    print_detection_report(p_det_metrics, s_det_metrics)

    # 绘制混淆矩阵
    plot_confusion_matrix(p_det_metrics, s_det_metrics,
                          save_path='confusion_matrix.png')

    # ── 不同阈值下的 F1 扫描 ──
    print(f"\n  [Threshold Scan — P-wave Detection F1]")
    print(f"  {'thresh':>8s}  {'F1(%)':>8s}  {'Prec(%)':>8s}  {'Recall(%)':>8s}")
    best_p_f1, best_p_th = 0.0, 0.5
    for th in np.arange(0.1, 0.91, 0.1):
        pm, _ = compute_detection_metrics(p_det_all, s_det_all,
                                          is_noise_all, s_valid_all,
                                          threshold=th)
        print(f"  {th:>8.1f}  {pm['f1']:>8.2f}  {pm['precision']:>8.2f}  {pm['recall']:>8.2f}")
        if pm['f1'] > best_p_f1:
            best_p_f1, best_p_th = pm['f1'], th
    print(f"  Best P F1={best_p_f1:.2f}% @ threshold={best_p_th:.1f}")

    # S 波阈值扫描
    if s_det_metrics['n_signal'] > 0:
        print(f"\n  [Threshold Scan — S-wave Detection F1]")
        print(f"  {'thresh':>8s}  {'F1(%)':>8s}  {'Prec(%)':>8s}  {'Recall(%)':>8s}")
        best_s_f1, best_s_th = 0.0, 0.5
        for th in np.arange(0.1, 0.91, 0.1):
            _, sm = compute_detection_metrics(p_det_all, s_det_all,
                                              is_noise_all, s_valid_all,
                                              threshold=th)
            print(f"  {th:>8.1f}  {sm['f1']:>8.2f}  {sm['precision']:>8.2f}  {sm['recall']:>8.2f}")
            if sm['f1'] > best_s_f1:
                best_s_f1, best_s_th = sm['f1'], th
        print(f"  Best S F1={best_s_f1:.2f}% @ threshold={best_s_th:.1f}")

    # ── 检测样例可视化 ──
    # 取前2个信号样本和1个噪声样本做可视化
    for vis_idx in range(min(3, len(test_ds))):
        plot_detection_example(model, test_ds, device, idx=vis_idx,
                               save_path=f'detection_example_{vis_idx}.png',
                               window_size=window_size)

    # ── 训练过程可视化 ──
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss (Detection)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_metrics, label='Val P-F1')
    plt.title('Validation P-F1 Score')
    plt.xlabel('Epoch'); plt.ylabel('F1 (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('detection_results_3C.png', dpi=300)
    plt.show()

    print("\nModel saved as 'best_detection_model_3C.pth'")
    print("Detection example plots saved as 'detection_example_*.png'")
    print("Training curve saved as 'detection_results_3C.png'")


if __name__ == '__main__':
    main(seed=apply_cli())

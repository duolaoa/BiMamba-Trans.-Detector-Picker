# 轻量化P波检测模型 - 三分量版本（双向Mamba版）
# 主要改动：
# 1. 数据加载：同时读取三个地震分量 (Z, N, E)
# 2. 预处理：对每个分量独立预处理
# 3. 直接输入原始三分量波形 [N, T, 3]，不需要额外特征提取
# 4. 模型输入维度：input_dim = 3（原始三分量波形）
# 5. 序列建模：用双向Mamba（BidirectionalMamba）替换双向GRU（ResidualGRU）
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
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
import warnings
import time
import os
import glob
# from regress import AddGaussianNoise, AmpScale, TimeJitter, PolarityFlip, sta_lta, teager_energy, envelope
from scipy.stats import norm, gaussian_kde
from scipy import stats
import math

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
PRESETS = {
    'pn197k':      dict(proj_dim=16,  hidden_dim=40),
    'pm322k':      dict(proj_dim=32,  hidden_dim=48),
    'eqt466k':     dict(proj_dim=56,  hidden_dim=48),
    'ctrl813k':    dict(proj_dim=96,  hidden_dim=48),
    'abl_nomamba': dict(proj_dim=188, hidden_dim=40, mamba_mode='none'),
    'abl_notrans': dict(proj_dim=54,  hidden_dim=8,  use_transformer=False),
    'abl_nolfem':  dict(proj_dim=16,  hidden_dim=40, use_lfem=False),
}

CFG = {
    'data_dir': _1DSIGNAL,
    # 训练机布局：chunk2/chunk4 在 autodl-tmp/ 下，chunk3 在项目根下 —— 所以
    # data_dir（或 STEAD_DIR 环境变量）必须指向**项目目录**，这两条相对路径才成立。
    # 换机器若布局不同，用 --train-chunks/--pool-chunk 或环境变量覆盖即可。
    'train_chunks': ['chunk2', 'chunk3'],
    'pool_chunk': 'chunk4',
    # BiMamba 拾取器**不需要**噪声：它是两段式流程里的第二段，
    # 只在检测器判定为信号的窗口内工作；噪声抑制由检测器
    #（detection_dual_head_train.py，训练时用了 chunk1）负责。
    'noise_chunks': [],
    # 位置增广：把 P 随机摆到窗内 [off_min, off_max]，打破"P 恒在 ~500"的位置先验
    # （端到端推理时检测器锚点误差 139-378 点，若不增广，拾取器会原样复述锚点）
    'offset_reps': 1,
    'off_min': 100,
    'off_max': 2900,
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
    ap.add_argument('--window-size', type=int, default=3000,
                    help='拾取窗长（采样点）；默认 3000，可设 6000 做窗口消融')
    ap.add_argument('--offset-reps', type=int, default=CFG['offset_reps'],
                    help='位置增广：每条 trace 生成几份随机偏移样本；'
                         '0=关闭（老行为），1=每条替换成 1 份（内存不增）')
    ap.add_argument('--offset-min', type=int, default=CFG['off_min'])
    # ── 宽度/容量（用于"参数量匹配"消融；默认=原实现，保证老权重兼容）──
    ap.add_argument('--proj-dim', type=int, default=96,
                    help='Mamba/Transformer 特征宽度（原实现 96）')
    ap.add_argument('--preset', default=None,
                    help='参数量/模块命名预设；与显式宽度参数同用时以预设为准')
    ap.add_argument('--hidden-dim', type=int, default=48,
                    help='hidden_dim（控制 transformer/fusion/head 宽度，原实现 48）')
    ap.add_argument('--mamba-layers', type=int, default=3)
    ap.add_argument('--d-state', type=int, default=16)
    ap.add_argument('--expand', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=100,
                    help='训练轮数（主路径原实现硬编码 100）')
    ap.add_argument('--tag', default='',
                    help='权重文件名后缀，如 _pm322k -> best_lightweight_model_3C_pm322k.pth')
    ap.add_argument('--offset-max', type=int, default=CFG['off_max'])
    ap.add_argument('--max-samples', type=int, default=CFG['max_samples'])
    ap.add_argument('--val-samples', type=int, default=CFG['val_samples'])
    ap.add_argument('--test-samples', type=int, default=CFG['test_samples'])
    ap.add_argument('--pool-samples', type=int, default=CFG['pool_samples'])
    ap.add_argument('--noise-samples', type=int, default=CFG['noise_samples'])
    ap.add_argument('--split-table', default=CFG['split_table'],
                    help='复用已有划分表（多模型共用同一划分）')
    ap.add_argument('--save-split-table', default=CFG['save_split_table'])
    ap.add_argument('--dry-run', action='store_true',
                    help='只装配数据并打印划分/形状，不训练')
    ap.add_argument('--seed', type=int, default=42)
    # ── 模块消融开关 ──
    ap.add_argument('--mamba-mode', choices=['none', 'unidirectional', 'bidirectional'],
                    default='bidirectional', help='none=去掉 BiMamba')
    ap.add_argument('--no-transformer', dest='use_transformer', action='store_false',
                    help='去掉 Transformer 编码器')
    ap.add_argument('--no-lfem', dest='use_lfem', action='store_false',
                    help='LFEM1D 换成普通 Conv1d')
    ap.add_argument('--cross-channel-attention', dest='use_cross_channel_attention',
                    action='store_true', help='打开跨通道注意力（默认关）')
    ap.set_defaults(use_transformer=True, use_lfem=True,
                    use_cross_channel_attention=False)
    a = ap.parse_args(argv)
    if getattr(a, 'preset', None):
        _p = PRESETS[a.preset]
        for _k, _v in _p.items():
            setattr(a, _k, _v)
        print(f'[preset] {a.preset} -> ' +
              ' '.join(f'{k}={v}' for k, v in _p.items()))
    CFG.update(data_dir=a.data_dir, train_chunks=a.train_chunks,
               pool_chunk=a.pool_chunk, noise_chunks=a.noise_chunks,
               ratios=tuple(float(x) for x in a.ratios.split(',')),
               max_samples=a.max_samples, val_samples=a.val_samples,
               test_samples=a.test_samples, pool_samples=a.pool_samples,
               noise_samples=a.noise_samples, split_table=a.split_table,
               save_split_table=a.save_split_table,
               total_samples=a.total_samples,
               epochs=a.epochs,
               mamba_mode=a.mamba_mode, use_transformer=a.use_transformer,
               use_lfem=a.use_lfem,
               use_cross_channel_attention=a.use_cross_channel_attention,
               proj_dim=a.proj_dim, hidden_dim=a.hidden_dim,
               mamba_layers=a.mamba_layers, d_state=a.d_state, expand=a.expand,
               tag=a.tag,
               window_size=a.window_size,
               offset_reps=a.offset_reps, off_min=a.offset_min,
               off_max=a.offset_max,
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

# 三个分量的列索引（STEAD数据集约定）
# STEAD: columns 0=E, 1=N, 2=Z （可根据实际数据集调整）
CH_Z = 2   # 垂直分量
CH_N = 1   # 南北分量
CH_E = 0   # 东西分量
CHANNEL_INDICES = [CH_Z, CH_N, CH_E]   # 提取顺序 Z, N, E
N_CHANNELS = len(CHANNEL_INDICES)       # = 3


# =================== 简化的数据预处理 ===================
def simple_preprocessing(data, fs=100):
    """简化预处理：基本滤波和归一化（适用于单通道一维数组）"""
    # 去趋势
    data = detrend(data, type='linear')

    # 简单带通滤波 (2-30Hz)
    nyq = fs / 2
    # 统一预处理：与 PhaseNet/EQT 一致的 1-40 Hz 带通
    low, high = 1.0 / nyq, min(40.0 / nyq, 0.99)
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




# =================== 波形直通（无需特征提取）===================
def simple_geophys_features(waves, fs=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    直接返回原始三分量波形（无需特征提取）。

    输入:
        waves: numpy数组，shape [N, T, 3]

    输出:
        features: numpy数组，shape [N, T, 3]，原始波形
    """
    if len(waves) == 0:
        raise ValueError("No input waves provided!")

    if isinstance(waves, list):
        waves = np.array(waves)

    if waves.ndim == 2:
        waves = waves[:, :, np.newaxis]

    N = waves.shape[0]
    print(f"Using raw waveforms: {waves.shape}")
    return waves.astype(np.float32)


# =================== 高斯软标签生成 ===================
def generate_gaussian_label(window_size, center, sigma=30):
    """
    生成高斯软标签（归一化为概率分布，和为1）。
    输入:
        window_size : 窗口大小（与模型实际输出序列等长）
        center      : P波中心位置，相对比例 [0, 1]
        sigma       : 高斯标准差（采样点数，默认30）
    输出:
        label : shape (window_size,)，归一化到和为1的概率分布
    """
    center_idx = center * window_size          # 保持浮点，避免量化误差
    x = np.arange(window_size, dtype=np.float32)
    gaussian = np.exp(-0.5 * ((x - center_idx) / sigma) ** 2)
    s = gaussian.sum()
    if s > 0:
        gaussian /= s                          # 归一化为概率分布（和为1）
    return gaussian.astype(np.float32)

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
        # 复现 EQTransformer 策略：30% 的样本在同一窗口内叠加
        # 另一个地震信号，幅度随机缩放后直接相加。
        # 标签（主 P 到时）不变，训练模型忽略干扰事件、专注主事件。
        if rng.random() < self.second_event_prob:
            if self.second_event_pool is not None:
                # 从真实波形库中随机取一条（与当前样本不同）
                idx2 = int(rng.integers(0, len(self.second_event_pool)))
                x2 = self.second_event_pool[idx2].copy()   # (T, F)
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
                # 在原始波形列上生成，派生特征列保持零（叠加后会被重提取）
                x2 = np.zeros_like(x)
                for col in self.raw_cols:
                    burst = rng.standard_normal(T).astype(np.float32)
                    # 简单平滑模拟地震包络
                    onset = int(rng.integers(0, max(1, T - T // 4)))
                    decay = np.exp(-np.arange(T - onset) / (T * 0.1)).astype(np.float32)
                    burst[onset:] *= decay
                    burst[:onset]  = 0.0
                    x2[:, col] = burst

            # 目标相对幅度：对数均匀采样，覆盖微弱叠加到同等强度
            main_rms = np.sqrt(np.mean(x[:, self.raw_cols] ** 2)) + 1e-8
            sec_rms  = np.sqrt(np.mean(x2[:, self.raw_cols] ** 2)) + 1e-8
            amp = np.exp(rng.uniform(
                np.log(self.second_event_amp_range[0]),
                np.log(self.second_event_amp_range[1])
            ))
            # 将第二事件缩放到目标相对幅度后叠加
            x2_scaled = x2 * (main_rms / sec_rms * amp)
            x += x2_scaled.astype(np.float32)

        return x.astype(np.float32)


# =============== 数据集 =================
class LightweightWaveDataset(Dataset):
    def __init__(self, waves, p_targets, s_targets=None,
                 is_train=True, scaler=None, transforms=None,
                 use_gaussian=False, window_size=6000, sigma=30):
        """
        waves      : (N, T, F)
        p_targets  : (N,)  P波相对位置
        s_targets  : (N,)  S波相对位置（-1 表示无效），传 None 则全部置 -1
        """
        # waves shape: (N, T, num_features)
        if waves.ndim == 2:
            waves = waves[..., None]

        self.waves = waves.astype(np.float32)
        self.p_targets = p_targets.astype(np.float32)
        self.s_targets = (s_targets.astype(np.float32)
                          if s_targets is not None
                          else np.full(len(p_targets), -1.0, dtype=np.float32))
        self.is_train = is_train
        self.transforms = transforms
        self.use_gaussian = use_gaussian
        self.window_size = window_size   # 与实际波形时间轴长度一致
        self.sigma = sigma

        # 统一预处理：simple_preprocessing 里已做"逐通道除以 std"，
        # 这里不再叠加 StandardScaler（原实现等于做了两遍归一化，与
        # PhaseNet/EQT 不一致）。保留一个 with_mean/std=False 的 no-op scaler，
        # 使 train_ds.scaler 的接口与 scaler_3C.pkl 的保存/加载不变。
        if scaler is None:
            self.scaler = StandardScaler(with_mean=False, with_std=False)
            self.scaler.fit(self.waves.reshape(-1, self.waves.shape[-1]))
        else:
            self.scaler = scaler

        self.waves = self.scaler.transform(
            self.waves.reshape(-1, self.waves.shape[-1])
        ).reshape(self.waves.shape).astype(np.float32)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        x = self.waves[idx].copy()
        if self.is_train and self.transforms:
            x = self.transforms(x)

        p_pos = self.p_targets[idx]
        s_pos = self.s_targets[idx]   # -1.0 表示无效

        if self.use_gaussian:
            # P 波高斯软标签（始终有效）
            p_label = generate_gaussian_label(self.window_size, p_pos, self.sigma)

            # S 波高斯软标签（无效时全零）
            if s_pos >= 0.0:
                s_label = generate_gaussian_label(self.window_size, s_pos, self.sigma)
                s_valid = 1.0
            else:
                s_label = np.zeros(self.window_size, dtype=np.float32)
                s_valid = 0.0

            return (torch.from_numpy(x).float(),
                    torch.tensor(p_label, dtype=torch.float32),
                    torch.tensor(s_label, dtype=torch.float32),
                    torch.tensor(s_valid, dtype=torch.float32))
        else:
            s_valid = 1.0 if s_pos >= 0.0 else 0.0
            return (torch.from_numpy(x).float(),
                    torch.tensor([p_pos], dtype=torch.float32),
                    torch.tensor([max(s_pos, 0.0)], dtype=torch.float32),
                    torch.tensor(s_valid, dtype=torch.float32))


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
        # return fwd_out


class BidirectionalMamba(nn.Module):
    """
    多层双向 Mamba 序列编码器，接口与原 ResidualGRU 兼容：

    输入 : (B, T, d_model)         ← 此处 d_model = 96（proj 层输出）
    输出 : (B, T, d_model * 2)     ← 末层将前/后向特征拼接，维度翻倍

    参数
    ----
    d_model    : 输入特征维度
    d_state    : Mamba SSM 状态维度（默认 16）
    d_conv     : 局部卷积核大小（默认 4）
    expand     : 内部扩展比（默认 2）
    num_layers : 堆叠层数（默认 3，对应原 ResidualGRU 的 num_layers=3）
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

        # 末层将前/后向分支显式拼接，对齐原 ResidualGRU(bidirectional=True) 的
        # 输出维度 hidden_dim * 2。这里用一个分离的前/后向 Mamba 做最终拼接。
        self.final_fwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.final_bwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中间层：双向融合保持维度
        out = x
        for layer in self.layers:
            out = self.dropout(layer(out))          # (B, T, d_model)

        # 最终层：前/后向分别输出再拼接 → (B, T, d_model * 2)
        fwd_final = self.final_fwd(out)             # (B, T, d_model)
        bwd_final = self.final_bwd(out.flip(1)).flip(1)  # (B, T, d_model)
        return torch.cat([fwd_final, bwd_final], dim=-1)  # (B, T, d_model*2)

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        # 删除: self.pos_encoder = nn.Parameter(torch.randn(1, 10000, hidden_size))
        # 改为正弦位置编码，0个可学习参数
        self.register_buffer('pos_encoder', 
                             self._build_sinusoidal(10000, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, 
            dim_feedforward=hidden_size * 4,   # 可进一步改为 hidden_size * 2 再省一半
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
        return pe  # register_buffer，不计入参数

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]  # 直接切片，无需插值
        x = self.dropout(x)
        return self.transformer_encoder(x)

class CrossChannelAttention(nn.Module):
    """
    轻量跨分量注意力：将三分量的特征拼接后做 1×1 卷积重加权。
    输入: x (B, T, 15)  —— 3分量×5特征
    输出: x (B, T, 15)  —— 重加权后，保持维度不变
    """
    def __init__(self, n_feat=3, n_ch=N_CHANNELS):
        super().__init__()
        per_ch = n_feat // n_ch   # = 1 
        # 对每个分量学一个标量门控权重
        self.gate = nn.Sequential(
            nn.Linear(n_feat, n_ch),
            nn.Sigmoid()
        )
        self.per_ch = per_ch
        self.n_ch   = n_ch

    def forward(self, x):
        # x: (B, T, F)
        g = self.gate(x)                          # (B, T, 3)
        # 将 (B,T,F) 拆为 (B,T,3,F//3)，然后按 gate 重加权
        B, T, F = x.shape
        xr = x.view(B, T, self.n_ch, self.per_ch)  # (B,T,3,F//3)
        gw = g.unsqueeze(-1)                        # (B,T,3,1)
        out = (xr * gw).view(B, T, F)               # (B,T,F)
        return out + x                               # 残差

class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # 深度卷积
        self.dwconv = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            padding=padding, groups=in_channels  # groups=dim → 深度卷积
        )
        # 逐点卷积
        self.pwconv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x

# ==============================
# 1D 空洞卷积（扩大感受野）
# ==============================
class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, padding=2):
        super().__init__()
        self.dconv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=padding, dilation=dilation
        )

    def forward(self, x):
        return self.dconv(x)

# ==============================
# 通道重排（Channel Shuffle）
# ==============================
class ChannelShuffle(nn.Module):
    """
    通道重排模块，用于提高特征提取效率。
    将通道维度重新排列，增强通道间的信息交互。
    """
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, time_steps = x.size()
        channels_per_group = num_channels // self.groups
        
        # reshape: (B, C, T) -> (B, groups, C//groups, T)
        x = x.view(batch_size, self.groups, channels_per_group, time_steps)
        
        # transpose: (B, groups, C//groups, T) -> (B, C//groups, groups, T)
        x = x.transpose(1, 2).contiguous()
        
        # reshape: (B, C//groups, groups, T) -> (B, C, T)
        x = x.view(batch_size, num_channels, time_steps)
        
        return x

# ==============================
#  1D 时序专用 LFEM（局部特征增强模块）
# ==============================
class LFEM1D(nn.Module):
    def __init__(self, dim, expansion=0.5, kernel_size=3):
        super().__init__()
        hidden_dim = int(dim * expansion)

        # 1x1 升维
        self.conv1x1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()

        # 多分支提取局部时序特征
        branch_out_dim = hidden_dim // 3
        # 确保三个分支的输出通道数之和等于hidden_dim
        branch0_out = branch_out_dim
        branch1_out = branch_out_dim
        branch2_out = hidden_dim - branch0_out - branch1_out
        self.branch0 = nn.Conv1d(hidden_dim, branch0_out, kernel_size, padding=kernel_size//2)
        self.branch1 = DilatedConv1d(hidden_dim, branch1_out, kernel_size=3, dilation=2, padding=2)
        self.branch2 = DepthWiseConv1d(hidden_dim, branch2_out, kernel_size=3, padding=1)

        # 通道混洗 - 只有当hidden_dim能被3整除时才使用
        self.use_shuffle = (hidden_dim % 3 == 0)
        if self.use_shuffle:
            self.shuffle = nn.ChannelShuffle(groups=3)

        # 1x1 降维 + 残差
        self.conv_out = nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x  # 残差连接
        out = self.act(self.bn(self.conv1x1(x)))

        # 三分支并行提取
        b0 = self.branch0(out)
        b1 = self.branch1(out)
        b2 = self.branch2(out)

        # 拼接 + 通道交互
        out = torch.cat([b0, b1, b2], dim=1)
        if self.use_shuffle:
            out = self.shuffle(out)

        # 输出 + 残差增强
        out = self.bn_out(self.conv_out(out))
        return self.act(out + residual)

# =============== 轻量化模型架构（三分量版）=================
class LightweightPWaveNet(nn.Module):
    """
    三分量P+S波到时密度估计网络（高斯软标签）。

    参数
    ----
    input_dim     : 输入特征通道数，三分量版默认为 3
    hidden_dim    : 隐层维度，双向Mamba输出为 hidden_dim * 2
    use_transformer: True  → Mamba + TransformerEncoder 并联后融合
                    False → 仅使用 Mamba，去掉Transformer分支（更轻量）
    mamba_mode    : 'none' → 不使用 Mamba，直接使用投影后的特征
                    'unidirectional' → 使用单向 Mamba
                    'bidirectional' → 使用双向 Mamba
    """
    def __init__(self, input_dim=3, hidden_dim=48,
                 use_transformer: bool = True, use_lfem: bool = True,
                 mamba_mode: str = 'none', use_cross_channel_attention: bool = False,
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
            # LFEM1D 特征提取器
            self.lfem_block1 = LFEM1D(input_dim, expansion=1.0, kernel_size=7)
            self.pool1 = nn.MaxPool1d(2)

            self.lfem_block2 = LFEM1D(input_dim, expansion=1.0, kernel_size=5)
            self.pool2 = nn.MaxPool1d(2)

            self.lfem_block3 = LFEM1D(input_dim, expansion=1.5, kernel_size=3)
        else:
            # 普通 1D 卷积特征提取器
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

        # 特征投影层（proj_dim 可配：默认 96 = 原实现）
        self.proj = nn.Conv1d(input_dim, proj_dim, kernel_size=1)

        # Mamba：三种模式：none、unidirectional、bidirectional
        if mamba_mode == 'bidirectional':
            # 双向 Mamba：输出维度 = proj_dim * 2 = 192
            self.mamba = BidirectionalMamba(proj_dim, d_state=d_state, d_conv=4,
                                           expand=expand,
                                           num_layers=mamba_layers, dropout=0.1)
            mamba_out_dim = proj_dim * 2   # 192
        elif mamba_mode == 'unidirectional':
            # 单向 Mamba：输出维度 = proj_dim
            from mamba_ssm import Mamba
            self.mamba = nn.Sequential(*[
                Mamba(d_model=proj_dim, d_state=d_state, d_conv=4, expand=expand)
                for _ in range(mamba_layers)])
            mamba_out_dim = proj_dim   # 96
        elif mamba_mode == 'none':
            # 不使用 Mamba：直接使用投影后的特征
            self.mamba = nn.Identity()
            mamba_out_dim = proj_dim   # 96
        else:
            raise ValueError(f"Invalid mamba_mode: {mamba_mode}. Must be 'none', 'unidirectional', or 'bidirectional'")

        # Transformer 分支（可选）：输出维度 = transformer_dim
        transformer_dim = hidden_dim * 2  # 128
        if use_transformer:
            self.transformer = TransformerEncoder(
                proj_dim, transformer_dim, num_layers=2, nhead=4, dropout=0.1)
            fusion_in_dim = mamba_out_dim + transformer_dim
            # fusion_in_dim=128  # 192 + 128 = 320
        else:
            self.transformer = None
            fusion_in_dim = mamba_out_dim                    # 192

        # 融合输出统一为 fused_dim
        fused_dim = hidden_dim * 2   # 128
        # fused_dim = 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ── P 波密度估计头 ────────────────────────────────────────────────
        # 输出每个时间步的 logit（不加 Sigmoid/Softmax，
        # 由 GaussianLoss 内部做 softmax，保留完整梯度）
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # ── S 波密度估计头（与 P 头结构对称，共享 fused 特征，参数独立）──
        self.s_regressor = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        返回 (p_pred, s_pred)：
          - density 模式: 均为 (batch, seq_len) logits
        """
        # x: (batch, time, features)
        batch_size, seq_len, _ = x.shape

        x = x.transpose(1, 2)           # (batch, features, time)

        # LFEM1D 特征提取
        x = self.lfem_block1(x)          # (batch, input_dim, time)
        x = self.pool1(x)                 # (batch, input_dim, time/2)
        x = self.lfem_block2(x)          # (batch, input_dim, time/2)
        x = self.pool2(x)                 # (batch, input_dim, time/4)
        x = self.lfem_block3(x)          # (batch, input_dim, time/4)
        x = self.proj(x)                  # (batch, 96, time/4)
        x = x.transpose(1, 2)           # (batch, time', 96)

        # Mamba（根据 mamba_mode：none/unidirectional/bidirectional）
        mamba_out = self.mamba(x)     # (batch, time', hidden*2 或 hidden)

        # Transformer（可选）
        if self.use_transformer:
            transformer_out = self.transformer(x)          # (batch, time', hidden*2)
            combined = torch.cat([mamba_out, transformer_out], dim=-1)  # (batch, time', hidden*4)
        else:
            combined = mamba_out                           # (batch, time', hidden*2)

        # 融合 → (batch, time', hidden*2)
        fused = self.fusion(combined)

        # P 波密度头
        p_pred = self.regressor(fused).squeeze(-1)     # (batch, time')
        p_pred = F.interpolate(
            p_pred.unsqueeze(1), size=seq_len,
            mode='linear', align_corners=False
        ).squeeze(1)                                   # (batch, seq_len)

        # S 波密度头
        s_pred = self.s_regressor(fused).squeeze(-1)   # (batch, time')
        s_pred = F.interpolate(
            s_pred.unsqueeze(1), size=seq_len,
            mode='linear', align_corners=False
        ).squeeze(1)                                   # (batch, seq_len)

        return p_pred, s_pred


# =============== 损失函数 =================

class GaussianLoss(nn.Module):
    """
    高斯软标签专用损失函数（density 模式）。

    组合三项：
    1. KL 散度（KLDiv）：衡量预测概率分布与目标高斯分布的差异。
    2. 软 argmax L1：直接约束预测期望位置与目标位置之差，提供强梯度信号。
    3. 峰值区域 Focal MSE：对目标高斯值 > 阈值的位置加权 MSE，聚焦峰值形状。

    输入:
        pred   : (B, T) — 原始 logits（内部做 softmax）
        target : (B, T) — 归一化为概率分布（和为1）的高斯软标签

    参数
    ----
    kl_weight         : KL 散度项权重（默认 1.0）
    soft_argmax_weight: 软 argmax L1 项权重（默认 5.0）
    focal_weight      : 峰值 MSE 项权重（默认 2.0）
    focal_threshold   : 峰值 mask 阈值，目标值 > 此阈值的位置参与 focal MSE（默认 0.1）
    """
    def __init__(self, kl_weight=1.0, soft_argmax_weight=5.0,
                 focal_weight=2.0, focal_threshold=0.1):
        super().__init__()
        self.kl_weight          = kl_weight
        self.soft_argmax_weight = soft_argmax_weight
        self.focal_weight       = focal_weight
        self.focal_threshold    = focal_threshold
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(pred).all():
            return pred.sum() * 0.0

        pred_prob   = F.softmax(pred, dim=1)
        log_pred    = F.log_softmax(pred, dim=1)
        safe_target = target.clamp(min=0)
        kl = self.kl_loss(
            log_pred,
            safe_target / (safe_target.sum(dim=1, keepdim=True) + 1e-8)
        )

        T         = pred.shape[1]
        positions = torch.arange(T, device=pred.device, dtype=pred.dtype) / (T - 1)
        pred_pos  = (pred_prob * positions.unsqueeze(0)).sum(dim=1)
        tgt_pos   = (target   * positions.unsqueeze(0)).sum(dim=1)
        soft_l1   = F.l1_loss(pred_pos, tgt_pos)

        peak_mask = (target > self.focal_threshold).float()
        focal_mse = (peak_mask * (pred_prob - target) ** 2).sum(dim=1).mean()

        loss = (self.kl_weight          * kl
                + self.soft_argmax_weight * soft_l1
                + self.focal_weight       * focal_mse)

        return loss if torch.isfinite(loss) else soft_l1 * self.soft_argmax_weight


class JointPSLoss(nn.Module):
    """
    P+S 波联合损失（density 模式，高斯软标签）。

    总损失 = p_weight × p_loss + s_weight × s_loss

    S 波仅对 s_valid==1 的样本计算；若 batch 内无有效 S 样本则 s_loss=0。

    参数
    ----
    p_criterion : GaussianLoss
        P 波损失函数实例。
    s_criterion : GaussianLoss
        S 波损失函数实例，可与 P 不同配置。
    p_weight : float
        P 波损失权重系数（默认 1.0）。
    s_weight : float
        S 波损失权重系数（默认 1.0）。

    示例（S 波权重设为 0.5）
    ------------------------
        p_crit = GaussianLoss(kl_weight=1.0, soft_argmax_weight=5.0, focal_weight=2.0)
        s_crit = GaussianLoss(kl_weight=1.0, soft_argmax_weight=5.0, focal_weight=2.0)
        criterion = JointPSLoss(p_crit, s_crit, p_weight=1.0, s_weight=0.5)
    """

    def __init__(self,
                 p_criterion: GaussianLoss,
                 s_criterion: GaussianLoss,
                 p_weight: float = 1.0,
                 s_weight: float = 1.0):
        super().__init__()
        self.p_criterion = p_criterion
        self.s_criterion = s_criterion
        self.p_weight    = p_weight
        self.s_weight    = s_weight

    def forward(self, p_pred, s_pred, p_target, s_target, s_valid):
        """
        参数
        ----
        p_pred   : (B, T) logits
        s_pred   : (B, T) logits
        p_target : (B, T) 高斯软标签
        s_target : (B, T) 高斯软标签
        s_valid  : (B,)  1.0=有效 S 标签，0.0=无效

        返回
        ----
        total_loss : 标量，加权联合损失
        p_loss     : 标量，P 波损失（未加权）
        s_loss     : 标量，S 波损失（未加权，无有效样本时为 0）
        """
        # ── P 波损失（全样本）────────────────────────────────
        p_loss = self.p_criterion(p_pred, p_target)

        # ── S 波损失（仅有效样本）────────────────────────────
        mask = s_valid.bool()
        if mask.any():
            s_loss = self.s_criterion(s_pred[mask], s_target[mask])
        else:
            s_loss = torch.tensor(0.0, device=p_pred.device)

        total_loss = self.p_weight * p_loss + self.s_weight * s_loss
        return total_loss, p_loss, s_loss


# ================ 数据加载（三分量版，P+S双到时）=====================
def load_lightweight_data(file_name, csv_file, window_size=6000, max_samples=50000, seed=42,
                          trace_names=None, offset_reps=0, off_min=100, off_max=2900,
                          rng=None):
    """
    三分量数据加载，同时读取 P 波和 S 波到时。
    读取每条波形的三个分量 (Z, N, E)，分别预处理后拼接为 (window_size, 3)。

    返回:
        sequences  : np.ndarray, shape (M, window_size, 3)
        p_targets  : np.ndarray, shape (M,)   P波相对位置 [0, 1]
        s_targets  : np.ndarray, shape (M,)   S波相对位置 [0, 1]，
                     无有效S到时的样本填 -1.0（训练时可用 mask 过滤）
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

    _rng = rng if rng is not None else np.random.default_rng(seed)
    if offset_reps and offset_reps > 0:
        print(f"[位置增广] 每条 trace 生成 {offset_reps} 份，P 随机落在 "
              f"[{off_min}, {off_max}] 点（左侧不足时用镜像填充）")
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

                    # --- 形状检查 ---
                    if data.ndim < 2:
                        print(f"Warning: Data shape {data.shape} for {name} is not 2D")
                        continue

                    if data.shape[1] < 3:
                        print(f"Warning: Data has only {data.shape[1]} channels for {name}")
                        continue

                    # --- 提取三个分量 (T, 3)，顺序 Z, N, E ---
                    data_3c = data[:, CHANNEL_INDICES]  # (T, 3)

                    # --- P波到时 ---
                    p_arrival = dset.attrs.get('p_arrival_sample', -1)
                    if p_arrival == -1:
                        for attr_name in ['p_arrival', 'P_arrival_sample', 'p_pick']:
                            p_arrival = dset.attrs.get(attr_name, -1)
                            if p_arrival != -1:
                                break

                    if p_arrival <= 0 or p_arrival >= data_3c.shape[0]:
                        continue

                    # --- S波到时（可选：无效则记录 -1）---
                    s_arrival = dset.attrs.get('s_arrival_sample', -1)
                    if s_arrival == -1:
                        for attr_name in ['s_arrival', 'S_arrival_sample', 's_pick']:
                            s_arrival = dset.attrs.get(attr_name, -1)
                            if s_arrival != -1:
                                break

                    if data_3c.shape[0] < window_size:
                        continue

                    # --- 三分量独立预处理 ---
                    try:
                        data_3c = preprocess_3c(data_3c)   # (T, 3) float32
                    except Exception as e:
                        if error_count < 5:
                            print(f"Preprocessing error for {name}: {e}")
                        error_count += 1
                        continue

                    # --- 位置增广：把 P 随机摆到窗内 [off_min, off_max] ---
                    # offset_reps<=0 -> 老行为（P 固定落在窗内 ~p_arrival 处）
                    if offset_reps and offset_reps > 0:
                        targets = [int(_rng.integers(off_min, off_max))
                                   for _ in range(int(offset_reps))]
                    else:
                        targets = [None]

                    for _t in targets:
                        if _t is None:
                            ws = 0
                            win = data_3c[ws:ws + window_size, :]
                        else:
                            ws = int(p_arrival) - int(_t)       # 窗起点（可为负）
                            pad = max(0, -ws)                   # 左侧镜像填充长度
                            if pad > 0:
                                if data_3c.shape[0] < pad + 1:
                                    continue
                                # 用已有的 P 前噪声做镜像，视觉上仍是噪声
                                left = data_3c[1:pad + 1][::-1]
                                src = np.concatenate([left, data_3c], axis=0)
                                ws2 = ws + pad
                            else:
                                src = data_3c
                                ws2 = ws
                            win = src[ws2:ws2 + window_size, :]

                        if win.shape[0] < window_size:
                            continue

                        # --- 归一化相对位置标签（按实际窗起点换算）---
                        _ws = 0 if _t is None else ws
                        p_rel = (float(p_arrival) - _ws) / window_size
                        if not (0.0 <= p_rel <= 1.0):
                            continue
                        s_shift = float(s_arrival) - _ws
                        if (s_arrival > p_arrival) and (0.0 <= s_shift < window_size):
                            s_rel = s_shift / window_size
                            s_valid_count += 1
                        else:
                            s_rel = -1.0

                        if np.isnan(win).any() or np.isinf(win).any():
                            continue

                        sequences.append(win.astype(np.float32))
                        p_targets.append(float(p_rel))
                        s_targets.append(float(s_rel))
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


def plot_predicted_vs_true_bimamba(y_true: np.ndarray, y_pred: np.ndarray,
                                    save_path: str = 'predicted_vs_true.pdf',
                                    window_size: int = 6000):
    """绘制预测值 vs 真实值散点图"""
    y_true_s = y_true * window_size
    y_pred_s = y_pred * window_size

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    try:
        xy = np.vstack([y_true_s, y_pred_s])
        z  = gaussian_kde(xy)(xy)
        ax.scatter(y_true_s, y_pred_s, c=z, s=4, cmap='viridis',
                    alpha=0.6, rasterized=True)
    except Exception:
        ax.scatter(y_true_s, y_pred_s, s=4, alpha=0.3, color='steelblue')
    lim = [0, window_size]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect')
    ax.set_xlabel('True Position (samples)', fontsize=11)
    ax.set_ylabel('Predicted Position (samples)', fontsize=11)
    ax.set_title('Predicted vs True', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('BiMamba 3C LFEM Gauss — Predicted vs True',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Predicted vs True 图已保存: {save_path}")


def plot_residual_scatter_bimamba(y_true: np.ndarray, y_pred: np.ndarray,
                                   save_path: str = 'residual_scatter.pdf',
                                   window_size: int = 6000):
    """绘制残差散点图"""
    y_true_s = y_true * window_size
    y_pred_s = y_pred * window_size
    residuals = y_pred_s - y_true_s

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    ax.scatter(y_true_s, residuals, s=3, alpha=0.3, color='purple')
    ax.axhline(0,   color='red',    lw=2, linestyle='-',  label='Zero')
    ax.axhline(20,  color='orange', lw=1.5, linestyle='--', alpha=0.7, label='±20 samp')
    ax.axhline(-20, color='orange', lw=1.5, linestyle='--', alpha=0.7)
    ax.set_xlabel('True Position (samples)', fontsize=11)
    ax.set_ylabel('Residual (Pred − True)', fontsize=11)
    ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-150, 150)

    fig.suptitle('BiMamba 3C LFEM Gauss — Residual Scatter',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Residual Scatter 图已保存: {save_path}")


def plot_residual_histogram_bimamba(y_true: np.ndarray, y_pred: np.ndarray,
                                    save_path: str = 'residual_histogram.pdf',
                                    window_size: int = 6000):
    """绘制残差直方图"""
    y_true_s = y_true * window_size
    y_pred_s = y_pred * window_size
    residuals = y_pred_s - y_true_s
    mu, std = np.mean(residuals), np.std(residuals)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    ax.hist(residuals, bins=80, density=True, alpha=0.7,
             color='skyblue', edgecolor='black', range=(-150, 150),
             label='Residuals')
    x_fit = np.linspace(-150, 150, 300)
    ax.plot(x_fit, norm.pdf(x_fit, mu, std), 'r-', lw=2,
             label=f'Normal  μ={mu:.1f}, σ={std:.1f}')
    ax.axvline(0,   color='black',  lw=2, alpha=0.8, label='Zero')
    ax.axvline(20,  color='orange', lw=1.5, linestyle='--', alpha=0.6, label='±20')
    ax.axvline(-20, color='orange', lw=1.5, linestyle='--', alpha=0.6)
    ax.set_xlabel('Residual (samples)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    fig.suptitle('BiMamba 3C LFEM Gauss — Residual Histogram',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Residual Histogram 图已保存: {save_path}")


def plot_cdf_bimamba(y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: str = 'cdf.pdf',
                      window_size: int = 6000, fs=100):
    """绘制绝对误差的 CDF 图"""
    y_true_s = y_true * window_size
    y_pred_s = y_pred * window_size
    residuals = y_pred_s - y_true_s
    errors = np.abs(residuals)
    acc10 = np.mean(errors <= 10) * 100
    acc20 = np.mean(errors <= 20) * 100
    acc50 = np.mean(errors <= 50) * 100

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    sorted_err = np.sort(errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax.plot(sorted_err, cdf, lw=2.5, color='navy', label='Empirical CDF')
    ax.axvline(10, color='g',      lw=2, linestyle='--', alpha=0.8,
                label=f'10 samp: {acc10:.1f}%')
    ax.axvline(20, color='orange', lw=2, linestyle='--', alpha=0.8,
                label=f'20 samp: {acc20:.1f}%')
    ax.axvline(50, color='r',      lw=2, linestyle='--', alpha=0.8,
                label=f'50 samp: {acc50:.1f}%')
    ax.set_xlabel('Absolute Error (samples)', fontsize=11)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax.set_title('CDF of Absolute Error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    fig.suptitle('BiMamba 3C LFEM Gauss — CDF',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  CDF 图已保存: {save_path}")


def plot_density_heatmap_bimamba(y_true: np.ndarray, y_pred: np.ndarray,
                                  save_path: str = 'density_heatmap.pdf',
                                  window_size: int = 6000):
    """绘制密度热图"""
    y_true_s = y_true * window_size
    y_pred_s = y_pred * window_size

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    h, xe, ye = np.histogram2d(y_true_s, y_pred_s,
                               bins=60, range=[[0, window_size], [0, window_size]])
    im = ax.imshow(np.log10(h.T + 1), origin='lower', aspect='auto',
                    extent=[0, window_size, 0, window_size],
                    cmap='hot', interpolation='gaussian')
    ax.plot([0, window_size], [0, window_size], 'cyan', lw=2, label='Perfect')
    ax.set_xlabel('True Position (samples)', fontsize=11)
    ax.set_ylabel('Predicted Position (samples)', fontsize=11)
    ax.set_title('Density Heatmap (log scale)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    plt.colorbar(im, ax=ax, label='log10(Count+1)')

    fig.suptitle('BiMamba 3C LFEM Gauss — Density Heatmap',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Density Heatmap 图已保存: {save_path}")


def plot_summary_bimamba(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str = 'summary.pdf',
                          window_size: int = 6000, fs=100):
    """绘制统计摘要"""
    y_true_s = y_true * window_size
    y_pred_s = y_pred * window_size
    residuals = y_pred_s - y_true_s
    errors = np.abs(residuals)
    mu, std = np.mean(residuals), np.std(residuals)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(residuals ** 2))
    acc10 = np.mean(errors <= 10) * 100
    acc20 = np.mean(errors <= 20) * 100
    acc50 = np.mean(errors <= 50) * 100

    skewness  = stats.skew(residuals)
    kurtosis_ = stats.kurtosis(residuals)
    n_sample = min(5000, len(residuals))
    rng = np.random.default_rng(0)
    sample_res = rng.choice(residuals, n_sample, replace=False)
    _, shapiro_p = stats.shapiro(sample_res)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')

    summary = f"""
╔══════════════════════════════════════╗
║      Test Set Results                ║
╚══════════════════════════════════════╝

Samples  : {len(y_true):,}

Residual
  Mean   : {mu:.2f} samp
  Std    : {std:.2f} samp
  Skew   : {skewness:.3f}
  Kurt   : {kurtosis_:.3f}

Normality (Shapiro-Wilk)
  p-value: {shapiro_p:.4f}  {'✓ Normal' if shapiro_p > 0.05 else '✗ Non-normal'}

Error
  MAE    : {mae:.2f} samp  ({mae/fs:.3f} s)
  RMSE   : {rmse:.2f} samp  ({rmse/fs:.3f} s)
  Median : {np.median(errors):.2f} samp

Accuracy
  ≤10 samp : {acc10:.2f}%
  ≤20 samp : {acc20:.2f}%
  ≤50 samp : {acc50:.2f}%

Percentiles (|error|)
  50th : {np.percentile(errors, 50):.1f}
  75th : {np.percentile(errors, 75):.1f}
  90th : {np.percentile(errors, 90):.1f}
  95th : {np.percentile(errors, 95):.1f}
"""
    ax.text(0.05, 0.5, summary, fontsize=9.5, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle('BiMamba 3C LFEM Gauss — Statistics Summary',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Statistics Summary 图已保存: {save_path}")


def plot_batch_results(y_true: np.ndarray, y_pred: np.ndarray,
                       save_path: str = 'test_results.pdf',
                       window_size: int = 6000, fs=100):
    """
    批量推断结果综合可视化，生成单独的图形文件。
    """
    y_true_s = y_true * window_size
    y_pred_s = y_pred * window_size

    residuals = y_pred_s - y_true_s
    errors    = np.abs(residuals)
    mu, std   = np.mean(residuals), np.std(residuals)
    mae       = np.mean(errors)
    rmse      = np.sqrt(np.mean(residuals ** 2))
    acc10     = np.mean(errors <= 10) * 100
    acc20     = np.mean(errors <= 20) * 100
    acc50     = np.mean(errors <= 50) * 100

    base_path = save_path.replace('.pdf', '')
    
    plot_predicted_vs_true_bimamba(y_true, y_pred, f'{base_path}_predicted_vs_true.pdf', window_size)
    plot_residual_scatter_bimamba(y_true, y_pred, f'{base_path}_residual_scatter.pdf', window_size)
    plot_residual_histogram_bimamba(y_true, y_pred, f'{base_path}_residual_histogram.pdf', window_size)
    plot_cdf_bimamba(y_true, y_pred, f'{base_path}_cdf.pdf', window_size, fs)
    plot_density_heatmap_bimamba(y_true, y_pred, f'{base_path}_density_heatmap.pdf', window_size)
    plot_summary_bimamba(y_true, y_pred, f'{base_path}_summary.pdf', window_size, fs)

    print(f"  测试结果图已保存到: {base_path}_*.pdf")
    return mae, rmse, acc10, acc20, acc50


# ================ 训练函数 =====================
def train_lightweight_model(model, train_loader, val_loader, device,
                            num_epochs=CFG.get('epochs', 200), lr=1e-3,
                            use_gaussian=False,
                            p_weight=1.0, s_weight=1.0):
    """
    训练 P+S 波联合检测模型。

    参数
    ----
    p_weight : float
        P 波损失在联合损失中的权重系数（默认 1.0）。
    s_weight : float
        S 波损失在联合损失中的权重系数（默认 1.0）。
        若数据集中 S 标签较稀疏，可适当调低（如 0.5）以避免过度惩罚。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 70, 90], gamma=0.3)

    # ── 独立构造 P/S 损失函数实例，超参可各自调整 ─────────────────────
    p_criterion = GaussianLoss(kl_weight=1.0, soft_argmax_weight=5.0,
                               focal_weight=2.0, focal_threshold=0.1)
    s_criterion = GaussianLoss(kl_weight=1.0, soft_argmax_weight=5.0,
                               focal_weight=2.0, focal_threshold=0.1)

    criterion = JointPSLoss(p_criterion=p_criterion,
                            s_criterion=s_criterion,
                            p_weight=p_weight,
                            s_weight=s_weight)

    print(f"Loss weights — P: {p_weight:.3f}, S: {s_weight:.3f}")

    best_mae = float('inf')
    train_losses, val_maes = [], []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        nan_batches = 0

        for batch in train_loader:
            x, p_label, s_label, s_valid = batch
            x       = x.to(device)
            p_label = p_label.to(device)
            s_label = s_label.to(device)
            s_valid = s_valid.to(device)

            if torch.isnan(x).any() or torch.isnan(p_label).any():
                nan_batches += 1
                continue

            p_pred, s_pred = model(x)
            total_loss, p_loss, s_loss = criterion(
                p_pred, s_pred, p_label, s_label, s_valid)

            if not torch.isfinite(total_loss):
                nan_batches += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        scheduler.step()

        if nan_batches > 0:
            print(f"  Warning: {nan_batches} NaN batches skipped this epoch")

        val_p_mae, val_s_mae, val_p_rmse, val_p_r2 = evaluate_lightweight(
            model, val_loader, device, use_gaussian=use_gaussian)

        train_losses.append(epoch_loss / max(1, len(train_loader)))
        val_maes.append(val_p_mae)

        if val_p_mae < best_mae:
            best_mae = val_p_mae
            torch.save({'model_state_dict': model.state_dict(),
                        'cfg': dict(proj_dim=CFG.get('proj_dim', 96),
                                    hidden_dim=CFG.get('hidden_dim', 48),
                                    mamba_layers=CFG.get('mamba_layers', 3),
                                    d_state=CFG.get('d_state', 16),
                                    expand=CFG.get('expand', 2),
                                    window_size=CFG.get('window_size', 3000),
                                    tag=CFG.get('tag', ''),
                                    offset_reps=CFG.get('offset_reps', 0),
                                    mamba_mode=CFG.get('mamba_mode', 'bidirectional'),
                                    use_transformer=CFG.get('use_transformer', True),
                                    use_lfem=CFG.get('use_lfem', True),
                                    use_cross_channel_attention=CFG.get(
                                        'use_cross_channel_attention', False))},
                       f'best_lightweight_model_3C{CFG.get("tag", "")}.pth')

        epoch_time     = time.time() - epoch_start
        remaining_time = epoch_time * (num_epochs - epoch - 1)
        hrs, rem       = divmod(remaining_time, 3600)
        mins, secs     = divmod(rem, 60)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            s_info = f", S_MAE={val_s_mae:.2f}" if val_s_mae >= 0 else ""
            print(f"Epoch {epoch:02d}: Loss={epoch_loss / max(1, len(train_loader)):.8f}, "
                  f"P_MAE={val_p_mae:.2f}{s_info}, RMSE={val_p_rmse:.2f}, "
                  f"R2={val_p_r2:.4f}, Best={best_mae:.2f}, "
                  f"ETA: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}")

    return train_losses, val_maes


def evaluate_lightweight(model, loader, device, use_gaussian=False):
    """
    评估：返回 P_MAE, S_MAE, P_RMSE, P_R²（单位为采样点）。
    S_MAE = -1.0 表示验证集内无有效 S 标签。
    """
    model.eval()
    p_true, p_pred_list = [], []
    s_true, s_pred_list = [], []

    with torch.no_grad():
        for batch in loader:
            x, p_label, s_label, s_valid = batch
            x       = x.to(device)
            p_label = p_label.to(device)
            s_label = s_label.to(device)
            s_valid = s_valid.to(device)

            p_pred, s_pred = model(x)

            if use_gaussian:
                T = p_pred.shape[1]
                positions = torch.arange(T, device=device, dtype=p_pred.dtype) / (T - 1)

                p_prob  = F.softmax(p_pred, dim=1)
                p_pos   = (p_prob * positions.unsqueeze(0)).sum(dim=1)
                pt_pos  = (p_label * positions.unsqueeze(0)).sum(dim=1)
                p_true.extend(pt_pos.cpu().numpy())
                p_pred_list.extend(p_pos.cpu().numpy())

                # S 波：仅对有效样本收集
                mask = s_valid.bool()
                if mask.any():
                    s_prob  = F.softmax(s_pred[mask], dim=1)
                    s_pos   = (s_prob * positions.unsqueeze(0)).sum(dim=1)
                    st_pos  = (s_label[mask] * positions.unsqueeze(0)).sum(dim=1)
                    s_true.extend(st_pos.cpu().numpy())
                    s_pred_list.extend(s_pos.cpu().numpy())
            else:
                p_true.extend(p_label.cpu().numpy())
                p_pred_list.extend(p_pred.cpu().numpy())
                mask = s_valid.bool()
                if mask.any():
                    s_true.extend(s_label[mask].cpu().numpy())
                    s_pred_list.extend(s_pred[mask].cpu().numpy())

    # 必须与 main() 的训练窗口一致，否则 MAE/RMSE 会按比例失真
    window_size = int(CFG.get('window_size', 3000))

    p_true_s = np.array(p_true).flatten() * window_size
    p_pred_s = np.array(p_pred_list).flatten() * window_size
    p_mae  = np.mean(np.abs(p_true_s - p_pred_s))
    p_rmse = np.sqrt(np.mean((p_true_s - p_pred_s) ** 2))
    y_mean = np.mean(p_true_s) if p_true_s.size > 0 else 0.0
    ss_tot = np.sum((p_true_s - y_mean) ** 2)
    ss_res = np.sum((p_true_s - p_pred_s) ** 2)
    p_r2   = 0.0 if ss_tot <= 0 else 1.0 - (ss_res / ss_tot)

    if len(s_true) > 0:
        s_true_s  = np.array(s_true).flatten() * window_size
        s_pred_s  = np.array(s_pred_list).flatten() * window_size
        s_mae = np.mean(np.abs(s_true_s - s_pred_s))
    else:
        s_mae = -1.0   # 无有效 S 标签

    return p_mae, s_mae, p_rmse, p_r2


# ==================== 主函数 ====================
def main(seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 50)
    print("轻量化P+S波检测模型 — 三分量版本")
    print(f"输入分量: Z(col{CH_Z}), N(col{CH_N}), E(col{CH_E})")
    print(f"Random seed: {seed}")
    print("=" * 50)

    window_size = int(CFG.get('window_size', 3000))
    print(f'[窗长] window_size = {window_size} 采样点 ({window_size/100:.1f}s)')

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
                seed=seed, trace_names=names,
                offset_reps=(CFG['offset_reps'] if split == 'train' else 0),
                off_min=CFG['off_min'], off_max=CFG['off_max'],
                rng=np.random.default_rng(seed))
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
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    for _nm, _arr in (('train', s_train), ('val', s_val), ('test', s_test)):
        if len(_arr):
            print(f"  {_nm}: S-wave valid fraction "
                  f"{(np.asarray(_arr) >= 0).mean() * 100:.1f}%")

    # 波形库：排除 val/test 事件，避免测试事件经增强泄漏进训练
    wave_pool = None
    if CFG['pool_chunk']:
        pm = load_chunks(_data_dir(), [CFG['pool_chunk']])
        pn = pool_names(pm, exclude, CFG['pool_samples'], seed)
        phf, pcf = _paths(CFG['pool_chunk'])
        pool_seq, _, _ = load_lightweight_data(
            phf, pcf, window_size=window_size,
            max_samples=CFG['pool_samples'], seed=seed, trace_names=pn)
        if len(pool_seq) > 0:
            wave_pool = pool_seq.astype(np.float32)
            print(f"Waveform pool shape: {wave_pool.shape} "
                  f"(已排除 val/test 事件)")

    # 兼容原代码后续对 waves.shape 的使用
    waves = X_train if len(X_train) else np.zeros((0, window_size, 3), np.float32)
    print(f"Training data shape: {waves.shape}")

    if CFG.get('dry_run'):
        _dry_report(X_train, p_train, s_train, X_val, p_val, s_val,
                    X_test, p_test, s_test, wave_pool, meta)
        return

    # 初始化增强器
    augmentation = SimpleAugmentation(
        per_ch=1,
        second_event_pool=wave_pool
    )

    # 高斯软标签开关
    use_gaussian = True
    gaussian_sigma = 10
    print(f"use_gaussian = {use_gaussian}  (sigma={gaussian_sigma} samples)")

    train_ds = LightweightWaveDataset(
        X_train, p_train, s_targets=s_train,
        is_train=True, transforms=augmentation,
        use_gaussian=use_gaussian, window_size=window_size, sigma=gaussian_sigma)
    val_ds = LightweightWaveDataset(
        X_val, p_val, s_targets=s_val,
        is_train=False, scaler=train_ds.scaler,
        use_gaussian=use_gaussian, window_size=window_size, sigma=gaussian_sigma)
    test_ds = LightweightWaveDataset(
        X_test, p_test, s_targets=s_test,
        is_train=False, scaler=train_ds.scaler,
        use_gaussian=use_gaussian, window_size=window_size, sigma=gaussian_sigma)

    with open('scaler_3C.pkl', 'wb') as f:
        pickle.dump(train_ds.scaler, f)
    print("Scaler saved to 'scaler_3C.pkl'")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=128,  shuffle=False)

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
        use_lfem=CFG.get('use_lfem', True),
        use_cross_channel_attention=CFG.get('use_cross_channel_attention', False),
    ).to(device)
    print(f"[宽度] proj_dim={CFG.get('proj_dim',96)} hidden_dim={CFG.get('hidden_dim',48)} "
          f"mamba_layers={CFG.get('mamba_layers',3)} d_state={CFG.get('d_state',16)} "
          f"expand={CFG.get('expand',2)}")
    print(f"[模块] mamba_mode={mamba_mode} use_transformer={use_transformer} "
          f"use_lfem={CFG.get('use_lfem', True)} "
          f"cross_channel_attn={CFG.get('use_cross_channel_attention', False)}")
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # ── P/S 损失权重（可在此处调整）────────────────────────────────────
    # p_weight : P 波损失权重，通常置 1.0 作为基准
    # s_weight : S 波损失权重；S 标签稀疏时可降低（如 0.5），
    #            S 波精度要求高时可提升（如 1.5）
    p_weight = 1.0
    s_weight = 0.6
    print(f"Loss weights — P: {p_weight:.3f}, S: {s_weight:.3f}")

    # 训练
    print("\nTraining (joint P+S loss)...")
    _ep = CFG.get('epochs', 100)
    print(f"[主路径] 训练轮数 = {_ep}")
    train_losses, val_maes = train_lightweight_model(
        model, train_loader, val_loader, device,
        num_epochs=_ep, lr=1e-3, use_gaussian=use_gaussian,
        p_weight=p_weight, s_weight=s_weight
    )

    # ─────────────── 训练后评估 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("=" * 60)
    _best = f'best_lightweight_model_3C{CFG.get("tag", "")}.pth'
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

    p_true_list, p_pred_list = [], []
    s_true_list, s_pred_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, p_label, s_label, s_valid = batch
            x        = x.to(device)
            p_label  = p_label.to(device)
            s_label  = s_label.to(device)
            s_valid  = s_valid.to(device)

            p_pred, s_pred = model(x)

            if use_gaussian:
                T = p_pred.shape[1]
                positions = torch.arange(T, device=device, dtype=p_pred.dtype) / (T - 1)

                p_prob = F.softmax(p_pred, dim=1)
                p_pos  = (p_prob * positions.unsqueeze(0)).sum(dim=1)
                pt_pos = (p_label * positions.unsqueeze(0)).sum(dim=1)
                p_true_list.extend(pt_pos.cpu().numpy())
                p_pred_list.extend(p_pos.cpu().numpy())

                mask = s_valid.bool()
                if mask.any():
                    s_prob = F.softmax(s_pred[mask], dim=1)
                    s_pos  = (s_prob * positions.unsqueeze(0)).sum(dim=1)
                    st_pos = (s_label[mask] * positions.unsqueeze(0)).sum(dim=1)
                    s_true_list.extend(st_pos.cpu().numpy())
                    s_pred_list.extend(s_pos.cpu().numpy())
            else:
                p_true_list.extend(p_label.cpu().numpy())
                p_pred_list.extend(p_pred.cpu().numpy())
                mask = s_valid.bool()
                if mask.any():
                    s_true_list.extend(s_label[mask].cpu().numpy())
                    s_pred_list.extend(s_pred[mask].cpu().numpy())

    # ── P 波评估 ──
    p_true_s = np.array(p_true_list).flatten() * window_size
    p_pred_s = np.array(p_pred_list).flatten() * window_size
    p_errors = p_true_s - p_pred_s
    p_mae    = np.mean(np.abs(p_errors))
    p_rmse   = np.sqrt(np.mean(p_errors ** 2))
    ss_res   = np.sum((p_true_s - p_pred_s) ** 2)
    ss_tot   = np.sum((p_true_s - np.mean(p_true_s)) ** 2)
    p_r2     = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    p_acc10  = np.mean(np.abs(p_errors) <= 10) * 100
    p_acc20  = np.mean(np.abs(p_errors) <= 20) * 100
    p_acc50  = np.mean(np.abs(p_errors) <= 50) * 100

    print(f"\n[P-wave Results]  n={len(p_true_s)}")
    print(f"  MAE    : {p_mae:.2f} samples  ({p_mae/100:.3f} s)")
    print(f"  RMSE   : {p_rmse:.2f} samples  ({p_rmse/100:.3f} s)")
    print(f"  R²     : {p_r2:.4f}")
    print(f"  Acc@10 : {p_acc10:.1f}%")
    print(f"  Acc@20 : {p_acc20:.1f}%")
    print(f"  Acc@50 : {p_acc50:.1f}%")

    # ── S 波评估（若有有效样本）──
    if len(s_true_list) > 0:
        s_true_s = np.array(s_true_list).flatten() * window_size
        s_pred_s = np.array(s_pred_list).flatten() * window_size
        s_errors = s_true_s - s_pred_s
        s_mae    = np.mean(np.abs(s_errors))
        s_rmse   = np.sqrt(np.mean(s_errors ** 2))
        ss_res_s = np.sum((s_true_s - s_pred_s) ** 2)
        ss_tot_s = np.sum((s_true_s - np.mean(s_true_s)) ** 2)
        s_r2     = 0.0 if ss_tot_s <= 0 else 1.0 - ss_res_s / ss_tot_s
        s_acc10  = np.mean(np.abs(s_errors) <= 10) * 100
        s_acc20  = np.mean(np.abs(s_errors) <= 20) * 100
        s_acc50  = np.mean(np.abs(s_errors) <= 50) * 100

        print(f"\n[S-wave Results]  n={len(s_true_s)}")
        print(f"  MAE    : {s_mae:.2f} samples  ({s_mae/100:.3f} s)")
        print(f"  RMSE   : {s_rmse:.2f} samples  ({s_rmse/100:.3f} s)")
        print(f"  R²     : {s_r2:.4f}")
        print(f"  Acc@10 : {s_acc10:.1f}%")
        print(f"  Acc@20 : {s_acc20:.1f}%")
        print(f"  Acc@50 : {s_acc50:.1f}%")
    else:
        print("\n[S-wave Results] No valid S arrivals in test set — skipped.")
        s_true_s, s_pred_s = None, None

    # ── P 波可视化 ──
    p_true_norm = p_true_s / window_size
    p_pred_norm = p_pred_s / window_size
    plot_batch_results(p_true_norm, p_pred_norm,
                       save_path='test_set_results_P.pdf', window_size=window_size)

    # ── S 波可视化（若有）──
    if s_true_s is not None:
        s_true_norm = s_true_s / window_size
        s_pred_norm = s_pred_s / window_size
        plot_batch_results(s_true_norm, s_pred_norm,
                           save_path='test_set_results_S.pdf', window_size=window_size)

    # ── 训练过程可视化 ──
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss (Joint P+S)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(val_maes, label='P-wave MAE')
    plt.title('Validation P-wave MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE (samples)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(p_true_s, p_pred_s, alpha=0.5, s=1, label='P-wave')
    if s_true_s is not None:
        plt.scatter(s_true_s, s_pred_s, alpha=0.5, s=1, label='S-wave', color='orange')
    plt.plot([0, window_size], [0, window_size], 'r--')
    plt.xlabel('True Arrival (samples)')
    plt.ylabel('Predicted Arrival (samples)')
    plt.title('Pred vs Truth (P & S)')
    plt.legend(markerscale=5)

    plt.tight_layout()
    plt.savefig('lightweight_results_3C.pdf', dpi=600)
    plt.show()

    print("\nModel saved as 'best_lightweight_model_3C.pth'")
    print("Results plots saved as 'test_set_results_P.pdf' / 'test_set_results_S.pdf'")
    print("Training curve saved as 'lightweight_results_3C.pdf'")


if __name__ == '__main__':
    main(seed=apply_cli())

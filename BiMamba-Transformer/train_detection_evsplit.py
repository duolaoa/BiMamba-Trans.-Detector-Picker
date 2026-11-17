# 双头检测模型 - 三分量版本（双向Mamba版）
# 1. 共享 backbone 同时输出三路：P检测头 / S检测头 / P精定位头
# 2. P 检测标签：固定宽度矩形（不再用 1.4×(S-P)，避免标签几何随震中距/震级漂移）
# 3. P 定位标签：高斯软标签 + offset 随机化，损失用 KL + soft-argmax L1 + focal MSE
# 4. 输入：6000点 三分量波形（Z/N/E，已预处理）
#
# 与 PS_Detection.py 的关系：模型结构 / 预处理 / 数据载入口径保持一致，
# 复用其中的 CHANNEL_INDICES / preprocess_3c / LFEM1D / BidirectionalMamba /
# TransformerEncoder / SimpleAugmentation，因此旧检测权重可直接加载到共享部分。
#
# 组织方式与 PS_Detection.py 对齐：
#   [标签生成] → [数据增强] → [Dataset] → [模型] → [损失] → [数据加载]
#   → [train_dual_head_model] → [evaluate_dual_head / compute_detection_metrics]
#   → [print_detection_report / plot_confusion_matrix / plot_detection_example
#      / plot_training_history] → [main(seed=42)]
#
# 显存说明（6000 点窗口 / bidirectional Mamba / fp32，实测 peak allocated）
# ---------------------------------------------------------------------
#   Transformer 自注意力作用于 T/4 = 1500 的序列，成本 O(B·nhead·T²)：
#   batch=256 时单层注意力矩阵 256×4×1500²×4B ≈ 9.2 GB，必然 OOM。
#   梯度累积把「峰值 batch」降到 micro-batch，等效 batch 不变、峰值不叠加。
#
#   batch=256（旧默认）                        → OOM
#   batch=32 × accum=8, 无 Transformer         → 2.77 GiB
#   batch=32 × accum=8, 有 Transformer         → 3.50 GiB
#   batch=64 × accum=4, 有 Transformer         → 3.76 GiB
#   batch=32 × accum=8, 有 Transformer +检查点  → 3.46 GiB
#
#   默认（--batch-size 32 --accum-steps 8 --no-transformer）已在 24GB 卡可用，
#   等效 batch 256 与原设定一致；要开 Transformer 请再配 --grad-checkpoint。
#   显存仍紧可加 --use-amp，但 Mamba 的 CUDA 核多为 fp32，收益有限且偶有数值
#   不稳，建议优先调 batch/accum。碎片问题可设
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
import pickle
import datetime
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
import warnings
import time
import os
import gc
import glob
from scipy.stats import norm, gaussian_kde
from scipy import stats
import math

# =========================================================================== #
# 事件级划分（回应审稿人：Please split by event）
# --------------------------------------------------------------------------- #
# 原实现把信号与噪声合并后用 train_test_split 按 **trace** 随机划分。STEAD 中同一
# 事件会有多条记录（跨 chunk、多台站），实测 chunk2+chunk3 上 **65.1% 的测试 trace
# 其事件也出现在训练集中**。这里改为按事件划分：先把事件分配到 train/val/test，
# 再在各集合内部抽样；波形库（第二事件增强）排除 val/test 的事件。
# 与三个拾取模型共用同一张 --split-table，保证全流程同一划分。
# =========================================================================== #
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# 本脚本 import 的 PS_Detection.py 位于上一级目录，一并加入搜索路径，
# 这样从任意工作目录运行都能导入（原脚本依赖"在 PS_Detection.py 同目录下运行"）。
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from event_split import (assert_disjoint, build_split_table, describe_splits,
                         events_in, load_chunks, load_split_table,
                         names_for_split, pool_names, save_split_table,
                         split_caps)

# 复用 PS_Detection 的既有约定（通道顺序、预处理、backbone 组件、增强）
from PS_Detection_evsplit import (
    CH_Z, CH_N, CH_E,
    CHANNEL_INDICES,
    N_CHANNELS,
    preprocess_3c,
    SimpleAugmentation,
    LFEM1D,
    BidirectionalMamba,
    TransformerEncoder,
    CrossChannelAttention,
)

warnings.filterwarnings('ignore')

# =================== 全局常量 ===================
WINDOW_SIZE = 6000        # 窗口大小（采样点）
FS          = 100         # 采样率 (Hz)
POOL_FACTOR = 4           # LFEM 两级池化 → 特征时间分辨率 T/4
NUM_EPOCHS  = 30          # 默认训练轮数
# P 位置先验自检用的 offset 列表（覆盖宽范围，便于发现位置依赖）
POS_CHECK_OFFSETS = (200, 550, 1000, 1600, 2400, 3200)

# 默认数据划分（与 PS_Detection.main 一致：chunk2/3 训练、chunk1 噪声、chunk4 波形库）
_DEFAULT_TRAIN_FILES = [('chunk2.hdf5', 'chunk2.csv'),
                        ('chunk3.hdf5', 'chunk3.csv')]
_DEFAULT_NOISE_FILES = [('chunk1.hdf5', 'chunk1.csv')]
_DEFAULT_POOL_FILES  = [('chunk4.hdf5', 'chunk4.csv')]


# =================== 标签生成 ===================
def generate_rect_label(window_size, p_center, width):
    """
    P 波固定宽度矩形标签（纯检测用）。

    与原版 generate_seismic_phase_label 的区别：宽度**不随 (S-P) 变化**。
    实测原标签宽度 ≈ 1.4×(S-P) 在 INSTANCE 上跨度 200~4400 采样点，
    导致网络输出平台的几何随震中距/震级漂移：输出平台/标签几何比在 1~2 时
    锚点 MAE ≈ 20，比值 >3 时 MAE 915。固定宽度后所有样本标签几何一致。

    输入:
        window_size : 窗口大小
        p_center    : P波到时相对位置 [0, 1]
        width       : 正区间长度（采样点）
    输出:
        label : shape (window_size,)，正样本区内值=1.0，其余0.0
    """
    p_idx = int(round(p_center * window_size))
    label = np.zeros(window_size, dtype=np.float32)
    if 0 <= p_idx < window_size:
        hi = min(window_size, p_idx + max(1, int(width)))
        label[p_idx:hi] = 1.0
    return label


def generate_rect_label_adaptive(window_size, p_center, s_center,
                                 scale=1.4, w_min=200, w_max=800):
    """
    P 波截断式自适应矩形标签：宽度 = clip(scale×(S-P), w_min, w_max)。

    保留一点对短 S-P 事件的自适应（避免矩形越过 S），但用上下限消除几何漂移。

    输入:
        window_size : 窗口大小
        p_center    : P波到时相对位置 [0, 1]
        s_center    : S波到时相对位置 [0, 1]，< 0 表示无效S到时
    输出:
        label : shape (window_size,)
    """
    p_idx = int(round(p_center * window_size))
    if s_center >= 0:
        s_idx = int(round(s_center * window_size))
        width = int(round(scale * max(s_idx - p_idx, 1)))
    else:
        width = w_min
    width = int(np.clip(width, w_min, w_max))
    return generate_rect_label(window_size, p_center, width)


def generate_s_phase_label(window_size, p_center, s_center, scale=0.5):
    """
    S 波相位矩形标签：从 S 波到时前 0.15×(S-P) 开始，持续 scale×(S-P) 长度。
    语义与 PS_Detection.generate_s_phase_label 一致。

    输入:
        window_size : 窗口大小
        p_center    : P波到时相对位置 [0, 1]
        s_center    : S波到时相对位置 [0, 1]，< 0 表示无效S到时
    输出:
        label : shape (window_size,)
    """
    label = np.zeros(window_size, dtype=np.float32)
    if s_center < 0:
        return label
    s_idx = int(round(s_center * window_size))
    p_idx = int(round(p_center * window_size))
    sp_gap = max(1, s_idx - p_idx)
    phase_len = max(100, int(round(scale * sp_gap)))
    start = max(0, s_idx - int(0.15 * sp_gap))
    hi = min(window_size, start + phase_len)
    if hi > start:
        label[start:hi] = 1.0
    return label


def generate_gaussian_label(window_size, center, sigma=8.0):
    """
    P 波高斯软标签（归一化为概率分布，和为1），供精定位头使用。

    输入:
        window_size : 窗口大小
        center      : P波中心位置，相对比例 [0, 1]
        sigma       : 高斯标准差（采样点，默认 8）
    输出:
        label : shape (window_size,)，和为 1 的概率分布
    """
    center_idx = float(center) * window_size
    x = np.arange(window_size, dtype=np.float32)
    gaussian = np.exp(-0.5 * ((x - center_idx) / sigma) ** 2)
    s = gaussian.sum()
    if s > 0:
        gaussian /= s
    return gaussian.astype(np.float32)


def build_offset_shift(p_pos_in_window, p_offset_min, p_offset_max,
                       is_train, rng):
    """
    计算窗口平移量 shift，使训练时 P 落在 [p_offset_min, p_offset_max] 采样点。

    shift > 0 → 窗口起点左移，P 在窗内位置由 ``p_pos_in_window`` 变为
    ``p_pos_in_window − shift``；Dataset 内波形与标签同步平移，二者始终一致。

    实测意义：固定 offset 会让模型在 P 贴近窗口边界时崩溃
    （≤20 命中 92%→8%），随机化后可容忍残余锚点误差。
    """
    base = np.asarray(p_pos_in_window, dtype=float)
    if is_train and p_offset_max > p_offset_min:
        want = rng.integers(p_offset_min, p_offset_max + 1, size=len(base))
    else:
        want = np.full(len(base), (p_offset_min + p_offset_max) // 2, dtype=int)
    return (base - want).astype(int)


# =================== 数据增强 ===================
class SimpleAugmentation3C:
    """
    包装 PS_Detection.SimpleAugmentation，作为双头检测的默认增强器。

    窗口整体平移（offset 随机化）由 Dataset 负责（保证标签同步位移），
    此处只做波形层面的幅度/噪声/分量丢弃/第二事件叠加增强。
    """

    def __init__(self, per_ch=1, second_event_pool=None, **kwargs):
        self.base = SimpleAugmentation(per_ch=per_ch,
                                       second_event_pool=second_event_pool,
                                       **kwargs)

    def __call__(self, x):
        return self.base(x)


# =================== 数据集 ===================
class DualHeadWaveDataset(Dataset):
    """
    双头检测数据集。

    每条样本产出：
        x        : (T, 3) 标准化后的波形
        p_rect   : (T,)   矩形标签     → p_det_head（BCEWithLogitsLoss）
        s_rect   : (T,)   矩形标签     → s_det_head（BCEWithLogitsLoss）
        p_gauss  : (T,)   高斯软标签   → pick_head （GaussianPickLoss）
        s_valid  : ()     1=有效 S
        is_noise : ()     1=噪声样本（标签全零 + 检测头抑制）

    参数
    ----
    waves        : (N, T, F) 已 preprocess_3c 的原始波形
    p_targets    : (N,) 窗内 P 位置（相对比例 [0,1]）；噪声样本填 -2.0
    s_targets    : (N,) 窗内 S 位置（相对比例）；无效填 -1.0
    shift        : (N,) 窗口平移量（采样点），>0 表示窗口起点左移。
                   波形与标签同步位移，保证二者一致。
    is_train     : 训练模式（启用增强）
    scaler       : 标准化器（None 则在本数据集上拟合）
    transforms   : 数据增强器
    rect_width   : P 矩形标签固定宽度（采样点）
    rect_adaptive: True 时用 clip(rect_scale×(S-P), rect_w_min, rect_w_max)
    pick_sigma   : 精定位高斯标签的 σ（采样点）
    """

    def __init__(self, waves, p_targets, s_targets, shift=None,
                 is_train=True, scaler=None, transforms=None,
                 window_size=WINDOW_SIZE, rect_width=300,
                 rect_adaptive=False, rect_scale=1.4,
                 rect_w_min=200, rect_w_max=800,
                 s_phase_scale=0.5, pick_sigma=8.0, base_offset=0):
        if waves.ndim == 2:
            waves = waves[..., None]

        self.waves          = waves.astype(np.float32)
        self.p_targets      = p_targets.astype(np.float32)
        self.s_targets      = (s_targets.astype(np.float32)
                               if s_targets is not None
                               else np.full(len(p_targets), -1.0, dtype=np.float32))
        self.shift          = (np.zeros(len(waves), dtype=int) if shift is None
                               else np.asarray(shift, dtype=int))
        self.is_train       = is_train
        self.transforms     = transforms
        self.window_size    = window_size
        self.rect_width     = rect_width
        self.rect_adaptive  = rect_adaptive
        self.rect_scale     = rect_scale
        self.rect_w_min     = rect_w_min
        self.rect_w_max     = rect_w_max
        self.s_phase_scale  = s_phase_scale
        self.pick_sigma     = pick_sigma
        # base_offset 仅作记录；offset 随机化在取窗阶段完成

        # 标准化
        if scaler is None:
            self.scaler = StandardScaler()
            w2 = self.waves.reshape(-1, self.waves.shape[-1])
            self.scaler.fit(w2)
        else:
            self.scaler = scaler

        w2  = self.waves.reshape(-1, self.waves.shape[-1])
        w2n = self.scaler.transform(w2)
        self.waves = w2n.reshape(self.waves.shape)

    def __len__(self):
        return len(self.waves)

    def _shifted(self, idx):
        """按 self.shift 平移窗口，越界处镜像填充，返回 (T, F)。"""
        T = self.window_size
        sh = int(self.shift[idx])
        if sh == 0:
            return self.waves[idx]
        w = self.waves[idx]
        if sh > 0:
            x = w[sh:]
            need = T - x.shape[0]
            if need > 0:
                pad = np.resize(w[:sh][::-1], (need, w.shape[1]))
                x = np.concatenate([x, pad], 0)
        else:
            x = w[:T + sh]
            need = T - x.shape[0]
            if need > 0:
                pad = np.resize(w[T + sh:][::-1], (need, w.shape[1]))
                x = np.concatenate([pad, x], 0)
        return np.ascontiguousarray(x)

    def __getitem__(self, idx):
        x     = self._shifted(idx).copy()
        p_pos = float(self.p_targets[idx])
        s_pos = float(self.s_targets[idx])
        sh    = int(self.shift[idx])

        if self.is_train and self.transforms:
            x = self.transforms(x)

        # ── 噪声样本（p_pos <= -1.5）────────────────────────────────────
        is_noise_flag = 1.0 if p_pos <= -1.5 else 0.0

        if is_noise_flag:
            return (torch.from_numpy(x).float(),
                    torch.zeros(self.window_size),   # p_rect  (全零)
                    torch.zeros(self.window_size),   # s_rect  (全零)
                    torch.zeros(self.window_size),   # p_gauss (全零)
                    torch.tensor(0.0),               # s_valid
                    torch.tensor(1.0))               # is_noise

        # ── 地震信号样本 ────────────────────────────────────────────────
        # 平移后 P/S 在窗内的相对位置
        p_pos = float(np.clip(p_pos - sh / self.window_size, 0.0, 1.0))
        if s_pos >= 0:
            s_pos = float(np.clip(s_pos - sh / self.window_size, 0.0, 1.0))
        p_idx = int(round(p_pos * self.window_size))
        s_idx = int(round(s_pos * self.window_size)) if s_pos >= 0 else -1

        # S 有效性：必须在窗内且晚于 P
        s_valid = 1.0 if (0 <= s_idx < self.window_size and s_idx > p_idx) else 0.0
        s_pos_eff = s_pos if s_valid > 0.5 else -1.0

        # ── P 检测标签：固定宽度矩形（或截断式自适应）──────────────────
        if self.rect_adaptive:
            p_rect = generate_rect_label_adaptive(
                self.window_size, p_pos, s_pos_eff, self.rect_scale,
                self.rect_w_min, self.rect_w_max)
        else:
            p_rect = generate_rect_label(self.window_size, p_pos,
                                         self.rect_width)

        # ── S 检测标签 ──────────────────────────────────────────────────
        s_rect = generate_s_phase_label(self.window_size, p_pos, s_pos_eff,
                                        self.s_phase_scale)

        # ── P 精定位标签（高斯软标签）──────────────────────────────────
        p_gauss = generate_gaussian_label(self.window_size, p_pos,
                                          self.pick_sigma)

        return (torch.from_numpy(x).float(),
                torch.tensor(p_rect,  dtype=torch.float32),
                torch.tensor(s_rect,  dtype=torch.float32),
                torch.tensor(p_gauss, dtype=torch.float32),
                torch.tensor(s_valid, dtype=torch.float32),
                torch.tensor(is_noise_flag, dtype=torch.float32))


# =================== 模型 ===================
class DualHeadMambaNet(nn.Module):
    """
    双头检测模型：共享 backbone + P检测头 / S检测头 / P精定位头。

    backbone 与 PS_Detection.LightweightPWaveNet 逐层一致
    （LFEM1D → 池化 → proj → Mamba → 可选 Transformer → fusion），
    因此旧检测权重可直接加载到共享部分（见 load_detection_backbone）。

    pick_head 使用**转置卷积**上采样（T/4 → T/2 → T）而非线性插值，
    避免插值把高斯峰抹平（线性插值会损失定位锐度）。

    参数
    ----
    input_dim      : 输入通道数（三分量=3）
    hidden_dim     : 隐层维度
    use_transformer: True → Mamba + TransformerEncoder 并联后融合
    use_lfem       : True → 使用 LFEM1D 特征提取器
    mamba_mode     : 'none' / 'unidirectional' / 'bidirectional'
    pool_factor    : 特征时间下采样倍数（默认 4）
    """

    def __init__(self, input_dim=3, hidden_dim=48,
                 use_transformer=False, use_lfem=True,
                 mamba_mode='bidirectional',
                 use_cross_channel_attention=False,
                 pool_factor=POOL_FACTOR,
                 grad_checkpoint=False):
        super().__init__()
        self.pool_factor = pool_factor
        # 梯度检查点：把 Transformer 激活从 O(B·nhead·T²) 降到 O(B·nhead·T)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.use_lfem = use_lfem
        self.use_transformer = use_transformer
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

        # 特征投影层
        proj_dim = 96
        self.proj = nn.Conv1d(input_dim, proj_dim, kernel_size=1)

        # Mamba：三种模式
        if mamba_mode == 'bidirectional':
            self.mamba = BidirectionalMamba(proj_dim, d_state=16, d_conv=4,
                                            expand=2, num_layers=3, dropout=0.1)
            mamba_out_dim = proj_dim * 2
        elif mamba_mode == 'unidirectional':
            from mamba_ssm import Mamba
            self.mamba = nn.Sequential(
                Mamba(d_model=proj_dim, d_state=16, d_conv=4, expand=2),
                Mamba(d_model=proj_dim, d_state=16, d_conv=4, expand=2),
                Mamba(d_model=proj_dim, d_state=16, d_conv=4, expand=2)
            )
            mamba_out_dim = proj_dim
        elif mamba_mode == 'none':
            self.mamba = nn.Identity()
            mamba_out_dim = proj_dim
        else:
            raise ValueError(f"Invalid mamba_mode: {mamba_mode}")

        # Transformer 分支（可选）
        transformer_dim = hidden_dim * 2
        if use_transformer:
            self.transformer = TransformerEncoder(
                proj_dim, transformer_dim, num_layers=2, nhead=4, dropout=0.1)
            fusion_in_dim = mamba_out_dim + transformer_dim
        else:
            self.transformer = None
            fusion_in_dim = mamba_out_dim

        # 融合输出
        fused_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        def _det_head():
            return nn.Sequential(
                nn.Linear(fused_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

        # ── P 波检测头（固定宽度矩形标签，BCEWithLogitsLoss）────────────
        self.p_det_head = _det_head()

        # ── S 波检测头（矩形标签，BCEWithLogitsLoss）────────────────────
        self.s_det_head = _det_head()

        # ── P 波精定位头（高斯软标签，KL + soft-argmax）─────────────────
        self.pick_head = _det_head()

        # 转置卷积上采样：T/4 → T/2 → T，每级后接细化卷积
        self.pick_up1 = nn.ConvTranspose1d(1, 8, kernel_size=4, stride=2,
                                           padding=1)
        self.pick_up2 = nn.ConvTranspose1d(8, 8, kernel_size=4, stride=2,
                                           padding=1)
        self.pick_refine = nn.Conv1d(8, 1, kernel_size=5, padding=2)

    def forward(self, x):
        """
        返回 (p_det, s_det, p_pick)，均为 (batch, seq_len) logits：
          p_det  : → BCEWithLogitsLoss（P 检测）
          s_det  : → BCEWithLogitsLoss（S 检测）
          p_pick : → GaussianPickLoss（P 精定位，损失内部做 softmax）
        """
        batch_size, seq_len, _ = x.shape

        x = self.cross_ch_attn(x)
        x = x.transpose(1, 2)                # (batch, features, time)

        # LFEM1D 特征提取
        x = self.lfem_block1(x)              # (batch, input_dim, time)
        x = self.pool1(x)                    # (batch, input_dim, time/2)
        x = self.lfem_block2(x)              # (batch, input_dim, time/2)
        x = self.pool2(x)                    # (batch, input_dim, time/4)
        x = self.lfem_block3(x)              # (batch, input_dim, time/4)
        x = self.proj(x)                     # (batch, 96, time/4)
        x = x.transpose(1, 2)                # (batch, time', 96)

        # Mamba
        mamba_out = self.mamba(x)            # (batch, time', hidden*2 或 hidden)

        # Transformer（可选）
        if self.use_transformer:
            if self.grad_checkpoint and self.training:
                # 把整个 Transformer 分支交给检查点：前向不存中间激活，
                # 反向时重算。use_reentrant=True 以兼容 nn.MultiheadAttention
                # （它拒绝 use_reentrant=False）；重算会先走一次 no_grad，
                # Mamba 只需一阶梯度，因此不受影响。
                try:
                    transformer_out = checkpoint(self.transformer, x,
                                                 use_reentrant=True)
                except Exception as exc:
                    print(f"  [warn] grad_checkpoint 不可用，已关闭: {exc}")
                    self.grad_checkpoint = False
                    transformer_out = self.transformer(x)
            else:
                transformer_out = self.transformer(x)
            combined = torch.cat([mamba_out, transformer_out], dim=-1)
        else:
            combined = mamba_out

        # 融合 → (batch, time', fused_dim)
        fused = self.fusion(combined)

        def _interp(logits):
            """将 time' 插值回 seq_len（检测头用）。"""
            return F.interpolate(
                logits.unsqueeze(1), size=seq_len,
                mode='linear', align_corners=False
            ).squeeze(1)

        # P / S 检测头
        p_det = _interp(self.p_det_head(fused).squeeze(-1))
        s_det = _interp(self.s_det_head(fused).squeeze(-1))

        # 精定位头：转置卷积上采样，保留峰形
        h = self.pick_head(fused).transpose(1, 2)     # (batch, 1, time')
        h = F.relu(self.pick_up1(h))
        h = F.relu(self.pick_up2(h))
        p_pick = self.pick_refine(h).squeeze(1)       # (batch, seq_len)
        if p_pick.shape[-1] != seq_len:               # 兼容非整除长度
            p_pick = F.interpolate(p_pick.unsqueeze(1), size=seq_len,
                                   mode='linear', align_corners=False).squeeze(1)

        return p_det, s_det, p_pick

    def load_detection_backbone(self, state_dict, verbose=True):
        """
        从旧检测 checkpoint（PS_Detection 训练产物）加载共享部分。

        只加载键名与形状都匹配的张量（backbone + p_det_head + s_det_head），
        pick_head / pick_up* / pick_refine 保持随机初始化。

        返回 (loaded_keys, skipped_keys)
        """
        own = self.state_dict()
        loaded, skipped = {}, []
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                loaded[k] = v
            else:
                skipped.append(k)
        own.update(loaded)
        self.load_state_dict(own, strict=True)
        if verbose:
            n_pick = sum(1 for k in own if k.startswith('pick'))
            print(f"  从旧检测权重载入 {len(loaded)} 个张量，跳过 {len(skipped)} 个；"
                  f"pick 相关 {n_pick} 个张量随机初始化")
        return list(loaded.keys()), skipped


# =============== 损失函数 =================
class GaussianPickLoss(nn.Module):
    """
    P 波精定位损失（高斯软标签专用）。

    组合三项（与 BiMamba_PS_3C_LFEM_gauss.GaussianLoss 一致）：
    1. KL 散度：预测概率分布 vs 目标高斯分布
    2. 软 argmax L1：直接约束预测期望位置，提供强梯度信号
    3. 峰值区域 Focal MSE：对目标高斯值 > 阈值的位置加权 MSE，聚焦峰形

    输入:
        pred   : (B, T) — 原始 logits（内部做 softmax）
        target : (B, T) — 归一化为概率分布的高斯软标签
        valid  : (B,)   — 1=参与损失（信号样本），0=跳过（噪声样本）

    参数
    ----
    kl_weight          : KL 散度项权重（默认 1.0）
    soft_argmax_weight : 软 argmax L1 项权重（默认 5.0）
    focal_weight       : 峰值 MSE 项权重（默认 2.0）
    focal_threshold    : 峰值 mask 阈值（默认 0.1）
    """

    def __init__(self, kl_weight=1.0, soft_argmax_weight=5.0,
                 focal_weight=2.0, focal_threshold=0.1):
        super().__init__()
        self.kl_weight          = kl_weight
        self.soft_argmax_weight = soft_argmax_weight
        self.focal_weight       = focal_weight
        self.focal_threshold    = focal_threshold
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target, valid=None):
        if valid is not None:
            mask = valid.bool()
            if not mask.any():
                return pred.sum() * 0.0
            pred, target = pred[mask], target[mask]

        pred   = pred.float()
        target = target.float()
        if not torch.isfinite(pred).all():
            return pred.sum() * 0.0

        pred_prob = F.softmax(pred, dim=1)
        log_pred  = F.log_softmax(pred, dim=1)
        safe_target = target.clamp(min=0)
        safe_target = safe_target / (safe_target.sum(dim=1, keepdim=True) + 1e-8)

        kl = self.kl_loss(log_pred, safe_target)

        T = pred.shape[1]
        positions = torch.arange(T, device=pred.device, dtype=pred.dtype) / (T - 1)
        pred_pos  = (pred_prob * positions.unsqueeze(0)).sum(dim=1)
        tgt_pos   = (safe_target * positions.unsqueeze(0)).sum(dim=1)
        soft_l1   = F.l1_loss(pred_pos, tgt_pos)

        peak_mask = (safe_target > self.focal_threshold).float()
        focal_mse = (peak_mask * (pred_prob - safe_target) ** 2).sum(dim=1).mean()

        loss = (self.kl_weight          * kl
                + self.soft_argmax_weight * soft_l1
                + self.focal_weight       * focal_mse)
        return loss if torch.isfinite(loss) else soft_l1 * self.soft_argmax_weight


class DualHeadLoss(nn.Module):
    """
    双头联合损失（P检测 + S检测 + 噪声抑制 + P精定位）。

    组合四项：
    1. P 检测 BCE 损失（信号样本）
    2. S 检测 BCE 损失（有效S标签的信号样本）
    3. 噪声抑制损失（噪声样本上压制检测激活）
    4. P 精定位损失（GaussianPickLoss，仅信号样本）

    参数
    ----
    pick_criterion : GaussianPickLoss 实例
    p_weight       : P 波检测损失权重（默认 1.0）
    s_weight       : S 波检测损失权重（默认 0.6）
    noise_weight   : 噪声抑制损失权重（默认 0.5）
    pick_weight    : 精定位损失权重（默认 1.0）
    pos_weight     : 矩形 BCE 正类权重（窄矩形标签下类别不平衡，可调大）
    """

    def __init__(self, pick_criterion, p_weight=1.0, s_weight=0.6,
                 noise_weight=0.5, pick_weight=1.0, pos_weight=1.0):
        super().__init__()
        self.pick_criterion = pick_criterion
        self.p_weight       = p_weight
        self.s_weight       = s_weight
        self.noise_weight   = noise_weight
        self.pick_weight    = pick_weight
        self.register_buffer('pos_weight_t', torch.tensor([float(pos_weight)]))

    def forward(self, p_det, s_det, p_pick,
                p_rect, s_rect, p_gauss, s_valid, is_noise):
        """
        参数
        ----
        p_det   : (B, T) logits — P 检测头
        s_det   : (B, T) logits — S 检测头
        p_pick  : (B, T) logits — P 精定位头
        p_rect  : (B, T) P相位矩形标签
        s_rect  : (B, T) S相位矩形标签
        p_gauss : (B, T) P高斯软标签
        s_valid : (B,)   1=有效 S，0=无效
        is_noise: (B,)   1=噪声，0=信号

        返回
        ----
        total_loss, (p_det_loss, s_det_loss, noise_loss, pick_loss)
        """
        signal_mask = ~is_noise.bool()
        noise_mask  = is_noise.bool()
        pw = self.pos_weight_t.to(p_det.device)

        # ── P 波检测损失（仅信号样本）────────────────────────────────
        if signal_mask.any():
            p_det_loss = F.binary_cross_entropy_with_logits(
                p_det[signal_mask], p_rect[signal_mask], pos_weight=pw)
        else:
            p_det_loss = p_det.sum() * 0.0

        # ── S 波检测损失（仅有效S标签的信号样本）──────────────────────
        s_signal_mask = signal_mask & s_valid.bool()
        if s_signal_mask.any():
            s_det_loss = F.binary_cross_entropy_with_logits(
                s_det[s_signal_mask], s_rect[s_signal_mask], pos_weight=pw)
        else:
            s_det_loss = s_det.sum() * 0.0

        # ── 噪声样本：检测头抑制（期望输出全为0）─────────────────────
        if noise_mask.any():
            noise_zeros = torch.zeros_like(p_det[noise_mask])
            p_noise_sup = F.binary_cross_entropy_with_logits(
                p_det[noise_mask], noise_zeros)
            s_noise_sup = F.binary_cross_entropy_with_logits(
                s_det[noise_mask], noise_zeros)
            noise_loss = 0.5 * (p_noise_sup + s_noise_sup)
        else:
            noise_loss = p_det.sum() * 0.0

        # ── P 精定位损失（仅信号样本）────────────────────────────────
        pick_loss = self.pick_criterion(p_pick, p_gauss, signal_mask.float())

        total_loss = (self.p_weight     * p_det_loss
                      + self.s_weight   * s_det_loss
                      + self.noise_weight * noise_loss
                      + self.pick_weight  * pick_loss)
        return total_loss, (p_det_loss, s_det_loss, noise_loss, pick_loss)


# =================== 数据加载 ===================
def _find_hdf5_dataset(f, name):
    """在已打开的 HDF5 中按多种常见路径查找 trace 数据集。"""
    for key in (name, 'data/' + str(name), 'waveforms/' + str(name)):
        if key in f and isinstance(f[key], h5py.Dataset):
            return f[key]
    return None


def _read_p_s_columns(df):
    """兼容多种 P/S 到时列名，返回 (p_col, s_col)。"""
    p_col = next((c for c in ('trace_P_arrival_sample', 'p_arrival_sample',
                              'P_arrival_sample') if c in df.columns), None)
    s_col = next((c for c in ('trace_S_arrival_sample', 's_arrival_sample',
                              'S_arrival_sample') if c in df.columns), None)
    return p_col, s_col


def _take_window(data, p, s, window_size, rng, off_min, off_max,
                 mode='target_offset', p_min_lead=100, p_max_lead=5000):
    """
    取一个 window_size 窗口，返回 (win, p_in, s_in)（窗内采样点位置）。

    两种模式
    --------
    1) ``mode='target_offset'``（默认，波形长度 ≈ 窗口时唯一可行）
           pre   ~ U[off_min, off_max]，受 p >= pre 约束
           start = p − pre
           p_in  = pre                        # P 在窗内位置即目标 offset
       适用于 STEAD chunk（T=W=6000）。缺点：P 的位置分布被压得很窄，
       训练出的模型会带位置先验。

    2) ``mode='random_start'``（波形远长于窗口时推荐，如 INSTANCE T=12000）
           start ~ U[0, T − window_size]
           要求 P 的前导长度 p−start ∈ [p_min_lead, p_max_lead]，
           以排除 P 贴边（训练学不到）或 S 被切出窗外的样本。
       P 在窗内的位置因此自然散布，可覆盖几乎整个窗口 → 训练出位置无关的模型。

    约束（两种模式共用）：
        start = p − pre >= 0        →  pre <= p
        start + window_size <= T    →  pre >= p + window_size − T
    """
    T = data.shape[0]
    if T < window_size:
        return None, None, None

    if mode == 'random_start':
        lo_lead = max(int(p_min_lead), p + window_size - T, 0)
        hi_lead = min(int(p_max_lead), int(p))
        max_start = T - window_size
        if hi_lead < lo_lead:
            return None, None, None
        # 在合法前导区间内取随机前导；等价于随机窗口起点
        pre = int(rng.integers(lo_lead, hi_lead + 1))
        start = p - pre
        if start < 0 or start > max_start:
            return None, None, None
    else:
        lo = max(int(off_min), p + window_size - T, 0)
        hi = min(int(off_max), int(p))
        if hi < lo:
            return None, None, None
        pre = int(rng.integers(lo, hi + 1))
        start = p - pre
        if start < 0 or start + window_size > T:
            return None, None, None

    win = data[start:start + window_size]
    if win.shape[0] < window_size:
        return None, None, None
    return win, float(p - start), (float(s - start) if s > 0 else -1.0)


def estimate_ram_gb(n_samples, window_size, n_ch=3, copies=1.0):
    """粗略估计驻留 n_samples 条 (window_size, n_ch) float32 窗口需要多少 GB。

    ``copies`` 是额外副本份数。预分配实现下增强过程的峰值份数是
    ``(2 + reps) * M``（原数组 M + 新数组 (1+reps)M），见调用处。
    """
    return n_samples * window_size * n_ch * 4 * copies / (1024 ** 3)


def available_ram_gb():
    """可用内存（GB）；拿不到返回 None。"""
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


def _place_at_offset(w, p_abs, off, W, pool=None, rng=None):
    """把波形 w（P 绝对位置 p_abs）放进长度 W 的窗口，使 P 落在窗内 off 处。

    * ``T > W``：直接滑窗取 ``[p_abs - off, +W)``；
    * ``T <= W``：无法滑窗，改为在窗内平移波形，空出来的部分用 ``pool`` 里的真实
      噪声填充（没给 pool 时退化为镜像复制并返回 ``filled=False`` 提示）。

    返回 ``(窗口, 窗内 P 位置, filled)``。
    """
    w = np.asarray(w, dtype=np.float32)
    T = w.shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)
    if T > W:
        st = int(np.clip(p_abs - off, 0, T - W))
        return w[st:st + W], float(p_abs - st), False

    d = int(off) - int(round(p_abs))
    x = np.zeros((W, w.shape[1]), dtype=np.float32)
    if d >= 0:
        n = min(W - d, T)
        if n > 0:
            x[d:d + n] = w[:n]
        gap, left = d, True
    else:
        m = min(W, T + d)
        if m > 0:
            x[:m] = w[-d:-d + m]
        gap, left = -d, False
    filled = False
    if gap > 0:
        if pool is not None and len(pool):
            seg = np.asarray(pool[int(rng.integers(len(pool)))], dtype=np.float32)
            f = (seg[:gap] if seg.shape[0] >= gap
                 else np.resize(seg, (gap, seg.shape[1])))
            filled = True
        else:
            src = (w[:gap][::-1] if left else w[m - d:][::-1]) \
                if (w[:gap].shape[0] if left else w[m - d:].shape[0]) else None
            f = (np.resize(src, (gap, w.shape[1])) if src is not None
                 else np.zeros((gap, w.shape[1]), dtype=np.float32))
        if left:
            x[:gap] = f
        else:
            x[W - gap:] = f
    return x, float(off), filled


def augment_offsets(seqs, p_in, s_in, noise_pool, reps=3,
                    off_min=150, off_max=None, rng=None):
    """把 P 在窗内的位置随机搬移，空出来的部分用**真实噪声**填充。

    STEAD 的 P 恒定在窗内 ~500-900（波形按 P 对齐切好，且 T=W=6000），取窗方式
    无法产生偏移多样性，模型因此只学会"P 在窗内 500~900 才有响应"。实测偏移一旦
    >1300，检测器触发率掉到 0、拾取器 MAE 从 6 涨到 2000+。

    这里合成任意偏移：
        want ~ U[off_min, off_max]      目标偏移（采样点）
        d    = want - p                 平移量，d>0 波形右移
        窗内空出来的 [0,d) 或 [W+d,W) 用噪声池里的真实噪声段填充
    S 位置同步平移；移出窗外则标为无效（-1）。

    参数
    ----
    seqs       : (M, W, 3) 已 preprocess_3c 的窗口
    p_in, s_in : (M,) 窗内采样点位置；噪声样本 p_in <= -1.5
    noise_pool : (K, W, 3) 真实噪声窗口（chunk1），用于填充
    reps       : 每条信号样本额外生成几份（0 = 关闭）

    返回 (新 seqs, 新 p_in, 新 s_in)，原样本保留在最前面。
    """
    W = int(seqs.shape[1])
    off_max = int(off_max if off_max is not None else W - 200)
    rng = rng if rng is not None else np.random.default_rng(0)
    p_in = np.asarray(p_in, float)
    s_in = np.asarray(s_in, float)
    keep = np.where(p_in >= 0)[0]              # 只对信号样本做
    if keep.size == 0 or reps <= 0:
        return seqs, p_in, s_in

    pool = None
    if noise_pool is not None and len(noise_pool) > 0:
        pool = np.asarray(noise_pool, dtype=np.float32)
        if pool.ndim != 3 or pool.shape[1] != W:
            pool = None

    def _fill(n, fallback):
        """取 n 个采样点做填充；优先真实噪声，退化时复制 fallback。"""
        if n <= 0:
            return None
        if pool is not None:
            seg = pool[int(rng.integers(len(pool)))]
            if seg.shape[0] >= n:
                st = int(rng.integers(0, seg.shape[0] - n + 1))
                return seg[st:st + n]
            return np.resize(seg, (n, seg.shape[1])).astype(np.float32)
        if fallback is not None and fallback.shape[0] > 0:
            return np.resize(fallback, (n, fallback.shape[1])).astype(np.float32)
        return np.zeros((n, seqs.shape[2]), dtype=np.float32)

    out_x, out_p, out_s = [], [], []
    for _ in range(int(reps)):
        wants = rng.integers(int(off_min), off_max + 1, size=keep.size)
        for k, want in zip(keep, wants):
            w = np.asarray(seqs[k], dtype=np.float32)
            p = float(p_in[k]); sv = float(s_in[k])
            d = int(want) - int(round(p))
            x = np.zeros((W, seqs.shape[2]), dtype=np.float32)
            if d >= 0:
                n = min(W - d, w.shape[0])
                if n > 0:
                    x[d:d + n] = w[:n]
                fill = _fill(d, w[:d][::-1] if d <= w.shape[0] else None)
            else:
                m = min(W, w.shape[0] + d)
                if m > 0:
                    x[:m] = w[-d:-d + m]
                fill = _fill(-d, w[m - d:][::-1] if (m - d) < w.shape[0] else None)
            if fill is not None:
                if d >= 0:
                    x[:d] = fill
                else:
                    x[W + d:] = fill
            p_new = p + d
            if not (0 <= p_new < W):
                continue
            s_new = (sv + d) if sv >= 0 else -1.0
            if not (0 <= s_new < W) or s_new <= p_new:
                s_new = -1.0
            out_x.append(x); out_p.append(p_new); out_s.append(s_new)

    if not out_x:
        return seqs, p_in, s_in
    # 预分配 + 逐条写入：比 concatenate([seqs, stack(out_x)]) 少一份临时副本
    M = seqs.shape[0]
    new_x = np.empty((M + len(out_x), W, seqs.shape[2]), dtype=np.float32)
    new_x[:M] = seqs
    for j, arr in enumerate(out_x):
        new_x[M + j] = arr
    new_p = np.concatenate([p_in, np.asarray(out_p, float)])
    new_s = np.concatenate([s_in, np.asarray(out_s, float)])
    return new_x, new_p, new_s


def _offset_hist(p_in, window_size, bins=8):
    """把窗内 P 偏移打成直方图字符串，便于确认增强是否真的铺开了。"""
    p = np.asarray(p_in, float)
    p = p[p >= 0]
    if p.size == 0:
        return '(无信号样本)'
    edges = np.linspace(0, window_size, bins + 1).astype(int)
    cnt, _ = np.histogram(p, bins=edges)
    tot = max(cnt.sum(), 1)
    return '  '.join('%d-%d:%.1f%%' % (edges[i], edges[i + 1],
                                       cnt[i] / tot * 100)
                     for i in range(bins))


def load_dual_head_data(file_name, csv_file, window_size=WINDOW_SIZE,
                        max_samples=50000, seed=42,
                        off_min=100, off_max=1000,
                        window_mode="target_offset",
                        p_min_lead=100, p_max_lead=5000,
                        trace_names=None):
    """
    从 STEAD/chunk 格式 hdf5+csv 加载三分量波形及 P/S 到窗内位置。

    与 PS_Detection.load_lightweight_data 的差别：
      * 不做标准化（标准化在 Dataset 内按训练集拟合）；
      * 额外记录 P/S 在**取窗后**的采样点位置，供 offset 随机化与标签生成使用。

    返回:
        sequences : ndarray (M, window_size, 3)
        p_targets : ndarray (M,) 窗内 P 绝对采样点位置
        s_targets : ndarray (M,) 窗内 S 绝对采样点位置（无效为 -1）
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Original dataset size: {len(df)}")

    if trace_names is not None:
        # 事件级划分：只加载指定 trace（先划分事件、再抽样）
        want = [str(t) for t in trace_names]
        df['trace_name'] = df['trace_name'].astype(str)
        df = df.set_index('trace_name').reindex(want).dropna(
            how='all').reset_index()
        print(f"Event-split trace list: {len(df)} rows")
    elif len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"Sampled to {max_samples} samples (seed={seed})")

    p_col, s_col = _read_p_s_columns(df)
    if p_col is None:
        print(f"ERROR: CSV 缺少 P 波到时列，可用: {list(df.columns)}")
        return np.array([]), np.array([]), np.array([])
    df = df.copy()
    df['trace_name'] = df['trace_name'].astype(str)
    df[p_col] = pd.to_numeric(df[p_col], errors='coerce')
    if s_col:
        df[s_col] = pd.to_numeric(df[s_col], errors='coerce')
    df = df[df[p_col].notna() & (df[p_col] > 0)]

    rng = np.random.default_rng(seed)
    sequences, p_targets, s_targets = [], [], []
    error_count = 0

    print(f"Loading HDF5 file: {file_name}")
    try:
        with h5py.File(file_name, 'r') as f:
            print(f"HDF5 keys: {list(f.keys())}")
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
                try:
                    dset = _find_hdf5_dataset(f, row['trace_name'])
                    if dset is None:
                        error_count += 1
                        continue
                    data = np.array(dset)
                    if data.ndim != 2:
                        error_count += 1
                        continue
                    if data.shape[0] == 3 and data.shape[1] != 3:
                        data = data.T
                    if data.shape[1] < 3 or data.shape[0] < window_size:
                        error_count += 1
                        continue
                    data_3c = data[:, CHANNEL_INDICES]

                    p_arrival = int(row[p_col])
                    if p_arrival <= 0 or p_arrival >= data_3c.shape[0]:
                        error_count += 1
                        continue
                    s_arrival = (int(row[s_col])
                                 if (s_col and pd.notna(row[s_col])) else -1)

                    win, p_in, s_in = _take_window(
                        data_3c, p_arrival, s_arrival, window_size, rng,
                        off_min, off_max, window_mode,
                        p_min_lead, p_max_lead)
                    if win is None:
                        error_count += 1
                        continue
                    win = preprocess_3c(win, fs=FS)
                    if np.isnan(win).any() or np.isinf(win).any():
                        error_count += 1
                        continue

                    sequences.append(win)
                    p_targets.append(p_in)
                    s_targets.append(s_in)
                except Exception:
                    error_count += 1
                    continue
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        return np.array([]), np.array([]), np.array([])

    print(f"[{timestamp}] Successfully loaded {len(sequences)} samples, "
          f"errors: {error_count}")
    if len(sequences) == 0:
        print("ERROR: No valid sequences found!")
        return np.array([]), np.array([]), np.array([])

    sequences = np.array(sequences, dtype=np.float32)
    p_targets = np.array(p_targets, dtype=np.float32)
    s_targets = np.array(s_targets, dtype=np.float32)
    print(f"Final dataset: {len(sequences)} sequences, shape: {sequences.shape}")
    print(f"P-in-window range: {p_targets.min():.0f} – {p_targets.max():.0f} 采样点")
    s_valid_mask = s_targets >= 0
    if s_valid_mask.any():
        print(f"S-in-window range: {s_targets[s_valid_mask].min():.0f} – "
              f"{s_targets[s_valid_mask].max():.0f} 采样点 "
              f"({s_valid_mask.sum()} valid / {len(s_targets)} total)")
    else:
        print("S-target: no valid S arrivals found in this file")
    return sequences, p_targets, s_targets


def load_noise_data(file_name, csv_file=None, window_size=WINDOW_SIZE,
                    max_samples=10000, seed=42):
    """
    加载纯噪声波形（无 P/S 到时标注）。

    P/S 位置全部填 -2.0（Dataset 内按噪声处理：标签全零 + 检测头抑制）。

    返回:
        sequences : ndarray (M, window_size, 3)
        p_targets : ndarray (M,) 全为 -2.0
        s_targets : ndarray (M,) 全为 -2.0
    """
    print(f"Loading noise HDF5: {file_name}")
    names = []
    if csv_file and os.path.exists(csv_file):
        ndf = pd.read_csv(csv_file)
        if 'trace_category' in ndf.columns:
            names = ndf[ndf['trace_category'].astype(str).str.contains(
                'noise', case=False, na=False)]['trace_name'].astype(str).tolist()
        if not names and 'trace_name' in ndf.columns:
            names = ndf['trace_name'].astype(str).tolist()

    rng = np.random.default_rng(seed)
    sequences = []
    with h5py.File(file_name, 'r') as f:
        if not names:
            names = (sorted(f['data'].keys()) if 'data' in f
                     else sorted(k for k in f if isinstance(f[k], h5py.Dataset)))
        if max_samples and len(names) > max_samples:
            names = list(rng.choice(names, max_samples, replace=False))
        for name in tqdm(names, desc="Loading noise"):
            try:
                dset = _find_hdf5_dataset(f, name)
                if dset is None:
                    continue
                data = np.array(dset)
                if data.ndim != 2:
                    continue
                if data.shape[0] == 3 and data.shape[1] != 3:
                    data = data.T
                if data.shape[1] < 3 or data.shape[0] < window_size:
                    continue
                data_3c = data[:, CHANNEL_INDICES]
                start = int(rng.integers(0, data_3c.shape[0] - window_size + 1))
                win = preprocess_3c(data_3c[start:start + window_size], fs=FS)
                if np.isnan(win).any() or np.isinf(win).any():
                    continue
                sequences.append(win)
            except Exception:
                continue

    print(f"Noise samples loaded: {len(sequences)}")
    if not sequences:
        return np.array([]), np.array([]), np.array([])
    sequences = np.array(sequences, dtype=np.float32)
    return (sequences,
            np.full(len(sequences), -2.0, dtype=np.float32),
            np.full(len(sequences), -2.0, dtype=np.float32))


# =================== 训练 ===================
def train_dual_head_model(model, train_loader, val_loader, device,
                          num_epochs=NUM_EPOCHS, lr=1e-3,
                          p_weight=1.0, s_weight=0.6, noise_weight=0.5,
                          pick_weight=1.0, pos_weight=1.0,
                          kl_weight=1.0, soft_argmax_weight=5.0,
                          focal_weight=2.0, save_path=None,
                          save_payload=None, use_amp=False,
                          accum_steps=1, report_memory=True,
                          pos_check_waves=None, pos_check_p=None,
                          pos_check_noise=None, pos_check_pool=None,
                          pos_check_interval=5):
    """
    训练双头检测模型（P检测 + S检测 + P精定位）。

    参数
    ----
    p_weight     : P 波检测损失权重（默认 1.0）
    s_weight     : S 波检测损失权重（默认 0.6）
    noise_weight : 噪声抑制损失权重（默认 0.5）
    pick_weight  : P 精定位损失权重（默认 1.0）
    pos_weight   : 矩形 BCE 正类权重（窄矩形标签下建议 >1）
    use_amp      : 是否使用混合精度（默认 False，Mamba fp16 不稳定）
    save_path    : 最优权重保存路径（完整 checkpoint，含 config/flags/scaler）
    save_payload : 写入 checkpoint 的附加字段

    返回
    ----
    history : list[dict]，逐 epoch 的损失与验证指标
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    milestones = [max(1, int(num_epochs * f)) for f in (0.5, 0.75, 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.3)

    criterion = DualHeadLoss(
        pick_criterion=GaussianPickLoss(
            kl_weight=kl_weight, soft_argmax_weight=soft_argmax_weight,
            focal_weight=focal_weight),
        p_weight=p_weight, s_weight=s_weight, noise_weight=noise_weight,
        pick_weight=pick_weight, pos_weight=pos_weight,
    ).to(device)
    print(f"DualHeadLoss — p_w={p_weight:.2f}, s_w={s_weight:.2f}, "
          f"noise_w={noise_weight:.2f}, pick_w={pick_weight:.2f}, "
          f"pos_weight={pos_weight:.2f}")

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print("AMP mixed precision: ON (float16)" if use_amp else "AMP: OFF (fp32)")
    accum_steps = max(1, int(accum_steps))
    eff_batch = getattr(train_loader, 'batch_size', 0) * accum_steps
    print(f"Gradient accumulation: {accum_steps} 步 → 等效 batch "
          f"{eff_batch}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    best_metric = float('inf')       # 主指标：精定位 MAE（越小越好）
    best_p_f1 = -1.0
    history = []
    nan_batches_total = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        loss_parts = np.zeros(4)
        nan_batches = 0

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            x, p_rect, s_rect, p_gauss, s_valid, is_noise = batch
            x        = x.to(device, non_blocking=True)
            p_rect   = p_rect.to(device, non_blocking=True)
            s_rect   = s_rect.to(device, non_blocking=True)
            p_gauss  = p_gauss.to(device, non_blocking=True)
            s_valid  = s_valid.to(device, non_blocking=True)
            is_noise = is_noise.to(device, non_blocking=True)

            if torch.isnan(x).any() or torch.isnan(p_rect).any():
                nan_batches += 1
                continue


            with torch.cuda.amp.autocast(enabled=use_amp):
                p_det, s_det, p_pick = model(x)

            # loss 在 fp32 下计算，避免 BCEWithLogits / KLDiv 在 fp16 下数值不稳定
            total_loss, parts = criterion(
                p_det.float(), s_det.float(), p_pick.float(),
                p_rect, s_rect, p_gauss, s_valid, is_noise)

            if not torch.isfinite(total_loss):
                nan_batches += 1
                continue

            # 梯度累积：按 accum_steps 缩放 loss，使等效 batch = batch_size*accum
            scaled_loss = total_loss / accum_steps
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_last = (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader)
            if is_last:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += total_loss.item()
            loss_parts += np.array([float(p.detach()) for p in parts])
            del total_loss, parts, p_det, s_det, p_pick

        scheduler.step()
        nan_batches_total += nan_batches
        if report_memory and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
            print(f"  [显存] epoch {epoch} 峰值 allocated={peak:.2f} GiB, "
                  f"reserved={reserved:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_parts = loss_parts / max(1, len(train_loader))

        # 验证
        val = evaluate_dual_head(model, val_loader, device)

        # ── P 位置先验自检（每 5 个 epoch 或最后一个 epoch）───────────
        rob = {}
        if pos_check_interval and (
                epoch % pos_check_interval == 0
                or epoch == num_epochs - 1):
            rob = evaluate_offset_robustness(
                model, pos_check_waves, pos_check_p, pos_check_noise, device,
                offsets=POS_CHECK_OFFSETS, noise_pool=pos_check_pool)
            _print_offset_robustness(rob, tag=f'epoch {epoch}')

        # 主指标：精定位 MAE（本次改动的目标）；同时记录 P/S 检出 F1
        metric = val['pick_mae'] if val['pick_mae'] is not None else float('inf')
        if metric < best_metric:
            best_metric = metric
            best_p_f1 = val['p_f1']
            if save_path:
                payload = dict(save_payload or {})
                payload.update({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'threshold': 0.5,
                    'val_metrics': val,
                })
                torch.save(payload, save_path)

        history.append({
            'epoch': epoch, 'loss': avg_train_loss,
            'p_det_loss': float(avg_parts[0]), 's_det_loss': float(avg_parts[1]),
            'noise_loss': float(avg_parts[2]), 'pick_loss': float(avg_parts[3]),
            'val': val,
            'offset_robustness': rob,
        })

        epoch_time     = time.time() - epoch_start
        remaining_time = epoch_time * (num_epochs - epoch - 1)
        hrs, rem       = divmod(remaining_time, 3600)
        mins, secs     = divmod(rem, 60)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            mae_str = f"{metric:.2f}" if np.isfinite(metric) else "n/a"
            med_str = (f"{val['pick_median']:.1f}"
                       if val['pick_median'] is not None else "n/a")
            print(f"Epoch {epoch:02d}: Loss={avg_train_loss:.6f} "
                  f"(p_det={avg_parts[0]:.4f} s_det={avg_parts[1]:.4f} "
                  f"noise={avg_parts[2]:.4f} pick={avg_parts[3]:.4f}), "
                  f"P_F1={val['p_f1']:.2f}%, pick_MAE={mae_str} "
                  f"(中位 {med_str}), Best={best_metric:.2f}, "
                  f"ETA: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}")

    if nan_batches_total > 0:
        print(f"  Warning: 共跳过 {nan_batches_total} 个 NaN batch")
    print(f"训练结束 — 最优精定位 MAE={best_metric:.2f} (P_F1={best_p_f1:.2f}%)")
    return history


# =================== 评估 ===================
def _edge_decode(p_curve, threshold=0.5):
    """检测头前缘解码：峰值所在连续 >= 阈值区段的起点。"""
    above = p_curve >= threshold
    if not above.any():
        return int(np.argmax(p_curve)), False
    peak = int(np.argmax(p_curve))
    if not above[peak]:
        return peak, False
    lo = peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    return int(lo), True


def evaluate_offset_robustness(model, waves, p_in_win, is_noise, device,
                               offsets=(200, 550, 1000, 1600, 2400, 3200),
                               max_traces=64, window_size=WINDOW_SIZE,
                               threshold=0.5, noise_pool=None, seed=0):
    """
    P 位置先验自检：在**原始波形**上按多个合成 offset 重新取窗，评估定位 MAE。

    为什么要单独做这件事
    --------------------
    训练/验证集的 offset 分布与目标域不一致时，常规验证指标会虚高（实测
    P F1 99.94% 但换数据集后触发率仅 30%）。本函数用完全相同的模型，
    只改变「P 落在窗内哪个位置」，因此能直接量化位置依赖程度：
    若各 offset 的 MAE / 触发率差异很大，说明模型学的是位置而非波形内容。

    参数
    ----
    waves     : (N, T, F) 原始波形（**未标准化**；本函数内部自己预处理+标准化）
    p_in_win  : (N,) 该窗口内 P 的采样点位置（用于反推窗口在波形中的起点）
    is_noise  : (N,) 1=噪声（跳过）
    offsets   : 依次测试的目标 offset（采样点）

    返回
    ----
    dict: {offset: {'mae':..., 'median':..., 'within_20':..., 'trigger_rate':...}}
    """
    from PS_Detection_evsplit import CHANNEL_INDICES
    scaler = getattr(model, '_scaler_for_check', None)
    if scaler is None:
        return {}

    sel = [i for i in range(len(waves))
           if not bool(is_noise[i]) and np.isfinite(p_in_win[i]) and p_in_win[i] > 0]
    if len(sel) > max_traces:
        sel = sel[::max(1, len(sel) // max_traces)][:max_traces]
    if not sel:
        return {}

    out = {}
    model.eval()
    # T == window_size（STEAD）时无法滑窗，必须靠平移+噪声填充合成偏移，
    # 否则 st 恒被 clip 到 0，各 offset 会得到完全相同的数字（本函数曾经就
    # 是这样静默失效的）。
    T0 = int(np.asarray(waves[sel[0]]).shape[0])
    synth = T0 <= window_size
    if synth and (noise_pool is None or len(noise_pool) == 0):
        print('  [位置先验自检] ⚠ 波形长度 ≤ 窗口长度且没有噪声池，填充退化为'
              '镜像复制，结果仅供参考')
    rng = np.random.default_rng(seed)
    _warned = [False]
    for off in offsets:
        errs, trg = [], []
        for i in sel:
            p_abs = float(p_in_win[i])
            win, p_off, filled = _place_at_offset(
                waves[i], p_abs, off, window_size, pool=noise_pool, rng=rng)
            if win.shape[0] < window_size:
                continue
            if synth and not filled and not _warned[0]:
                _warned[0] = True
            pre = preprocess_3c(win[:, CHANNEL_INDICES], fs=FS)
            norm = scaler.transform(pre.reshape(-1, 3)).reshape(
                1, window_size, 3).astype(np.float32)
            x = torch.from_numpy(norm).to(device)
            with torch.no_grad():
                p_det, _, _ = model(x)
            prob = torch.sigmoid(p_det)[0].cpu().numpy()
            pos, trig = _edge_decode(prob, threshold)
            errs.append(abs(pos - p_off))
            trg.append(trig)
        if not errs:
            continue
        e = np.array(errs, dtype=float)
        out[int(off)] = {
            'n': len(errs),
            'mae': float(e.mean()),
            'median': float(np.median(e)),
            'within_20': float((e <= 20).mean() * 100),
            'trigger_rate': float(np.mean(trg) * 100),
        }
    return out


def _print_offset_robustness(rob, tag=''):
    if not rob:
        return
    print(f"  [位置先验自检{(' ' + tag) if tag else ''}] "
          f"（不同 offset 下 P 定位 MAE / 触发率，差异大 = 依赖位置）")
    for off, m in sorted(rob.items()):
        flag = '' if m['trigger_rate'] >= 80 else '  ← 明显掉下来'
        print(f"      offset={off:5d}: N={m['n']:3d} MAE={m['mae']:8.2f} "
              f"中位={m['median']:7.2f} ≤20={m['within_20']:5.1f}% "
              f"触发率={m['trigger_rate']:5.1f}%{flag}")


def evaluate_dual_head(model, loader, device, threshold=0.5):
    """
    验证/测试集评估：返回检出与定位指标字典。

    返回字段：
        p_f1 / p_precision / p_recall : P 检测（信号 vs 噪声）
        s_f1 / s_precision / s_recall : S 检测
        pick_mae / pick_median / pick_within_5 / pick_within_20 : 精定位头
        pick_softargmax_mae : 精定位头软 argmax 解码误差
        edge_mae / edge_median : 矩形检测头前缘解码误差（与 pick 头对照）
    """
    model.eval()
    all_p, all_s, all_k = [], [], []
    all_is_noise, all_s_valid, all_p_gauss = [], [], []

    with torch.no_grad():
        for batch in loader:
            x, p_rect, s_rect, p_gauss, s_valid, is_noise = batch
            p_det, s_det, p_pick = model(x.to(device))
            all_p.append(p_det.cpu())
            all_s.append(s_det.cpu())
            all_k.append(p_pick.cpu())
            all_p_gauss.append(p_gauss)
            all_is_noise.append(is_noise)
            all_s_valid.append(s_valid)

    p_det_all    = torch.cat(all_p, dim=0)
    s_det_all    = torch.cat(all_s, dim=0)
    p_pick_all   = torch.cat(all_k, dim=0)
    p_gauss_all  = torch.cat(all_p_gauss, dim=0)
    is_noise_all = torch.cat(all_is_noise, dim=0)
    s_valid_all  = torch.cat(all_s_valid, dim=0)

    # 检出混淆矩阵
    p_det_metrics, s_det_metrics = compute_detection_metrics(
        p_det_all, s_det_all, is_noise_all, s_valid_all, threshold=threshold)

    # 精定位与矩形前缘解码
    p_prob    = torch.sigmoid(p_det_all).numpy()
    pick_prob = torch.softmax(p_pick_all, dim=1).numpy()
    gauss     = p_gauss_all.numpy()
    is_noise_np = is_noise_all.numpy().astype(bool)

    pick_err, edge_err, pick_sa_err = [], [], []
    T = p_prob.shape[1]
    positions = np.arange(T) / (T - 1)
    for i in range(len(p_prob)):
        if is_noise_np[i]:
            continue
        gt = int(gauss[i].argmax())
        pick_err.append(abs(int(pick_prob[i].argmax()) - gt))
        pick_sa_err.append(abs(float((pick_prob[i] * positions).sum() * (T - 1))
                               - gt))
        above = p_prob[i] >= threshold
        if above.any():
            peak = int(p_prob[i].argmax())
            lo = peak
            while lo > 0 and above[lo - 1]:
                lo -= 1
            edge_err.append(abs(lo - gt))

    return {
        'p_f1': p_det_metrics['f1'],
        'p_precision': p_det_metrics['precision'],
        'p_recall': p_det_metrics['recall'],
        's_f1': s_det_metrics['f1'],
        's_precision': s_det_metrics['precision'],
        's_recall': s_det_metrics['recall'],
        'pick_mae': float(np.mean(pick_err)) if pick_err else None,
        'pick_median': float(np.median(pick_err)) if pick_err else None,
        'pick_within_5': float(np.mean(np.array(pick_err) <= 5) * 100)
        if pick_err else None,
        'pick_within_20': float(np.mean(np.array(pick_err) <= 20) * 100)
        if pick_err else None,
        'pick_softargmax_mae': float(np.mean(pick_sa_err)) if pick_sa_err else None,
        'edge_mae': float(np.mean(edge_err)) if edge_err else None,
        'edge_median': float(np.median(edge_err)) if edge_err else None,
    }


def compute_detection_metrics(p_det_logits, s_det_logits,
                              is_noise, s_valid, threshold=0.5):
    """
    由检测头 logits 计算 P/S 检出混淆矩阵指标。

    判定规则（与 PS_Detection.compute_detection_metrics 一致）：
      * 信号样本：任一时刻检测概率 >= threshold 判为检出（正类）
      * 噪声样本：期望完全不检出（负类）
      * S 指标仅统计有效 S 标签的信号样本

    返回 (p_metrics, s_metrics)，各含
    TP/TN/FP/FN/precision/recall/f1/accuracy/n_signal/n_noise
    """
    p_prob = torch.sigmoid(p_det_logits).numpy()
    s_prob = torch.sigmoid(s_det_logits).numpy()
    noise  = is_noise.numpy().astype(bool)
    sv     = s_valid.numpy() > 0.5
    signal = ~noise

    def _cm(gt_pos, pred_pos):
        gt_pos = np.asarray(gt_pos, dtype=int)
        pred_pos = np.asarray(pred_pos, dtype=int)
        TP = int(np.sum((gt_pos == 1) & (pred_pos == 1)))
        TN = int(np.sum((gt_pos == 0) & (pred_pos == 0)))
        FP = int(np.sum((gt_pos == 0) & (pred_pos == 1)))
        FN = int(np.sum((gt_pos == 1) & (pred_pos == 0)))
        prec = TP / (TP + FP + 1e-10) * 100
        rec  = TP / (TP + FN + 1e-10) * 100
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        acc  = (TP + TN) / (TP + TN + FP + FN + 1e-10) * 100
        return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'precision': prec,
                'recall': rec, 'f1': f1, 'accuracy': acc,
                'n_signal': int(gt_pos.sum()), 'n_noise': int((gt_pos == 0).sum())}

    # P：正类 = 全部信号样本
    p_max_signal = p_prob[signal].max(axis=1) if signal.any() else np.array([])
    p_max_noise  = p_prob[noise].max(axis=1) if noise.any() else np.array([])
    p_det_metrics = _cm(
        [1] * len(p_max_signal) + [0] * len(p_max_noise),
        list((p_max_signal >= threshold).astype(int))
        + list((p_max_noise >= threshold).astype(int)))

    # S：正类 = 有效 S 的信号样本
    s_mask = signal & sv
    s_max_signal = s_prob[s_mask].max(axis=1) if s_mask.any() else np.array([])
    s_det_metrics = _cm(
        [1] * len(s_max_signal) + [0] * len(p_max_noise),
        list((s_max_signal >= threshold).astype(int))
        + list((p_max_noise >= threshold).astype(int)))

    return p_det_metrics, s_det_metrics


# =================== 可视化 ===================
def print_detection_report(p_metrics, s_metrics):
    """打印 P/S 检出报告（与 PS_Detection.print_detection_report 风格一致）。"""
    for name, m in (('P-wave', p_metrics), ('S-wave', s_metrics)):
        print(f"\n  [{name} Detection]")
        print(f"    TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
        print(f"    Precision={m['precision']:.2f}%  Recall={m['recall']:.2f}%  "
              f"F1={m['f1']:.2f}%  Acc={m['accuracy']:.2f}%")
        print(f"    n_signal={m['n_signal']}  n_noise={m['n_noise']}")


def plot_confusion_matrix(p_metrics, s_metrics, save_path='confusion_matrix.png'):
    """绘制 P/S 检测混淆矩阵热力图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, met, label in zip(axes, [p_metrics, s_metrics], ['P', 'S']):
        matrix = np.array([[met['TP'], met['FN']],
                           [met['FP'], met['TN']]], dtype=float)
        im = ax.imshow(matrix, cmap='Blues')
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                val = int(matrix[i, j])
                pct = val / total * 100 if total > 0 else 0
                color = 'white' if matrix[i, j] > matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                        fontsize=15, fontweight='bold', color=color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred Pos', 'Pred Neg'], fontsize=11)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['True Pos', 'True Neg'], fontsize=11)
        ax.set_title(f'{label}  Prec={met["precision"]:.1f}%  '
                     f'Recall={met["recall"]:.1f}%  F1={met["f1"]:.1f}%',
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Dual-Head Detection Confusion Matrix', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\nConfusion matrix saved: {save_path}")


def plot_detection_example(model, dataset, device, idx=0,
                           save_path='detection_example.png',
                           window_size=WINDOW_SIZE):
    """
    绘制双头检测样例（5 行）：
      Row 0–2 : Z / N / E 三分量波形
      Row 3   : P/S 检测概率 vs 矩形标签
      Row 4   : P 精定位概率 vs 高斯标签（含 pick / gt 竖线与误差）
    """
    x, p_rect, s_rect, p_gauss, s_valid, is_noise = dataset[idx]
    x_tensor = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        p_det, s_det, p_pick = model(x_tensor)

    p_prob = torch.sigmoid(p_det).cpu().numpy().flatten()
    s_prob = torch.sigmoid(s_det).cpu().numpy().flatten()
    k_prob = torch.softmax(p_pick, dim=1).cpu().numpy().flatten()

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

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
    ax.plot(p_rect.numpy(), 'b--', lw=1, alpha=0.5, label='P rect label')
    ax.plot(s_rect.numpy(), 'r--', lw=1, alpha=0.5, label='S rect label')
    ax.axhline(0.5, color='gray', lw=0.5, linestyle=':')
    ax.set_ylabel('Detection Prob')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # 精定位概率
    ax = axes[4]
    k_norm = k_prob / max(k_prob.max(), 1e-9)
    g_np = p_gauss.numpy()
    g_norm = g_np / max(g_np.max(), 1e-9)
    ax.plot(k_norm, 'g-', lw=1.5, label='P pick prob (norm)')
    ax.plot(g_norm, 'k--', lw=1, alpha=0.6, label='P gauss label (norm)')
    ax.axvline(int(k_prob.argmax()), color='g', lw=1, ls=':',
               label=f'pick={int(k_prob.argmax())}')
    ax.axvline(int(g_np.argmax()), color='k', lw=1, ls=':',
               label=f'gt={int(g_np.argmax())}')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Pick Prob')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    noise_str = 'NOISE' if is_noise.item() > 0.5 else 'SIGNAL'
    extra = ''
    if is_noise.item() < 0.5:
        extra = (f", S_valid={s_valid.item():.0f}, "
                 f"pick err={int(k_prob.argmax()) - int(g_np.argmax()):+d}")
    fig.suptitle(f'Dual-Head Detection Example (idx={idx}, {noise_str}{extra})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detection example saved: {save_path}")


def plot_training_history(history, save_path='dual_head_results.png'):
    """绘制训练过程：总损失 / 各项损失 / 验证 F1 / 验证定位 MAE。"""
    ep = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(ep, [h['loss'] for h in history])
    axes[0, 0].set_title('Training Loss (total)')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, [h['p_det_loss'] for h in history], label='p_det')
    axes[0, 1].plot(ep, [h['s_det_loss'] for h in history], label='s_det')
    axes[0, 1].plot(ep, [h['noise_loss'] for h in history], label='noise')
    axes[0, 1].plot(ep, [h['pick_loss'] for h in history], label='pick')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ep, [h['val']['p_f1'] for h in history], label='P-F1')
    axes[1, 0].plot(ep, [h['val']['s_f1'] for h in history], label='S-F1')
    axes[1, 0].set_title('Validation F1')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('F1 (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    mae  = [h['val']['pick_mae'] if h['val']['pick_mae'] is not None else np.nan
            for h in history]
    edge = [h['val']['edge_mae'] if h['val']['edge_mae'] is not None else np.nan
            for h in history]
    axes[1, 1].plot(ep, mae, label='pick_head MAE')
    axes[1, 1].plot(ep, edge, label='rect edge MAE')
    axes[1, 1].set_title('P Localization Error (val)')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('MAE (samples)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved: {save_path}")


# =================== 辅助 ===================
def _resolve_file_pairs(here, items, defaults):
    """把 'a.hdf5,a.csv' / 'a.hdf5' / None 解析为 (hdf5, csv) 绝对路径列表。"""
    if not items:
        pairs = defaults
    else:
        pairs = []
        for it in items:
            parts = str(it).split(',')
            h5 = parts[0].strip()
            csv = parts[1].strip() if len(parts) > 1 else \
                os.path.splitext(h5)[0] + '.csv'
            pairs.append((h5, csv))
    out = []
    for h5, csv in pairs:
        out.append((h5 if os.path.isabs(h5) else os.path.join(here, h5),
                    csv if os.path.isabs(csv) else os.path.join(here, csv)))
    return out


def build_argparser():
    """命令行参数（默认值对齐 PS_Detection.main 的数据划分与超参）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description='双头检测模型训练（P检测 + S检测 + P精定位）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 数据
    ap.add_argument('--train-files', nargs='+', default=None,
                    help='训练 hdf5,csv 对，形如 chunk2.hdf5,chunk2.csv')
    ap.add_argument('--noise-files', nargs='+', default=None)
    ap.add_argument('--pool-files', nargs='+', default=None,
                    help='第二事件增强用的波形库')
    ap.add_argument('--max-samples', type=int, default=32000,
                    help='每个 chunk 的 train trace 上限；'
                         '32000×2 chunk×80%% ≈ 64k，与原文训练量等量')
    ap.add_argument('--val-samples', type=int, default=5000,
                    help='每个 chunk 的 val trace 上限（与拾取模型保持一致，'
                         '这样 val/test 用的是同一批 trace）')
    ap.add_argument('--test-samples', type=int, default=5000,
                    help='每个 chunk 的 test trace 上限（同上）')
    ap.add_argument('--noise-max-samples', type=int, default=0,
                    help='每个噪声 chunk 的显式加载上限；0 表示按 '
                         '--total-noise-samples 自动折算（推荐）')
    ap.add_argument('--total-noise-samples', type=int, default=20000,
                    help='噪声总量，按 --split-ratios 划分 '
                         '(默认 20000 -> 16000/3000/1000)')
    ap.add_argument('--val-fraction', type=float, default=0.2)
    ap.add_argument('--test-fraction', type=float, default=0.2)
    # 事件级划分（回应审稿人：split by event）
    ap.add_argument('--split-ratios', default='0.8,0.15,0.05',
                    help='事件级划分比例 train,val,test')
    ap.add_argument('--total-samples', type=int, default=100000,
                    help='信号总量，按 --split-ratios 拆分；'
                         '0 表示改用 --max-samples/--val-samples/'
                         '--test-samples')
    ap.add_argument('--split-table', default='',
                    help='复用已有事件划分表（与拾取模型共用同一划分时用）')
    ap.add_argument('--save-split-table', default='event_split_table.csv',
                    help='生成划分表时的保存路径')
    ap.add_argument('--pool-max-samples', type=int, default=30000,
                    help='波形库（第二事件增强）最多加载多少条；'
                         '30000 条约 2.2 GB 内存，显存/内存不足时调小')
    ap.add_argument('--dry-run', action='store_true',
                    help='只装配数据并打印划分/形状，不训练')

    # 训练
    ap.add_argument('--num-epochs', type=int, default=NUM_EPOCHS)
    ap.add_argument('--batch-size', type=int, default=64,
                    help='micro-batch 大小（显存不足时调小，配合 --accum-steps）')
    ap.add_argument('--accum-steps', type=int, default=8,
                    help='梯度累积步数，等效 batch = batch_size × accum_steps')
    ap.add_argument('--grad-checkpoint', action='store_true', default=False,
                    help='对 Transformer 逐层梯度检查点（省显存，慢约 30%%）')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--pos-check-interval', type=int, default=5,
                    help='每多少个 epoch 做一次 P 位置先验自检（0=关闭）')
    ap.add_argument('--use-amp', action='store_true',
                    help='混合精度（Mamba fp16 不稳定，默认关闭）')

    # 标签
    ap.add_argument('--rect-width', type=int, default=300,
                    help='P 矩形标签固定宽度（采样点）')
    ap.add_argument('--rect-adaptive', action='store_true',
                    help='改用 clip(1.4*(S-P), --rect-w-min, --rect-w-max)')
    ap.add_argument('--rect-w-min', type=int, default=200)
    ap.add_argument('--rect-w-max', type=int, default=800)
    ap.add_argument('--s-phase-scale', type=float, default=0.5)
    ap.add_argument('--pick-sigma', type=float, default=8.0)
    ap.add_argument('--window-mode', choices=['target_offset', 'random_start'],
                    default='target_offset',
                    help="取窗方式。'target_offset' 主动把 P 摆到 --p-offset-* "
                         "（STEAD T=W=6000 时唯一可行）；'random_start' 随机窗口"
                         "起点，让 P 在窗内自然散布（INSTANCE T=12000 等长波形"
                         "推荐，可训练位置无关的检测器）")
    ap.add_argument('--p-offset-min', type=int, default=100)
    ap.add_argument('--p-offset-max', type=int, default=1000)
    ap.add_argument('--offset-augment-reps', type=int, default=1,
                    help='偏移增强：每条信号样本额外合成几份任意偏移窗口'
                         '（0=关闭）。STEAD 的 T=W=6000 使取窗方式无法产生偏移'
                         '多样性，必须靠这个合成，否则模型只在窗内 500~900 有效')
    ap.add_argument('--offset-aug-min', type=int, default=150,
                    help='偏移增强的目标偏移下界（采样点）')
    ap.add_argument('--offset-aug-max', type=int, default=0,
                    help='偏移增强的目标偏移上界（0 = 窗口长度 - 200）')
    ap.add_argument('--p-min-lead', type=int, default=100,
                    help='random_start 模式下 P 前导长度下限（避免 P 贴窗口左边界）')
    ap.add_argument('--p-max-lead', type=int, default=5000,
                    help='random_start 模式下 P 前导长度上限')

    # 损失权重
    ap.add_argument('--p-weight', type=float, default=1.0)
    ap.add_argument('--s-weight', type=float, default=0.6)
    ap.add_argument('--noise-weight', type=float, default=0.5)
    ap.add_argument('--pick-weight', type=float, default=1.0)
    ap.add_argument('--pos-weight', type=float, default=1.0)
    ap.add_argument('--kl-weight', type=float, default=1.0)
    ap.add_argument('--soft-argmax-weight', type=float, default=5.0)
    ap.add_argument('--focal-weight', type=float, default=2.0)

    # 模型
    ap.add_argument('--hidden-dim', type=int, default=48)
    ap.add_argument('--mamba-mode', default='bidirectional',
                    choices=['none', 'unidirectional', 'bidirectional'])
    # Transformer 自注意力在 T/4=1500 上是 O(B·nhead·T²)：24GB 卡上
    # 建议 --no-transformer（或配 --grad-checkpoint + 小 batch），
    # 实测 Mamba 分支单独训练已可达 P F1≈98%。
    ap.add_argument('--use-transformer', dest='use_transformer',
                    action='store_true', default=False)
    ap.add_argument('--no-transformer', dest='use_transformer',
                    action='store_false')
    ap.add_argument('--no-lfem', dest='use_lfem', action='store_false',
                    default=True)
    ap.add_argument('--init-detection-ckpt', default=None,
                    help='用旧检测权重初始化共享 backbone + 检测头')

    ap.add_argument('--output-dir', default=os.path.join(here, 'dual_head_runs'))
    return ap


# =================== 主函数 ===================
def main(seed=42, argv=None):
    args = build_argparser().parse_args(argv)
    if argv is None and seed != 42:
        args.seed = seed
    seed = args.seed

    here = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark = True")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("双头检测模型 — 三分量版本（双向Mamba）")
    print(f"输入分量: Z(col{CH_Z}), N(col{CH_N}), E(col{CH_E})")
    print(f"Random seed: {seed}")
    print(f"输入窗口: {WINDOW_SIZE} 点 ({WINDOW_SIZE / FS:.0f}s @ {FS}Hz)")
    if args.rect_adaptive:
        print(f"P 矩形标签: clip({args.rect_w_min}, 1.4×(S-P), {args.rect_w_max})")
    else:
        print(f"P 矩形标签: 固定宽度 {args.rect_width} 点（不随 S-P 缩放）")
    print(f"P 精定位标签: 高斯 σ={args.pick_sigma} 点，")
    if args.window_mode == "random_start":
        print(f"取窗模式: random_start（随机窗口起点，P 前导 ∈ "
              f"[{args.p_min_lead}, {args.p_max_lead}]）→ 位置无关训练")
    else:
        print(f"取窗模式: target_offset（P 目标 offset ∈ "
              f"[{args.p_offset_min}, {args.p_offset_max}]）")
        if args.p_offset_max - args.p_offset_min < 2000:
            print("           ⚠ offset 区间偏窄易产生位置先验；若数据波形远长于"
                  "窗口，建议改用 --window-mode random_start")
    print("=" * 50)

    window_size = WINDOW_SIZE

    # ── 1. 载入训练 / 噪声 / 波形库（事件级划分：先分事件、再抽样）──────
    train_files = _resolve_file_pairs(here, args.train_files, _DEFAULT_TRAIN_FILES)
    noise_files = _resolve_file_pairs(here, args.noise_files, _DEFAULT_NOISE_FILES)
    pool_files  = _resolve_file_pairs(here, args.pool_files,  _DEFAULT_POOL_FILES)

    # chunk 名保留**相对脚本目录的子路径**（例如 'autodl-tmp/chunk2' 与 'chunk3'
    # 可以不在同一层目录），data_dir 统一取脚本目录；
    # 这样 os.path.join(data_dir, c + '.hdf5') 与 load_chunks(data_dir, [c]) 都能定位到。
    data_dir = here
    chunk_names = []
    for _h5, _csv in train_files:
        rel = os.path.relpath(os.path.abspath(_h5), here)
        if rel.endswith('.hdf5'):
            chunk_names.append(rel[:-5])
    if not chunk_names:
        raise SystemExit('无法从 --train-files 推断 chunk 名')
    print(f"[数据] data_dir={data_dir}")
    print(f"[数据] chunks={chunk_names}")

    ratios = tuple(float(x) for x in args.split_ratios.split(','))
    if args.total_samples > 0:
        _c = split_caps(args.total_samples, ratios, len(chunk_names))
        args.max_samples, args.val_samples, args.test_samples = (
            _c['train'], _c['val'], _c['test'])
        print("[事件级划分] 信号总量=%d 比例=%s -> 每个 chunk 上限 "
              "train=%d, val=%d, test=%d；合计=%d 条"
              % (args.total_samples, ratios, _c['train'], _c['val'],
                 _c['test'], _c['total']))
    reused = bool(args.split_table) and os.path.exists(args.split_table)
    if reused:
        print(f"使用已有事件划分表: {args.split_table}")
        meta = load_split_table(args.split_table)
    else:
        print(f"构建事件级划分（chunks={chunk_names}, ratios={ratios}）...")
        meta = build_split_table(load_chunks(data_dir, chunk_names), ratios, seed)
    print('\n' + describe_splits(meta))
    assert_disjoint(meta)
    exclude = events_in(meta, 'val') | events_in(meta, 'test')
    if args.save_split_table and not reused:
        save_split_table(meta, args.save_split_table)

    def _load_sig_split(split, max_samples):
        sq, pi, si = [], [], []
        for c in chunk_names:
            names = names_for_split(meta, split, c, max_samples, seed)
            if not names:
                continue
            h5 = os.path.join(data_dir, c + '.hdf5')
            cv = os.path.join(data_dir, c + '.csv')
            if not (os.path.exists(h5) and os.path.exists(cv)):
                print(f"[skip] 文件不存在: {h5}")
                continue
            print(f"\n[{split}] {c}: 指定 {len(names)} 条")
            a, b, c2 = load_dual_head_data(
                h5, cv, window_size=window_size, max_samples=max_samples,
                seed=seed, off_min=args.p_offset_min, off_max=args.p_offset_max,
                window_mode=args.window_mode, p_min_lead=args.p_min_lead,
                p_max_lead=args.p_max_lead, trace_names=names)
            if len(a) > 0:
                sq.append(a); pi.append(b); si.append(c2)
        if not sq:
            return (np.zeros((0, window_size, 3), np.float32),
                    np.zeros(0, np.float32), np.zeros(0, np.float32))
        return (np.concatenate(sq, 0), np.concatenate(pi, 0),
                np.concatenate(si, 0))

    print("\nLoading training data (event-level split)...")
    SIG = {'train': _load_sig_split('train', args.max_samples),
           'val':   _load_sig_split('val',   args.val_samples),
           'test':  _load_sig_split('test',  args.test_samples)}
    for sp in ('train', 'val', 'test'):
        print(f"  signal[{sp}] = {len(SIG[sp][0])} 条")

    # 噪声（chunk1）：无事件归属、与信号 trace 不重叠，仍按 trace 划分。
    # 总量 --total-noise-samples（默认 20000）按**与信号相同的比例** 80/15/5 划分
    # -> train 16000 / val 3000 / test 1000。
    print("\nLoading noise data...")
    n_noise = max(len(noise_files), 1)
    NC = split_caps(args.total_noise_samples, ratios, n_noise)
    noise_cap = (args.noise_max_samples if args.noise_max_samples > 0
                 else NC['train'] + NC['val'] + NC['test'])
    print(f"[噪声] 总量={args.total_noise_samples} 比例={ratios} -> 每个 chunk 上限 "
          f"{noise_cap}（train/val/test = {NC['train']}/{NC['val']}/{NC['test']}）；"
          f"合计 {NC['total']} 条")
    noise_pool_seqs = []
    noise_parts = {'train': [], 'val': [], 'test': []}
    for hdf5_file, csv_file in noise_files:
        if not (os.path.exists(hdf5_file) and os.path.exists(csv_file)):
            print(f"\n[skip] 文件不存在: {hdf5_file} / {csv_file}")
            continue
        print(f"\nLoading noise from: {hdf5_file} & {csv_file}")
        nseq, np_in, ns_in = load_noise_data(
            hdf5_file, csv_file, window_size=window_size,
            max_samples=noise_cap, seed=seed)
        if len(nseq) > 0:
            noise_pool_seqs.append(nseq)
            idx = np.random.default_rng(seed).permutation(len(nseq))
            n_tr, n_va = NC['train'], NC['val']
            for key, part in (('train', idx[:n_tr]),
                              ('val', idx[n_tr:n_tr + n_va]),
                              ('test', idx[n_tr + n_va:])):
                noise_parts[key].append((nseq[part], np_in[part], ns_in[part]))
            print(f"  noise 划分: train/val/test = {n_tr}/"
                  f"{len(idx[n_tr:n_tr + n_va])}/{len(idx[n_tr + n_va:])}")

    def _assemble(split):
        a, b, c2 = SIG[split]
        for w, pp, ss in noise_parts[split]:
            a = np.concatenate([a, w], 0)
            b = np.concatenate([b, pp], 0)
            c2 = np.concatenate([c2, ss], 0)
        return a, b, c2

    seqs_tr, p_tr_in, s_tr_in = _assemble('train')
    seqs_va, p_va_in, s_va_in = _assemble('val')
    seqs_te, p_te_in, s_te_in = _assemble('test')

    # ── offset 偏移增强（与原实现一致，逐 split 各自增强）──────────────
    noise_pool = (np.concatenate(noise_pool_seqs, axis=0)[:256]
                  if noise_pool_seqs else None)

    def _augment(seqs_, p_in_, s_in_, tag):
        if args.offset_augment_reps <= 0:
            return seqs_, p_in_, s_in_
        if noise_pool is None:
            print(f"[偏移增强/{tag}] ⚠ 无噪声数据，退化为复制原波形填充（效果差）")
        before = len(seqs_)
        need = estimate_ram_gb(before * (2 + args.offset_augment_reps), window_size)
        avail = available_ram_gb()
        print(f"[偏移增强/{tag}] {before} 条，预计峰值 {need:.1f} GB"
              + (f"，可用 {avail:.1f} GB" if avail is not None else ""))
        out = augment_offsets(
            seqs_, p_in_, s_in_, noise_pool,
            reps=args.offset_augment_reps, off_min=args.offset_aug_min,
            off_max=(args.offset_aug_max or (window_size - 200)),
            rng=np.random.default_rng(seed))
        print(f"[偏移增强/{tag}] {before} → {len(out[0])} 条")
        return out

    if args.offset_augment_reps > 0:
        print(f"[偏移增强] 逐 split 增强各 {args.offset_augment_reps} 份")
    else:
        print("[偏移增强] 未启用。STEAD 的 T=W=6000 使 P 恒定落在窗内 ~500~900，"
              "模型会带强位置先验；")
        print("           在 P 位置散布的数据（如 INSTANCE）上偏移>1300 即失效。"
              "建议加 --offset-augment-reps 3")
    seqs_tr, p_tr_in, s_tr_in = _augment(seqs_tr, p_tr_in, s_tr_in, 'train')
    seqs_va, p_va_in, s_va_in = _augment(seqs_va, p_va_in, s_va_in, 'val')
    seqs_te, p_te_in, s_te_in = _augment(seqs_te, p_te_in, s_te_in, 'test')

    # 波形库（第二事件增强）：排除 val/test 事件，避免测试事件经增强泄漏
    wave_pool, pool_seqs = None, []
    for hdf5_file, csv_file in pool_files:
        if not (os.path.exists(hdf5_file) and os.path.exists(csv_file)):
            print(f"\n[波形库] 跳过（文件不存在）: {hdf5_file}")
            continue
        pc = os.path.relpath(os.path.abspath(hdf5_file), here)
        pc = pc[:-5] if pc.endswith('.hdf5') else pc
        pmeta = meta if pc in chunk_names else load_chunks(data_dir, [pc])
        pn = pool_names(pmeta, exclude, args.pool_max_samples, seed)
        print(f"\nLoading waveform pool from: {hdf5_file} & {csv_file}"
              f"（{len(pn)} 条，已排除 val/test 事件）")
        ps, _, _ = load_dual_head_data(
            hdf5_file, csv_file, window_size=window_size,
            max_samples=args.pool_max_samples,
            seed=seed, off_min=args.p_offset_min, off_max=args.p_offset_max,
            window_mode=args.window_mode, p_min_lead=args.p_min_lead,
            p_max_lead=args.p_max_lead, trace_names=pn)
        if len(ps) > 0:
            pool_seqs.append(ps)
    if pool_seqs:
        wave_pool = np.concatenate(pool_seqs, axis=0)
        # copy=False：已是 float32 时不再复制一份，峰值内存减半
        wave_pool = wave_pool.astype(np.float32, copy=False)
        print(f"\nWaveform pool shape: {wave_pool.shape}")

    # ── 2. 划分已由事件级划分完成（此处仅做噪声标记与自检）──────────────
    X_train, p_train, s_train = seqs_tr, p_tr_in, s_tr_in
    X_val,   p_val,   s_val   = seqs_va, p_va_in, s_va_in
    X_test,  p_test,  s_test  = seqs_te, p_te_in, s_te_in
    n_train = np.where(p_train <= -1.5, 1, 0)
    n_val   = np.where(p_val <= -1.5, 1, 0)
    n_test  = np.where(p_test <= -1.5, 1, 0)
    seqs = X_train if len(X_train) else np.zeros((0, window_size, 3), np.float32)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Noise in train/val/test: {int(n_train.sum())}/{int(n_val.sum())}/"
          f"{int(n_test.sum())}")

    if args.dry_run:
        print('\n[dry-run] 数据装配完成，跳过训练。'
              '正式训练请用 --split-table 复用同一张划分表。')
        return 0

    # ── 3. offset 已在载入取窗时随机化，此处 shift 恒为 0 ─────────────
    # （保留 Dataset 的 shift 机制以备将来需要；若改成「载入时固定基准 +
    #   训练时平移」，务必让两边的坐标约定一致，否则标签会与波形错位。）
    sh_train = np.zeros(len(p_train), dtype=int)
    sh_val   = np.zeros(len(p_val), dtype=int)
    sh_test  = np.zeros(len(p_test), dtype=int)

    def _p_rel(p_samp, shift, noise_mask):
        """窗内 P 采样点位置 → 相对比例；噪声样本填 -2.0。"""
        rel = (np.asarray(p_samp, float) - shift) / window_size
        rel = np.clip(rel, 0.0, 1.0)
        rel[np.asarray(noise_mask, bool)] = -2.0
        return rel

    def _s_rel(s_samp, shift, noise_mask):
        """窗内 S 采样点位置 → 相对比例；无效/噪声填 -1.0。"""
        srel = np.where(np.asarray(s_samp, float) >= 0,
                        (np.asarray(s_samp, float) - shift) / window_size, -1.0)
        srel = np.where(srel >= 0, np.clip(srel, 0.0, 1.0), -1.0)
        srel[np.asarray(noise_mask, bool)] = -1.0
        return srel

    p_train_rel = _p_rel(p_train, sh_train, n_train)
    p_val_rel   = _p_rel(p_val,   sh_val,   n_val)
    p_test_rel  = _p_rel(p_test,  sh_test,  n_test)
    s_train_rel = _s_rel(s_train, sh_train, n_train)
    s_val_rel   = _s_rel(s_val,   sh_val,   n_val)
    s_test_rel  = _s_rel(s_test,  sh_test,  n_test)

    _p_sig = p_train_rel[(p_train_rel > -1.5) & (p_train_rel > 0)]
    if _p_sig.size:
        print(f"P-in-window offset: {_p_sig.min() * window_size:.0f}–"
              f"{_p_sig.max() * window_size:.0f} 采样点 "
              f"(目标区间 [{args.p_offset_min}, {args.p_offset_max}])")

    # ── 4. 增强器 / Dataset / DataLoader ──────────────────────────────
    augmentation = SimpleAugmentation3C(per_ch=1, second_event_pool=wave_pool)

    ds_kwargs = dict(window_size=window_size,
                     rect_width=args.rect_width,
                     rect_adaptive=args.rect_adaptive,
                     rect_w_min=args.rect_w_min,
                     rect_w_max=args.rect_w_max,
                     s_phase_scale=args.s_phase_scale,
                     pick_sigma=args.pick_sigma)

    train_ds = DualHeadWaveDataset(
        X_train, p_train_rel, s_train_rel, shift=sh_train,
        is_train=True, transforms=augmentation, **ds_kwargs)
    val_ds = DualHeadWaveDataset(
        X_val, p_val_rel, s_val_rel, shift=sh_val,
        is_train=False, scaler=train_ds.scaler, **ds_kwargs)
    test_ds = DualHeadWaveDataset(
        X_test, p_test_rel, s_test_rel, shift=sh_test,
        is_train=False, scaler=train_ds.scaler, **ds_kwargs)

    scaler_path = os.path.join(args.output_dir, 'scaler_3C.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(train_ds.scaler, f)
    print(f"Scaler saved to '{scaler_path}'")

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, batch_size=max(1, args.batch_size // 2),
                              shuffle=False, **loader_kwargs)

    # ── 5. 模型 ───────────────────────────────────────────────────────
    input_dim = seqs.shape[-1]
    print(f"\nModel input_dim = {input_dim}  (raw 3-component waveforms)")
    print(f"use_transformer = {args.use_transformer}, use_lfem = {args.use_lfem}, "
          f"mamba_mode = {args.mamba_mode}")

    model = DualHeadMambaNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        use_transformer=args.use_transformer,
        use_lfem=args.use_lfem,
        mamba_mode=args.mamba_mode,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    if args.init_detection_ckpt and os.path.exists(args.init_detection_ckpt):
        print(f"\n从旧检测权重初始化: {args.init_detection_ckpt}")
        ck = torch.load(args.init_detection_ckpt, map_location=device,
                        weights_only=False)
        state = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        model.load_detection_backbone(state)

    # ── 6. 训练 ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.output_dir, 'best.pt')
    save_payload = {
        'model_config': {
            'input_dim': input_dim, 'hidden_dim': args.hidden_dim,
            'use_transformer': args.use_transformer, 'use_lfem': args.use_lfem,
            'mamba_mode': args.mamba_mode,
            'use_cross_channel_attention': False,
            'pool_factor': POOL_FACTOR,
            'heads': ['p_det', 's_det', 'pick'],
        },
        'flags': {'rect_adaptive': args.rect_adaptive,
                  'rect_width': args.rect_width,
                  'rect_w_min': args.rect_w_min, 'rect_w_max': args.rect_w_max,
                  'pick_sigma': args.pick_sigma,
                  'p_offset_min': args.p_offset_min,
                  'p_offset_max': args.p_offset_max,
                  'window_size': window_size},
        'scaler': train_ds.scaler,
    }

    # ── 位置先验自检的素材：用验证集原始波形（未标准化）──────────
    pos_waves = np.asarray(X_val, dtype=np.float32)
    pos_p = np.asarray(p_val, dtype=float) - np.asarray(sh_val, dtype=float)
    pos_noise = np.asarray(n_val, dtype=bool)
    print(f"位置先验自检: 将在 {len(pos_waves)} 条验证波形上测试 "
          f"offset={POS_CHECK_OFFSETS}")
    model._scaler_for_check = train_ds.scaler

    print("\nTraining (dual-head: detect + localize)...")
    history = train_dual_head_model(
        model, train_loader, val_loader, device,
        num_epochs=args.num_epochs, lr=args.lr,
        p_weight=args.p_weight, s_weight=args.s_weight,
        noise_weight=args.noise_weight, pick_weight=args.pick_weight,
        pos_weight=args.pos_weight,
        kl_weight=args.kl_weight, soft_argmax_weight=args.soft_argmax_weight,
        focal_weight=args.focal_weight,
        save_path=ckpt_path, save_payload=save_payload, use_amp=args.use_amp,
        accum_steps=args.accum_steps,
        pos_check_waves=pos_waves, pos_check_p=pos_p,
        pos_check_noise=pos_noise,
        pos_check_pool=(noise_pool_seqs[0][:64] if noise_pool_seqs else None),
        pos_check_interval=args.pos_check_interval)

    # ── 6.5 训练结束后的位置先验自检（最终权重）──────────────────────
    rob_final = evaluate_offset_robustness(
        model, pos_waves, pos_p, pos_noise, device,
        offsets=POS_CHECK_OFFSETS,
        noise_pool=(noise_pool_seqs[0][:64] if noise_pool_seqs else None))
    _print_offset_robustness(rob_final, tag='最终权重')
    if rob_final:
        worst = min(rob_final.items(), key=lambda kv: kv[1]['trigger_rate'])
        best = max(rob_final.items(), key=lambda kv: kv[1]['trigger_rate'])
        gap = best[1]['trigger_rate'] - worst[1]['trigger_rate']
        if gap > 30:
            print(f"      ⚠ 触发率在 offset={best[0]} 与 {worst[0]} 之间相差 "
                  f"{gap:.1f} 个百分点 → 存在位置先验；在 P 位置不同的目标域"
                  f"（如 INSTANCE）上会明显掉点。建议扩大 --p-offset-min/max "
                  f"并用波形长于窗口的数据集训练。")

    # ── 7. 测试集评估 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("=" * 60)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        print(f"已载入最优权重 (epoch={ck.get('epoch')})")
    model.eval()

    for tag, loader in (('VALIDATION', val_loader), ('TEST', test_loader)):
        met = evaluate_dual_head(model, loader, device)
        print(f"\n{'=' * 60}")
        print(f"{tag} — DETECTION & LOCALIZATION")
        print(f"{'=' * 60}")
        print(f"  P 检测: Precision={met['p_precision']:.2f}%  "
              f"Recall={met['p_recall']:.2f}%  F1={met['p_f1']:.2f}%")
        print(f"  S 检测: Precision={met['s_precision']:.2f}%  "
              f"Recall={met['s_recall']:.2f}%  F1={met['s_f1']:.2f}%")
        print(f"  P 定位: pick_head MAE={met['pick_mae']}  中位={met['pick_median']}  "
              f"≤5={met['pick_within_5']}%  ≤20={met['pick_within_20']}%")
        print(f"          软argmax MAE={met['pick_softargmax_mae']}  |  "
              f"矩形前缘 MAE={met['edge_mae']} 中位={met['edge_median']}")

    # 混淆矩阵（测试集）
    with torch.no_grad():
        p_all, s_all, nz_all, sv_all = [], [], [], []
        for batch in test_loader:
            x, p_rect, s_rect, p_gauss, s_valid, is_noise = batch
            p_det, s_det, _ = model(x.to(device))
            p_all.append(p_det.cpu()); s_all.append(s_det.cpu())
            nz_all.append(is_noise); sv_all.append(s_valid)
    p_det_metrics, s_det_metrics = compute_detection_metrics(
        torch.cat(p_all), torch.cat(s_all), torch.cat(nz_all), torch.cat(sv_all))
    print_detection_report(p_det_metrics, s_det_metrics)
    plot_confusion_matrix(p_det_metrics, s_det_metrics,
                          save_path=os.path.join(args.output_dir,
                                                 'confusion_matrix.png'))

    # 阈值扫描（P 波检测 F1）
    print(f"\n  [Threshold Scan — P-wave Detection F1]")
    print(f"  {'thresh':>8s}  {'F1(%)':>8s}  {'Prec(%)':>8s}  {'Recall(%)':>8s}")
    for th in np.arange(0.1, 0.91, 0.1):
        pm, _ = compute_detection_metrics(
            torch.cat(p_all), torch.cat(s_all), torch.cat(nz_all),
            torch.cat(sv_all), threshold=th)
        print(f"  {th:>8.1f}  {pm['f1']:>8.2f}  {pm['precision']:>8.2f}  "
              f"{pm['recall']:>8.2f}")

    # 检测样例可视化（前 2 个信号样本 + 1 个噪声样本）
    for vis_idx in range(min(3, len(test_ds))):
        plot_detection_example(
            model, test_ds, device, idx=vis_idx,
            save_path=os.path.join(args.output_dir,
                                   f'detection_example_{vis_idx}.png'),
            window_size=window_size)

    # 训练过程可视化
    plot_training_history(history,
                          save_path=os.path.join(args.output_dir,
                                                 'dual_head_results.png'))

    # 保存训练历史
    with open(os.path.join(args.output_dir, 'history.json'), 'w',
              encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nModel saved as '{ckpt_path}'")
    print(f"Detection examples saved as 'detection_example_*.png'")
    print(f"Training curves saved as 'dual_head_results.png'")
    return 0


if __name__ == '__main__':
    main()

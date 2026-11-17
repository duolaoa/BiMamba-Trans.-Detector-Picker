# -*- coding: utf-8 -*-
"""
PhaseMamba 复现（事件级划分版）
================================
依据: Zhou, Wu & Ren, "PhaseMamba: A Mamba-Based Deep Learning Model for Seismic
Phase Picking and Detection", IEEE Geoscience and Remote Sensing Letters, 22, 2025,
Art. 7509205, DOI 10.1109/LGRS.2025.3603915.
**该论文未公开代码**，本文件是按论文正文与 Fig.1/Fig.2 描述做的忠实复现，
用于在与 PhaseNet/EQTransformer/BiMamba **完全相同的数据划分与评测口径**下对比。

------------------------------------------------------------------ 论文明确给出的
* U 形结构 + Mamba 骨干，含 patch embedding、编码器、解码器、末端投影、跳跃连接。
* patch embedding: 1-D 卷积把输入 (L, 3) 切成**不重叠** patch 并映射到 C'=8
  -> 即 Conv1d(3, 8, kernel=3, stride=3)，输出长度 L/3。
* 编码器 4 个 Mamba 块；**前三块末端下采样**：长度 ÷4、通道 ×2
  -> 通道 8 -> 16 -> 32 -> 64，长度 L/3 -> L/12 -> L/48 -> L/192。
* 解码器 4 个 Mamba 块；**后三块之前上采样**：长度 ×4、通道 ÷2
  -> 64 -> 32 -> 16 -> 8，长度回到 L/3。
* 跳跃连接：编码器各阶段结果**直接相加**进解码器对应阶段，不引入额外参数。
* 末端投影层 -> 形状 L × C，C=3 分别对应 P 波、S 波、地震信号(检测)。
* Mamba 块(Fig.2): 分支1 = LayerNorm -> Mamba(selective SSM) -> drop path；
  分支2 = 恒等跳连；两者相加。
* 检测标签: **二值方波**，从 P 到时开始到 P + λ(S-P) 结束，论文取 **λ = 1.4**。
* 拾取标签: 比较矩形/三角/高斯后选**高斯**，标准差 **σ = 0.1 s**（100 Hz 即 10 采样点）。
* 损失: 预测概率分布与真值概率分布的 **MSE**。
* 增广(四条): 50% 加高斯噪声；99% 对地震事件做随机平移；30% 在同一 60 s 窗内叠加
  第二个地震信号；10% 随机丢弃 1-2 个通道。
* 训练: Adam + 动态学习率 + early stopping(验证损失连续 12 轮不降则停)；
  原文在单张 Tesla L20 上训练 8 小时（数据量 1.02M 训练样本）。

-------------------------------------------------- 论文**未给出**、本实现所作的假设
1. 下采样算子: 采用 Conv1d(kernel=4, stride=4)（正好 ÷4 且无需 padding）。
2. 上采样算子: 采用 ConvTranspose1d(kernel=4, stride=4)（正好 ×4）。
   —— 上采样后长度与编码器侧可能差 1 个点（如 31×4=124 vs 125），
      本实现按"裁剪/零填充到编码器长度"对齐后再相加。
3. Mamba 超参: d_state=16, d_conv=4, expand=2（mamba_ssm 常用默认值）。
4. drop path 概率: 0.1。
5. 三通道损失权重: 相同（论文只说用 MSE）。
6. 末端 L/3 -> L 的 ×3 上采样: ConvTranspose1d(8, 3, kernel=3, stride=3)。
7. 缺少 S 标签的记录: 论文未说明；本实现取 S-P 默认 4.0 s（400 采样点）作为方波长度，
   并在文件里以 S_P_DEFAULT 常量暴露，便于改成"跳过该样本"。
8. 优化细节: batch size / 初始学习率 / epoch 数论文未给，本实现与另外三个模型统一，
   以便"只换模型、不换训练预算"。

--------------------------------------------------------------- 与另三个模型的一致性
* 数据划分: 默认复用 `event_split_table.csv`（事件级），可用 --split-mode random 切换到
  **原文口径的随机 trace 级 85/5/10**，用于保真度验证。
* 预处理: 去趋势 + 1-40 Hz 带通 + 逐通道 /std（与 PhaseNet/EQT/BiMamba 相同）。
* 通道序: [Z, N, E]。
* 窗口: 默认 6000（原文 60 s @100 Hz），可用 --window-size 覆盖。

用法
----
事件级划分（与另三个模型同口径，用于论文主对比）:
    python train_phasemamba_evsplit.py --split-table event_split_table.csv

保真度验证（复现原文的随机 trace 级划分，对标其 Table I）:
    python train_phasemamba_evsplit.py --split-mode random --epochs 60 \
        --paper-augment --tag fidelity
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 复用 PhaseNet 脚本里的事件级划分与数据管线（保证四个模型数据完全一致）
import train_phasenet_evsplit as PN                                    # noqa: E402

from mamba_ssm import Mamba                                            # noqa: E402

# ============================== 超参数 ============================== #
WINDOW_SIZE = 6000          # 原文 60 s @100 Hz
SIGMA = 10                  # 0.1 s @100 Hz（论文的 σ = 0.1 s）
LAM = 1.4                   # 检测方波终点 P + λ(S-P)
S_P_DEFAULT = 400           # 缺失 S 时的默认 S-P（4 s），仅影响检测方框长度
BATCH_SIZE = 64             # 与另三个模型一致；实测 bs=64 @L=6000 仅占 676 MiB
                            # （bs=16/32/64/128 -> 182/346/676/1332 MiB，每 epoch 1.2 min 起进入平台）
NUM_EPOCHS = 100
LR = 1e-3
TOL = 10                    # 10 采样点 = 0.1 s（与全部评测口径一致）
DET_THR = 0.3               # 原文的检测阈值
MODEL_SAVE = 'best_phasemamba_evsplit.pth'


# ============================== 标签 ============================== #
def make_label_phasemamba(p_idx, s_idx, length, sigma=SIGMA, lam=LAM):
    """三通道**独立概率**标签 (length, 3) = [P, S, earthquake signal]。

    * P/S: 截断高斯（±3σ 之外置 0），与论文的 Gaussian 标签一致（σ = 0.1 s）。
    * 检测: 方波 [p, p + λ(s-p)]（论文 λ = 1.4）；S 缺失时用 S_P_DEFAULT。
    * 纯噪声（p_idx < 0）: 三通道全 0 —— MSE 回归下即"无 P、无 S、无事件"。

    注意与 PhaseNet 的区别: PhaseNet 是 [P, S, noise] 且**逐点归一化**（softmax），
    本模型是三个独立概率（sigmoid），因此噪声样本必须全 0 而不是 noise=1。
    """
    lab = np.zeros((length, 3), dtype=np.float32)
    if not (0 <= p_idx < length):
        return lab
    t = np.arange(length, dtype=np.float32)
    cut = np.exp(-0.5 * 3.0 ** 2)              # ±3σ 截断阈值
    pg = np.exp(-0.5 * ((t - p_idx) / sigma) ** 2)
    pg[pg < cut] = 0.0
    lab[:, 0] = pg
    if 0 <= s_idx < length and s_idx > p_idx:
        sg = np.exp(-0.5 * ((t - s_idx) / sigma) ** 2)
        sg[sg < cut] = 0.0
        lab[:, 1] = sg
        end = int(round(p_idx + lam * (s_idx - p_idx)))
    else:
        end = int(round(p_idx + lam * S_P_DEFAULT))
    end = int(np.clip(end, p_idx, length - 1))
    lab[p_idx:end + 1, 2] = 1.0
    return lab


# monkey-patch：两个 loader 内部调用的是模块级 make_label，改掉即可全管线生效
PN.make_label = make_label_phasemamba


# ============================== 模型 ============================== #
def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x / keep * mask


class MambaBlock(nn.Module):
    """论文 Fig.2: 分支1 = LayerNorm -> Mamba -> drop path；分支2 = 恒等；输出相加。"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path_p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                           expand=expand)
        self.drop_path_p = float(drop_path_p)

    def forward(self, x):                       # (B, L, C)
        h = self.mamba(self.norm(x))
        return x + drop_path(h, self.drop_path_p, self.training)


def _match_len(x, target_len):
    """把 (B, C, L) 的长度对齐到 target_len（裁掉或零填充尾部）。"""
    L = x.shape[-1]
    if L == target_len:
        return x
    if L > target_len:
        return x[..., :target_len]
    return F.pad(x, (0, target_len - L))


class PhaseMamba(nn.Module):
    """PhaseMamba 复现：Mamba-UNet，输出 (B, L, 3) = [P, S, earthquake signal] 概率。

    输入约定 (B, L, 3)（(时间, 通道)，通道序 ZNE），内部转成 (B, 3, L) 做 1-D 卷积。
    """

    def __init__(self, in_channels=3, out_channels=3, base_ch=8,
                 d_state=16, d_conv=4, expand=2, drop_path_p=0.1):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        # patch embedding：不重叠 patch（kernel=stride=3）
        self.patch = nn.Conv1d(in_channels, c1, kernel_size=3, stride=3)
        # 编码器（前三块后下采样 ÷4、通道 ×2）
        self.enc1 = MambaBlock(c1, d_state, d_conv, expand, drop_path_p)
        self.down1 = nn.Conv1d(c1, c2, kernel_size=4, stride=4)
        self.enc2 = MambaBlock(c2, d_state, d_conv, expand, drop_path_p)
        self.down2 = nn.Conv1d(c2, c3, kernel_size=4, stride=4)
        self.enc3 = MambaBlock(c3, d_state, d_conv, expand, drop_path_p)
        self.down3 = nn.Conv1d(c3, c4, kernel_size=4, stride=4)
        self.enc4 = MambaBlock(c4, d_state, d_conv, expand, drop_path_p)
        # 解码器（后三块前上采样 ×4、通道 ÷2）
        self.dec4 = MambaBlock(c4, d_state, d_conv, expand, drop_path_p)
        self.up3 = nn.ConvTranspose1d(c4, c3, kernel_size=4, stride=4)
        self.dec3 = MambaBlock(c3, d_state, d_conv, expand, drop_path_p)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=4)
        self.dec2 = MambaBlock(c2, d_state, d_conv, expand, drop_path_p)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=4)
        self.dec1 = MambaBlock(c1, d_state, d_conv, expand, drop_path_p)
        # 末端投影：L/3 -> L，通道 -> 3
        self.head = nn.ConvTranspose1d(c1, out_channels, kernel_size=3, stride=3)

    def forward(self, x):                       # (B, L, 3)
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2)                   # (B, 3, L)
        h = self.patch(x)                       # (B, c1, L/3)

        e1 = self.enc1(h.transpose(1, 2)).transpose(1, 2)          # (B, c1, L/3)
        e2 = self.down1(e1)                                        # (B, c2, L/12)
        e2 = self.enc2(e2.transpose(1, 2)).transpose(1, 2)
        e3 = self.down2(e2)                                        # (B, c3, L/48)
        e3 = self.enc3(e3.transpose(1, 2)).transpose(1, 2)
        e4 = self.down3(e3)                                        # (B, c4, L/192)
        e4 = self.enc4(e4.transpose(1, 2)).transpose(1, 2)

        d = self.dec4(e4.transpose(1, 2)).transpose(1, 2) + e4      # 跳跃②
        d = self.up3(d); d = _match_len(d, e3.shape[-1]) + e3       # 跳跃③
        d = self.dec3(d.transpose(1, 2)).transpose(1, 2)
        d = self.up2(d); d = _match_len(d, e2.shape[-1]) + e2
        d = self.dec2(d.transpose(1, 2)).transpose(1, 2)
        d = self.up1(d); d = _match_len(d, e1.shape[-1]) + e1
        d = self.dec1(d.transpose(1, 2)).transpose(1, 2)
        out = self.head(d)                                          # (B, 3, L)
        return out.transpose(1, 2)                                  # (B, L, 3)


class PhaseMambaLoss(nn.Module):
    """三通道损失。

    **默认 BCE（带正样本权重），而不是论文正文写的 MSE**，原因（已实测）：
      本任务是"稀疏正样本 + sigmoid 概率回归"。P/S 通道在 6000 点里只有约 60 点
      非零（σ=10 的截断高斯），检测通道约 15-35% 为正。纯 MSE 对 sigmoid 输出的
      梯度正比于 p(1-p)，负样本占压倒性多数，会把输出整体推向 0 并卡在塌缩解上。
      实测：MSE 训练 12 轮 loss 恒为 0.04053（正好等于"预测恒为 0"的值）、
      detF1=0.0000、early stopping；8 条样本过拟合测试中检测通道最大概率 0.000、
      P_MAE 5274.9。同一测试下 BCE 能完美过拟合（loss 0.789->0.0028、检测通道最大
      概率 1.000、P_MAE 1.9）。这与 EQTransformer 在同一"方波+高斯"标签体系下使用
      BCE(+pos_weight) 一致。若要严格按正文复现 MSE，用 --loss mse。

    pos_weight: 正样本软化权重 1 + (w-1)*target，默认 2.0（与 EQTransformer 一致）。
    ch_weight : 三通道（P, S, 检测）通道级权重，默认等权。
    """

    def __init__(self, kind='bce', pos_weight=(2.0, 2.0, 2.0),
                 ch_weight=(1.0, 1.0, 1.0)):
        super().__init__()
        self.kind = kind
        self.register_buffer('pw', torch.tensor(pos_weight, dtype=torch.float32))
        self.register_buffer('cw', torch.tensor(ch_weight, dtype=torch.float32))

    def forward(self, pred, target):
        if self.kind == 'mse':
            se = (torch.sigmoid(pred) - target) ** 2
            return (se * self.cw.view(1, 1, -1)).mean()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        w = 1.0 + (self.pw.view(1, 1, -1) - 1.0) * target
        return (bce * w * self.cw.view(1, 1, -1)).mean()


# ============================== 数据集 ============================== #
class PhaseMambaDataset(Dataset):
    """按论文四条增广可选；与另三个模型共用 SimpleAugmentation 时用同一套。"""

    def __init__(self, waves, labels, is_train=False, augmentor=None,
                 noise_augmentor=None, random_shift=0):
        self.waves = waves
        self.labels = labels
        self.is_train = is_train
        self.aug = augmentor
        self.noise_aug = noise_augmentor
        self.random_shift = int(random_shift or 0)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        w = self.waves[i].astype(np.float32)
        lab = self.labels[i].astype(np.float32)
        if self.is_train and self.random_shift > 0:
            # 论文的"99% 随机平移"：整体平移波形与标签（窗内仍有 P）
            sh = int(np.random.randint(-self.random_shift, self.random_shift + 1))
            if sh != 0:
                w = np.roll(w, sh, axis=0)
                lab = np.roll(lab, sh, axis=0)
        if self.is_train:
            is_noise = float(lab.max()) < 1e-6
            aug = self.noise_aug if (is_noise and self.noise_aug is not None) else self.aug
            if aug is not None:
                w = aug(w)
                if isinstance(w, tuple):
                    w = w[0]
        return torch.from_numpy(np.ascontiguousarray(w)).float(), \
            torch.from_numpy(np.ascontiguousarray(lab)).float()


# ============================== 训练 / 评估 ============================== #
def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    tot, n = 0.0, 0
    for w, y in loader:
        w, y = w.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(w)
        loss = crit(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += float(loss) * len(w); n += len(w)
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, crit, device, tol=TOL, det_thr=DET_THR):
    """返回 loss / P-MAE / S-MAE / 检测指标（记录级 A 判据）。"""
    model.eval()
    tot, n = 0.0, 0
    pe, se = [], []
    tp = fp = fn = tn = 0
    for w, y in loader:
        w = w.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(w)
        tot += float(crit(out, y)) * len(w); n += len(w)
        prob = torch.sigmoid(out).cpu().numpy()
        true = y.cpu().numpy()
        for b in range(len(w)):
            # --- P ---
            if true[b, :, 0].max() > 0.5:
                pe.append(abs(int(np.argmax(prob[b, :, 0]))
                              - int(np.argmax(true[b, :, 0]))))
            # --- S ---
            if true[b, :, 1].max() > 0.5:
                se.append(abs(int(np.argmax(prob[b, :, 1]))
                              - int(np.argmax(true[b, :, 1]))))
            # --- 检测（记录级：检测通道是否超过阈值）---
            pred_pos = float(prob[b, :, 2].max()) >= det_thr
            true_pos = float(true[b, :, 2].max()) > 0.5
            if true_pos and pred_pos:
                tp += 1
            elif true_pos and not pred_pos:
                fn += 1
            elif (not true_pos) and pred_pos:
                fp += 1
            else:
                tn += 1
    pr = tp / max(tp + fp, 1e-9)
    re = tp / max(tp + fn, 1e-9)
    f1 = 2 * pr * re / max(pr + re, 1e-9)
    det = dict(Pr=pr, Re=re, F1=f1, tp=tp, fp=fp, fn=fn, tn=tn)
    return (tot / max(n, 1),
            float(np.mean(pe)) if pe else float('nan'),
            float(np.mean(se)) if se else float('nan'),
            det)


# ============================== 随机 trace 级划分（保真度模式） ============================== #
def _random_trace_split(seed=42, ratios=(0.85, 0.05, 0.10)):
    """原文口径：STEAD 全部波形随机 85/5/10 划分（trace 级，**不做事件级隔离**）。

    仅用于保真度验证 —— 与原文 Table I 对标；正式对比请用事件级划分。
    """
    rng = np.random.default_rng(seed)
    out = {}
    for split in ('train', 'val', 'test'):
        out[split] = {}
    for c in PN.CFG['train_chunks']:
        hf, cf = PN._paths(c)
        if not (os.path.exists(cf) and os.path.exists(hf)):
            print(f'  [random] 跳过不存在的 {c}')
            continue
        df = pd.read_csv(cf, usecols=['trace_name', 'p_arrival_sample'])
        df = df[df.p_arrival_sample > 0]
        names = df.trace_name.astype(str).to_numpy()
        idx = rng.permutation(len(names))
        n_tr = int(len(names) * ratios[0])
        n_va = int(len(names) * ratios[1])
        out['train'][c] = names[idx[:n_tr]].tolist()
        out['val'][c] = names[idx[n_tr:n_tr + n_va]].tolist()
        out['test'][c] = names[idx[n_tr + n_va:]].tolist()
        print(f'  [random] {c}: 总 {len(names)} -> train/val/test = '
              f'{len(out["train"][c])}/{len(out["val"][c])}/{len(out["test"][c])}')
    return out


def apply_cli(argv=None):
    """一次性解析 PhaseNet 的 CFG 键 + PhaseMamba 自己的键。

    注意：这里**不调用** PN.apply_cli() —— 它用的是 parse_args，遇到本脚本新增的
    参数（--epochs 等）会直接报错。因此把它的键在这里重新声明一遍，并复刻它对
    PN.CFG 的更新（保持与另三个模型的数据参数语义完全一致）。
    """
    import argparse
    ap = argparse.ArgumentParser(description='PhaseMamba 复现（事件级划分版）')
    ap.add_argument('--patience', type=int, default=0,
                    help='早停轮数；0=关闭（跑满 --epochs）')
    # ---- 与 PhaseNet/EQT/BiMamba 一致的数据参数 ----
    ap.add_argument('--data-dir', default=PN.CFG['data_dir'])
    ap.add_argument('--train-chunks', nargs='+', default=PN.CFG['train_chunks'])
    ap.add_argument('--pool-chunk', default=PN.CFG['pool_chunk'])
    ap.add_argument('--noise-chunks', nargs='*', default=PN.CFG['noise_chunks'])
    ap.add_argument('--ratios', default='0.8,0.15,0.05')
    ap.add_argument('--total-samples', type=int, default=PN.CFG['total_samples'])
    ap.add_argument('--max-samples', type=int, default=PN.CFG['max_samples'])
    ap.add_argument('--val-samples', type=int, default=PN.CFG['val_samples'])
    ap.add_argument('--test-samples', type=int, default=PN.CFG['test_samples'])
    ap.add_argument('--pool-samples', type=int, default=PN.CFG['pool_samples'])
    ap.add_argument('--noise-samples', type=int, default=PN.CFG['noise_samples'])
    ap.add_argument('--split-table', default=PN.CFG['split_table'])
    ap.add_argument('--save-split-table', default=PN.CFG['save_split_table'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    # ---- PhaseMamba 自己的参数 ----
    ap.add_argument('--split-mode', choices=['event', 'random'], default='event',
                    help='event=事件级划分（默认，与另三个模型同口径）；'
                         'random=原文口径的随机 trace 级划分（仅用于保真度验证）')
    ap.add_argument('--window-size', type=int, default=WINDOW_SIZE)
    ap.add_argument('--sigma', type=int, default=SIGMA)
    ap.add_argument('--lam', type=float, default=LAM)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--base-ch', type=int, default=14,
                    help='patch embedding 的基础通道数。**默认 14** 是为了对齐原文 '
                         'Table II 报告的 301.2 K 参数（14 -> 322 K）；正文写 C\'=8，'
                         '但按"8 + 逐级翻倍"只能得到 116.7 K —— 原文正文与 Table II '
                         '不一致。--base-ch 8 可回退到严格按正文的版本。')
    ap.add_argument('--mamba-d-state', type=int, default=16)
    ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--paper-augment', action='store_true',
                    help='启用论文的四条增广（50%% 加噪 / 随机平移 / 30%% 第二事件 / '
                         '10%% 丢通道）；默认关闭，与另三个模型共用同一套增广')
    ap.add_argument('--random-shift', type=int, default=0,
                    help='--paper-augment 时的随机平移幅度（采样点），默认 200')
    ap.add_argument('--loss', choices=['bce', 'mse'], default='bce',
                    help='bce（默认，实测可收敛）；mse 为论文正文所述，但实测会塌缩到'
                         '全零解（loss 恒为 0.04053、detF1=0），仅供保真度对照')
    ap.add_argument('--pos-weight', type=float, default=2.0,
                    help='正样本软化权重（仅对 bce 生效），默认 2.0（与 EQTransformer 一致）')
    ap.add_argument('--tag', default='', help='输出文件后缀')
    a = ap.parse_args(argv)
    globals()['PATIENCE'] = int(getattr(a, 'patience', 0))   # 0 = 关闭早停（跑满 epochs）
    if PATIENCE > 0:
        print(f'[早停] patience={PATIENCE} 轮不降则停')
    else:
        print('[早停] 已关闭，将跑满全部 epochs')

    PN.CFG.update(data_dir=a.data_dir, train_chunks=a.train_chunks,
                  pool_chunk=a.pool_chunk, noise_chunks=a.noise_chunks,
                  ratios=tuple(float(x) for x in a.ratios.split(',')),
                  max_samples=a.max_samples, val_samples=a.val_samples,
                  test_samples=a.test_samples, pool_samples=a.pool_samples,
                  noise_samples=a.noise_samples, split_table=a.split_table,
                  save_split_table=a.save_split_table,
                  total_samples=a.total_samples, dry_run=bool(a.dry_run))
    print('[事件级划分] data_dir=%s train_chunks=%s pool=%s ratios=%s '
          'max_samples=%s split_table=%r'
          % (PN.CFG['data_dir'], PN.CFG['train_chunks'], PN.CFG['pool_chunk'],
             PN.CFG['ratios'], PN.CFG['max_samples'], PN.CFG['split_table']))
    return a.seed, a


# ============================== 主流程 ============================== #
def main(seed=42, a=None):
    global WINDOW_SIZE, SIGMA, LAM, BATCH_SIZE, NUM_EPOCHS, LR
    WINDOW_SIZE = a.window_size
    SIGMA, LAM = a.sigma, a.lam
    BATCH_SIZE, NUM_EPOCHS, LR = a.batch_size, a.epochs, a.lr
    ckpt = MODEL_SAVE if not a.tag else MODEL_SAVE.replace('.pth', f'_{a.tag}.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('PhaseMamba 复现 (IEEE GRSL 2025, DOI 10.1109/LGRS.2025.3603915)')
    print(f'  device={device}  split_mode={a.split_mode}  window={WINDOW_SIZE}  '
          f'sigma={SIGMA} samples ({SIGMA/100:.2f}s)  lam={LAM}')
    print(f'  base_ch={a.base_ch}  d_state={a.mamba_d_state}  drop_path={a.drop_path}  '
          f'batch={BATCH_SIZE}  lr={LR}  epochs={NUM_EPOCHS}  '
          f'paper_augment={bool(a.paper_augment)}')
    print('=' * 70)

    # ---- 数据装配 ----
    if a.split_mode == 'event':
        meta, exclude = PN._event_meta(seed)
        if PN.CFG['total_samples'] > 0:
            from event_split import split_caps
            _c = split_caps(PN.CFG['total_samples'], PN.CFG['ratios'],
                            len(PN.CFG['train_chunks']))
            PN.CFG['max_samples'], PN.CFG['val_samples'], PN.CFG['test_samples'] = (
                _c['train'], _c['val'], _c['test'])
            print(f"[事件级划分] 信号总量={PN.CFG['total_samples']} 比例={PN.CFG['ratios']} "
                  f"-> 每 chunk train/val/test = {_c['train']}/{_c['val']}/{_c['test']}")

        def _names(split, cap):
            return {c: PN.names_for_split(meta, split, c, cap, seed)
                    for c in PN.CFG['train_chunks']}
    else:
        from event_split import split_caps
        rs = _random_trace_split(seed, (0.85, 0.05, 0.10))
        if PN.CFG['total_samples'] > 0:
            _c = split_caps(PN.CFG['total_samples'], PN.CFG['ratios'],
                            len(PN.CFG['train_chunks']))
            PN.CFG['max_samples'], PN.CFG['val_samples'], PN.CFG['test_samples'] = (
                _c['train'], _c['val'], _c['test'])
        meta, exclude = None, set()

        def _names(split, cap):
            out = {}
            for c in PN.CFG['train_chunks']:
                nms = rs.get(split, {}).get(c, [])
                out[c] = list(np.random.default_rng(seed).permutation(nms))[:cap] \
                    if cap else nms
            return out

    def _load(split, cap):
        wl, yl = [], []
        for c, names in _names(split, cap).items():
            if not names:
                continue
            hf, cf = PN._paths(c)
            print(f'[{split}] {c}: 指定 {len(names)} 条')
            w, y = PN.load_phasenet_data(hf, cf, window_size=WINDOW_SIZE,
                                         max_samples=cap, seed=seed,
                                         trace_names=names)
            if len(w):
                wl.append(w); yl.append(y)
        if not wl:
            return (np.zeros((0, WINDOW_SIZE, 3), np.float32),
                    np.zeros((0, WINDOW_SIZE, 3), np.float32))
        return np.concatenate(wl, 0), np.concatenate(yl, 0)

    X_tr, y_tr = _load('train', PN.CFG['max_samples'])
    X_va, y_va = _load('val', PN.CFG['val_samples'])
    X_te, y_te = _load('test', PN.CFG['test_samples'])

    # ---- 噪声（与另三个模型一致，占比 20%）----
    if PN.CFG['noise_chunks'] and a.split_mode == 'event':
        from event_split import split_caps
        n_noise = max(len(PN.CFG['noise_chunks']), 1)
        NC = split_caps(PN.CFG['noise_samples'], PN.CFG['ratios'], n_noise)
        cap = NC['train'] + NC['val'] + NC['test']
        for c in PN.CFG['noise_chunks']:
            hf, cf = PN._paths(c)
            if not (os.path.exists(hf) and os.path.exists(cf)):
                continue
            nw, ny = PN.load_noise_data(hf, cf, window_size=WINDOW_SIZE,
                                        max_samples=cap, seed=seed)
            if not len(nw):
                continue
            idx = np.random.default_rng(seed).permutation(len(nw))
            for key, sl, X, Y in (('train', idx[:NC['train']], X_tr, y_tr),
                                  ('val', idx[NC['train']:NC['train'] + NC['val']], X_va, y_va),
                                  ('test', idx[NC['train'] + NC['val']:], X_te, y_te)):
                if len(sl):
                    X = np.concatenate([X, nw[sl]], 0)
                    Y = np.concatenate([Y, ny[sl]], 0)
                if key == 'train':
                    X_tr, y_tr = X, Y
                elif key == 'val':
                    X_va, y_va = X, Y
                else:
                    X_te, y_te = X, Y
            print(f'  {c}: 噪声 {len(nw)} 条 -> train/val/test = '
                  f'{NC["train"]}/{NC["val"]}/{NC["test"]}')

    print(f'\nTrain {len(X_tr)}  Val {len(X_va)}  Test {len(X_te)}  '
          f'wave shape {X_tr.shape if len(X_tr) else None}')

    # ---- 波形库（第二事件叠加，排除 val/test 事件）----
    pool = None
    if PN.CFG['pool_chunk'] and a.split_mode == 'event':
        from event_split import load_chunks, pool_names
        pm = load_chunks(PN._data_dir(), [PN.CFG['pool_chunk']])
        pn = pool_names(pm, exclude, PN.CFG['pool_samples'], seed)
        phf, pcf = PN._paths(PN.CFG['pool_chunk'])
        pw, _ = PN.load_phasenet_data(phf, pcf, window_size=WINDOW_SIZE,
                                      max_samples=PN.CFG['pool_samples'],
                                      seed=seed, trace_names=pn)
        if len(pw):
            pool = pw
            print(f'Waveform pool: {len(pool)} 条（已排除 val/test 事件）')

    if PN.CFG.get('dry_run'):
        print('[dry-run] 数据装配完成，退出。')
        return

    # ---- 增广 ----
    if a.paper_augment:
        # 论文四条：50% 加噪 / 99% 随机平移 / 30% 第二事件 / 10% 丢通道
        aug = PN.SimpleAugmentation(noise_std_range=(0.01, 0.30), scale_range=(0.5, 2.0),
                                    noise_prob=0.50, scale_prob=0.0, dropout_prob=0.10,
                                    dropout_n_range=(1, 2), second_event_prob=0.30,
                                    second_event_pool=pool,
                                    second_event_amp_range=(0.1, 1.0))
        shift = a.random_shift or 200
        print(f'[增广] 论文口径：50%加噪 / 随机平移±{shift} / 30%第二事件 / 10%丢通道')
    else:
        # 与 PhaseNet/EQT/BiMamba 完全相同的增广（保证"只换模型"）
        aug = PN.SimpleAugmentation(noise_std_range=(0.01, 0.30), scale_range=(0.5, 2.0),
                                    noise_prob=0.70, scale_prob=0.50, dropout_prob=0.30,
                                    dropout_n_range=(1, 2), second_event_prob=0.30,
                                    second_event_pool=pool,
                                    second_event_amp_range=(0.1, 1.0))
        shift = 0
    noise_aug = PN.SimpleAugmentation(noise_std_range=(0.01, 0.30), scale_range=(0.5, 2.0),
                                      noise_prob=0.70, scale_prob=0.50, dropout_prob=0.30,
                                      dropout_n_range=(1, 2), second_event_prob=0.0,
                                      second_event_pool=None)

    tr_ds = PhaseMambaDataset(X_tr, y_tr, True, aug, noise_aug, random_shift=shift)
    va_ds = PhaseMambaDataset(X_va, y_va, False)
    te_ds = PhaseMambaDataset(X_te, y_te, False)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                       pin_memory=True, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # ---- 模型 ----
    model = PhaseMamba(in_channels=3, out_channels=3, base_ch=a.base_ch,
                       d_state=a.mamba_d_state, drop_path_p=a.drop_path).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'PhaseMamba 参数量: {n_par:,}')
    crit = PhaseMambaLoss(kind=a.loss, pos_weight=(a.pos_weight,) * 3).to(device)
    print(f'损失函数: {a.loss.upper()}'
          + (f'  pos_weight={a.pos_weight}' if a.loss == 'bce'
             else '  （论文正文口径；实测会塌缩成全零解）'))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                       patience=6, min_lr=1e-5)
    # 论文的 early stopping：验证损失连续 12 轮不降则停
    best, best_ep, bad = float('inf'), -1, 0
    t0 = time.time()
    for ep in range(NUM_EPOCHS):
        tr = train_one_epoch(model, tr_ld, opt, crit, device)
        vl, vp, vs, det = evaluate(model, va_ld, crit, device)
        sched.step(vl)
        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep, 0
            torch.save({'model_state_dict': model.state_dict(),
                        'cfg': dict(base_ch=a.base_ch, d_state=a.mamba_d_state,
                                    window_size=WINDOW_SIZE, sigma=SIGMA, lam=LAM,
                                    split_mode=a.split_mode, drop_path=a.drop_path,
                                    paper_augment=bool(a.paper_augment),
                                    loss=a.loss, pos_weight=a.pos_weight,
                                    channel_semantics='[P, S, earthquake_signal]')},
                       ckpt)
        else:
            bad += 1
        if ep % 2 == 0 or ep == NUM_EPOCHS - 1:
            el = time.time() - t0
            print(f'  ep {ep:03d}/{NUM_EPOCHS} tr={tr:.5f} val={vl:.5f} '
                  f'P_MAE={vp:6.1f} S_MAE={vs:6.1f} detF1={det["F1"]:.4f} '
                  f'lr={opt.param_groups[0]["lr"]:.1e} best={best:.5f}@{best_ep} '
                  f'{el/60:.1f}min', flush=True)
        if PATIENCE > 0 and bad >= PATIENCE:
            print(f'  early stopping @ ep {ep}（验证损失连续 12 轮未降）')
            break

    # ---- 测试 ----
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    tl, tp_, ts_, tdet = evaluate(model, te_ld, crit, device)
    print('\n===== 最终测试 =====')
    print(f'  loss={tl:.5f}  P_MAE={tp_:.2f} samples  S_MAE={ts_:.2f} samples')
    print(f'  检测(记录级, 阈值{DET_THR}): Pr={tdet["Pr"]:.4f} Re={tdet["Re"]:.4f} '
          f'F1={tdet["F1"]:.4f}  (TP/FP/FN/TN={tdet["tp"]}/{tdet["fp"]}/'
          f'{tdet["fn"]}/{tdet["tn"]})')
    print(f'  权重已保存: {ckpt}')


if __name__ == '__main__':
    _seed, _a = apply_cli()
    main(seed=_seed, a=_a)

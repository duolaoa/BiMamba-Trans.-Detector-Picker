# -*- coding: utf-8 -*-
"""用 event_split_train 下新训练的权重，在 INSTANCE / CREW / EQS-Denoiser 上做统一推断评估。

统一协议（三家完全一致，消除 anchor 与阈值口径差异）
---------------------------------------------------------------------------
* 通道序统一 [Z,N,E]。测试集原始存的是 [E,N,Z]：
    - INSTANCE  hdf5 data_format/component_order = 'ENZ'
    - CREW      metadata channels = ['BHE','BHN','BHZ']
    - EQS-D     由 make_bench_hdf5.py 生成为 [E,N,Z]
* 预处理统一：去趋势 + Butterworth 2 阶 1-40 Hz + 逐通道除以 std
  （BiMamba 检测器除外：它是 2-30 Hz + 自带 scaler，按其训练口径走）
* 拾取 = window-level（用真值 P 摆窗 [p-500, p+2500)，隔离检测误差）：
    模型在最内 P 通道 argmax → 绝对到时；S 同法（需 s-p < 2500）
* 检测 = sliding（不使用真值）：窗口滑过整条记录，取"存在相位"通道的最大值作为
    该记录"有事件"的得分；在 事件集(正) + 噪声集(负) 上扫阈值 → P/R/F1

模型与通道语义
---------------------------------------------------------------------------
  BiMamba 拾取  best_lightweight_model_3C.pth    (p_logit, s_logit)          P=out0
  PhaseNet      best_phasenet_baseline.pth       softmax (B,T,3)[P,S,noise]  P=ch0
  EQTransformer best_eqtransformer_baseline.pth  sigmoid (B,T,3)[det,P,S]    P=ch1
  BiMamba 检测  best_detection_model_3C.pth      (p_det,s_det) logits        det=out0（6000 窗）
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, detrend, filtfilt, sosfiltfilt
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as _pd

_KEEP = None          # set or None：仅保留这些 trace
_DROP = None          # set or None：剔除这些 trace


def _passes_filter(name):
    k = str(name)
    if _KEEP is not None and k not in _KEEP:
        return False
    if _DROP is not None and k in _DROP:
        return False
    return True


def _load_names(path):
    d = _pd.read_csv(path)
    col = 'trace' if 'trace' in d.columns else d.columns[0]
    return set(d[col].astype(str))


# Defaults correspond to the configuration reported in the paper.
# Every entry can be overridden with an environment variable so that other
# checkpoints (ablation variants, parameter-matched presets) can be evaluated
# without editing this file.
CKPT = dict(
    picker=os.environ.get('EVAL_PICKER_CKPT',
                          os.path.join(HERE, 'best_lightweight_model_3C_pn197k_w6000v3.pth')),
    phasenet=os.environ.get('EVAL_PHASENET_CKPT',
                            os.path.join(HERE, 'best_phasenet_baseline_6000.pth')),
    eqt=os.environ.get('EVAL_EQT_CKPT',
                       os.path.join(HERE, 'best_eqtransformer_baseline.pth')),
    detector=os.environ.get('EVAL_DET_CKPT',
                            os.path.join(HERE, 'best_detection_model_3C_det_pn197k.pth')),
    det_scaler=os.environ.get('EVAL_DET_SCALER',
                              os.path.join(HERE, 'scaler_3C_detection.pkl')),
    phasemamba=os.environ.get('EVAL_PHASEMAMBA_CKPT',
                              os.path.join(HERE, 'best_phasemamba_evsplit__phasemamba6000.pth')),
)

INSTANCE = os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'INSTANCE_DATASET')
CREW_DIR = os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'CREW_Dataset_files')
BENCH = os.path.join(HERE, 'eval_data')
STEAD = os.path.join(os.environ.get('SEISMIC_DATA_ROOT', os.path.expanduser('~/seismic_data')), 'STEAD_data', '1Dsignal')
FS = 100.0
TOL = 10                      # 10 采样点 = 0.1 s
ENZ_TO_ZNE = [2, 1, 0]        # 原始 [E,N,Z] -> [Z,N,E]


# ============================ 预处理 ============================
def _det_pre_band_fast(data3c, lo=1.0, hi=40.0, fs=FS):
    """向量化版：三通道一次去趋势 + 一次 SOS 带通 + 逐通道/std。

    与旧实现数值等价（同为 2 阶 Butterworth 带通 + 线性去趋势 + 逐通道除以 std），
    但把 3 次 Python/filtfilt 调用合并为 1 次，速度约快 2-4 倍。
    """
    x = np.asarray(data3c, dtype=np.float64)
    if x.shape[0] == 3 and x.shape[1] != 3:
        x = x.T
    x = detrend(x, axis=0, type='linear')
    nyq = fs / 2.0
    sos = butter(2, [lo / nyq, min(hi / nyq, 0.99)], btype='band', output='sos')
    y = sosfiltfilt(sos, x, axis=0)
    sd = y.std(axis=0, keepdims=True)
    sd[sd <= 0] = 1.0
    return (y / sd).astype(np.float32)


def _det_pre_band(data3c, lo=1.0, hi=40.0, fs=FS):
    """去趋势 + 指定频带带通 + 逐通道除以 std（与 PS_Detection 新预处理一致）。"""
    return _det_pre_band_fast(data3c, lo=lo, hi=hi, fs=fs)


def preprocess_40(data3c, fs=FS):
    """去趋势 + 1-40 Hz 带通 + 逐通道除以 std（三家拾取模型训练时的口径）。"""
    x = np.asarray(data3c, dtype=np.float64)
    if x.shape[0] == 3 and x.shape[1] != 3:
        x = x.T
    out = np.zeros_like(x, dtype=np.float32)
    nyq = fs / 2.0
    b, a = butter(2, [1.0 / nyq, min(40.0 / nyq, 0.99)], btype='band')
    for c in range(x.shape[1]):
        v = detrend(x[:, c], type='linear')
        v = filtfilt(b, a, v)
        s = np.std(v)
        if s > 0:
            v = v / s
        out[:, c] = v.astype(np.float32)
    return out


# ============================ 模型封装 ============================
class ModelWrap:
    name = 'model'
    window = 3000
    layout = 'TFC'          # 'TFC' = (B,T,3)；'CF' = (B,3,T)

    def to_input(self, wins):
        """wins: (B,T,3) float32 -> 该模型 __getitem__ 实际喂进去的形状。"""
        if self.layout == 'CF':
            return np.ascontiguousarray(wins.transpose(0, 2, 1))
        return wins

    def score(self, win_batch):
        """win_batch: 已按 to_input 摆好形状 -> dict of (B, T) float32"""
        raise NotImplementedError


class BiMambaPicker(ModelWrap):
    name = 'BiMamba-Picker'
    window = 3000

    def __init__(self, device):
        from train_bimamba_evsplit import LightweightPWaveNet
        ck = torch.load(CKPT['picker'], map_location='cpu', weights_only=False)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            self.cfg = dict(ck.get('cfg', {}))
            sd = ck['model_state_dict']
        else:                        # 旧权重：纯 state_dict
            self.cfg, sd = {}, ck
        kw = dict(proj_dim=int(self.cfg.get('proj_dim', 96)),
                  hidden_dim=int(self.cfg.get('hidden_dim', 48)),
                  mamba_layers=int(self.cfg.get('mamba_layers', 3)),
                  d_state=int(self.cfg.get('d_state', 16)),
                  expand=int(self.cfg.get('expand', 2)))
        _mm = str(self.cfg.get('mamba_mode', 'bidirectional'))
        _ut = bool(self.cfg.get('use_transformer', True))
        _ul = bool(self.cfg.get('use_lfem', True))
        _ca = bool(self.cfg.get('use_cross_channel_attention', False))
        m = LightweightPWaveNet(input_dim=3, use_transformer=_ut,
                                mamba_mode=_mm, use_lfem=_ul,
                                use_cross_channel_attention=_ca, **kw).to(device)
        m.load_state_dict(sd)
        m.eval()
        self.m = m
        self.device = device
        self.window = _win_override(self.window)
        # ★ 窗长从权重 cfg 读取（否则 6000 窗权重被喂 3000 窗 ✗）
        self.window = int(self.cfg.get('window_size', 3000))
        if self.cfg:
            print(f"  [BiMamba-Picker] 按权重 cfg 构建: " +
                  ' '.join(f'{k}={v}' for k, v in kw.items()) +
                  f" 参数量={sum(p.numel() for p in m.parameters()):,}")

    @torch.no_grad()
    def score(self, win_batch):
        x = torch.from_numpy(win_batch).to(self.device)
        p, s = self.m(x)                     # logits (B,T)
        return {'P': p.float().cpu().numpy(), 'S': s.float().cpu().numpy()}


# ── 全局窗长覆盖（EVAL_WINDOW），用于评测 6000 窗权重 ✓ ──
def _win_override(default):
    import os as _os
    try:
        return int(_os.environ.get('EVAL_WINDOW') or default)
    except Exception:
        return default

class PhaseNetWrap(ModelWrap):
    name = 'PhaseNet'
    window = 3000
    layout = 'CF'      # PhaseNet 的 Dataset 做了 wave.T -> (B,3,T)

    def __init__(self, device):
        from train_phasenet_evsplit import PhaseNet
        m = PhaseNet(in_channels=3, n_classes=3, filters_root=8, depths=5,
                     kernel_size=7, pool_size=4).to(device)
        m.load_state_dict(torch.load(CKPT['phasenet'], map_location=device,
                                     weights_only=False))
        m.eval()
        self.m = m
        self.device = device
        self.window = _win_override(self.window)

    @torch.no_grad()
    def score(self, win_batch):
        x = torch.from_numpy(win_batch).to(self.device)
        y = self.m(x)                        # (B,T,3) softmax
        y = y.float().cpu().numpy()
        return {'P': y[:, :, 0], 'S': y[:, :, 1], 'noise': y[:, :, 2]}


class EQTWrap(ModelWrap):
    name = 'EQTransformer'
    window = 3000
    layout = 'CF'      # EQT 的 Dataset 做了 wave.T -> (B,3,T)

    def __init__(self, device):
        from train_eqt_evsplit import EQTransformer
        m = EQTransformer(in_channels=3, kernel_size=3, lstm_hidden=64,
                          lstm_pool_factor=1, n_heads=8, dropout=0.1).to(device)
        m.load_state_dict(torch.load(CKPT['eqt'], map_location=device,
                                     weights_only=False))
        m.eval()
        self.m = m
        self.device = device
        self.window = _win_override(self.window)

    @torch.no_grad()
    def score(self, win_batch):
        x = torch.from_numpy(win_batch).to(self.device)
        y = self.m(x)                        # (B,T,3) sigmoid: [det, P, S]
        y = y.float().cpu().numpy()
        return {'det': y[:, :, 0], 'P': y[:, :, 1], 'S': y[:, :, 2]}


class PhaseMambaWrap(ModelWrap):
    """PhaseMamba 复现（IEEE GRSL 2025）包装。

    * 输入 (B, T, 3)（模型内部转成 (B,3,T)）
    * 窗口 6000（原文 60 s @100 Hz）
    * 输出经 sigmoid -> 三通道 [P, S, earthquake signal]；检测取 index 2
    """
    name = 'PhaseMamba'
    layout = 'TFC'
    window = 6000

    def __init__(self, device):
        import importlib
        pm = importlib.import_module('train_phasemamba_evsplit')
        ck = torch.load(CKPT['phasemamba'], map_location='cpu',
                        weights_only=False)
        cfg = ck.get('cfg', {}) if isinstance(ck, dict) else {}
        sd = ck['model_state_dict'] if (isinstance(ck, dict)
                                        and 'model_state_dict' in ck) else ck
        base_ch = int(cfg.get('base_ch', 14))
        d_state = int(cfg.get('d_state', 16))
        win = int(cfg.get('window_size', 6000))
        m = pm.PhaseMamba(in_channels=3, out_channels=3, base_ch=base_ch,
                          d_state=d_state, drop_path_p=0.0).to(device)
        m.load_state_dict(sd)
        m.eval()
        self.m = m
        self.device = device
        self.window = _win_override(self.window)
        type(self).window = win      # 以权重里记录的窗长为准
        print(f'  [PhaseMamba] base_ch={base_ch} d_state={d_state} '
              f'window={win} 参数量={sum(p.numel() for p in m.parameters()):,}')

    @torch.no_grad()
    def score(self, win_batch):
        x = torch.from_numpy(win_batch).to(self.device)
        o = torch.sigmoid(self.m(x))
        return {'P': o[..., 0].float().cpu().numpy(),
                'S': o[..., 1].float().cpu().numpy(),
                'det': o[..., 2].float().cpu().numpy()}


class BiMambaDetector(ModelWrap):
    """独立检测器：6000 窗 + 2-30 Hz + 自带 scaler（按其训练口径）。"""
    name = 'BiMamba-Detector'
    window = 6000

    def __init__(self, device):
        from PS_Detection_evsplit import LightweightPWaveNet as DetNet
        from PS_Detection_evsplit import preprocess_3c as det_pre
        self.det_pre = det_pre
        _ck0 = torch.load(CKPT['detector'], map_location='cpu', weights_only=False)
        _cfg0 = (_ck0.get('cfg', {}) if isinstance(_ck0, dict) else {}) or {}
        _kw = dict(proj_dim=int(_cfg0.get('proj_dim', 96)),
                   hidden_dim=int(_cfg0.get('hidden_dim', 48)),
                   mamba_layers=int(_cfg0.get('mamba_layers', 3)),
                   d_state=int(_cfg0.get('d_state', 16)),
                   expand=int(_cfg0.get('expand', 2)))
        _mm = str(_cfg0.get('mamba_mode', 'bidirectional'))
        _ut = bool(_cfg0.get('use_transformer', True))
        _ul = bool(_cfg0.get('use_lfem', False))
        m = DetNet(input_dim=3, use_transformer=_ut,
                   mamba_mode=_mm, use_lfem=_ul, **_kw).to(device)
        _win = int(_cfg0.get('window_size', 6000))
        type(self).window = _win      # 以权重 cfg 记录的窗长为准
        print(f'  [BiMamba-Detector] 窗长={_win} 采样点 ({_win/100:.1f}s)')
        print(f'  [BiMamba-Detector] 模块组合: mamba={_mm} '
              f'transformer={_ut} lfem={_ul}')
        print('  [BiMamba-Detector] 按权重 cfg 构建: ' +
              ' '.join(f'{k}={v}' for k, v in _kw.items()))
        _ck = _ck0
        if isinstance(_ck, dict) and 'model_state_dict' in _ck:
            self.cfg = dict(_ck.get('cfg', {}))
            _sd = _ck['model_state_dict']
        else:                      # 旧权重：纯 state_dict
            self.cfg = {}
            _sd = _ck
        m.load_state_dict(_sd)
        m.eval()
        self.m = m
        self.device = device
        _lo = float(self.cfg.get('band_low', 2.0))
        _hi = float(self.cfg.get('band_high', 30.0))
        _sc = bool(self.cfg.get('use_scaler', True))
        self._det_band = (_lo, _hi)
        self._det_scaler_on = _sc
        print(f'  [BiMamba-Detector] 预处理按权重 cfg: {_lo}-{_hi} Hz, '
              f'scaler={"ON" if _sc else "OFF"}')
        with open(CKPT['det_scaler'], 'rb') as fh:
            self.scaler = pickle.load(fh)

    def prep(self, win):
        """按权重内嵌 cfg 决定预处理口径。

        * 旧权重（无 cfg）：2-30 Hz + StandardScaler —— 与历史结果完全一致
        * 新权重（优先级2 训练）：1-40 Hz + 逐通道 /std（与三个拾取器统一）-> 不套 scaler
        """
        z = self.det_pre(win) if self._det_scaler_on else _det_pre_band(
            win, *self._det_band)
        if not self._det_scaler_on:
            return z
        return self.scaler.transform(z.reshape(-1, 3)).reshape(z.shape).astype(np.float32)

    @torch.no_grad()
    def score(self, win_batch):
        # 注意：检测器的 p_det/s_det 是 **logits**（训练时直接送 BCEWithLogitsLoss），
        # 必须过 sigmoid 才能与 PhaseNet(softmax)/EQT(sigmoid) 的概率阈值可比。
        x = torch.from_numpy(win_batch).to(self.device)
        p, s = self.m(x)
        return {'det': torch.sigmoid(p).float().cpu().numpy(),
                'detS': torch.sigmoid(s).float().cpu().numpy()}


# ============================ 数据集读取 ============================
def _fix(win, key):
    """(3,T)[E,N,Z] -> (T,3)[Z,N,E] float32"""
    a = np.asarray(win, dtype=np.float32)
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    return np.ascontiguousarray(a[:, ENZ_TO_ZNE])


def iter_stead_ch5(which='events', limit=0):
    """STEAD chunk5 的**事件级均衡**测试集：10,000 条 / 2,500 个事件（每事件 4 条）。

    波形存在单一 dataset 'data'，形状 (N, 6000, 3)，通道序 [E,N,Z]（_fix 会转成
    (T,3)[Z,N,E]）；P/S 到时取索引 CSV。
    """
    base = os.path.join(STEAD, 'chunk5')
    idx = pd.read_csv(os.path.join(base, 'chunk5_event10k_index.csv'))
    if limit:
        idx = idx.iloc[:limit]
    with h5py.File(os.path.join(base, 'chunk5_event10k.hdf5'), 'r') as f:
        arr = f['data']
        for i, r in idx.iterrows():
            name = str(r['trace_name'])
            yield (name, _fix(arr[int(i)][()], name),
                   float(r['p_arrival_sample']), float(r['s_arrival_sample']))


def iter_stead_ch5c(which='events', limit=0):
    """STEAD chunk5 **无事件泄漏**版：10,000 条 / 3,334 个事件（每事件 3 条），
    已排除所有在 chunk2/3/4 出现过的事件。"""
    base = os.path.join(STEAD, 'chunk5')
    idx = pd.read_csv(os.path.join(base, 'chunk5_event10k_clean_index.csv'))
    if limit:
        idx = idx.iloc[:limit]
    with h5py.File(os.path.join(base, 'chunk5_event10k_clean.hdf5'), 'r') as f:
        arr = f['data']
        for i, r in idx.iterrows():
            name = str(r['trace_name'])
            yield (name, _fix(arr[int(i)][()], name),
                   float(r['p_arrival_sample']), float(r['s_arrival_sample']))


def iter_instance(which='events', limit=0):
    if which == 'events':
        h5 = f'{INSTANCE}/INSTANCE_SP/Instance_events_gm_10k.hdf5'
        csv = f'{INSTANCE}/INSTANCE_metadata/metadata_Instance_events_10k.csv'
    else:
        h5 = f'{INSTANCE}/INSTANCE_SP/Instance_noise_1k.hdf5'
        csv = f'{INSTANCE}/INSTANCE_metadata/metadata_Instance_noise_1k.csv'
    df = pd.read_csv(csv, low_memory=False)
    gt = {str(r['trace_name']): (r.get('trace_P_arrival_sample', np.nan),
                                 r.get('trace_S_arrival_sample', np.nan))
          for _, r in df.iterrows()}
    with h5py.File(h5, 'r') as f:
        keys = list(f['data'].keys())
        if limit:
            keys = keys[:limit]
        for k in keys:
            p, s = gt.get(k, (np.nan, np.nan))
            yield k, _fix(f['data'][k][()], k), p, s


def iter_crew(which='events', limit=0):
    h5 = f'{CREW_DIR}/000_CREW.hdf5'
    with h5py.File(h5, 'r') as f:
        keys = list(f['data'].keys())
        if limit:
            keys = keys[:limit]
        for k in tqdm(keys, desc='CREW'):
            d = f['data'][k]
            p = float(d.attrs.get('P_arrival_sample', np.nan))
            s = float(d.attrs.get('S_arrival_sample', np.nan))
            for v in ('Pg_arrival_sample', 'Pn_arrival_sample'):
                pass
            yield k, _fix(d[()], k), p, s


def iter_bench(tag, which='events', limit=0):
    """tag in {'0db','impulsive','noise'}; 0dB/impulsive 的 P 标签固定在采样点 1000。"""
    if which == 'noise' or tag == 'noise':
        h5 = f'{BENCH}/bench0db_noise.hdf5'
    elif tag == 'impulsive':
        h5 = f'{BENCH}/bench0db_impulsive.hdf5'
    else:
        h5 = f'{BENCH}/bench0db.hdf5'
    if not os.path.exists(h5):
        return
    with h5py.File(h5, 'r') as f:
        keys = list(f['data'].keys())
        if limit:
            keys = keys[:limit]
        for k in keys:
            p = 1000.0 if which == 'events' else np.nan
            yield k, _fix(f['data'][k][()], k), p, np.nan


_STEAD_NOISE_CACHE = {}


def _stead_noise_names(limit=0, seed=42, train_n=20000):
    """chunk1 中**未参与训练**的纯噪声 trace 名。

    训练时的抽样（PS_Detection_evsplit.load_noise_data）可精确复现：
        df = read_csv(chunk1.csv); df = df[trace_category=='noise']
        df = df.sample(n=train_n, random_state=seed)
    本函数用同样的随机种子取出这 train_n 条，然后返回其**补集**。
    另加保险：只保留 trace_category=='noise' 且 P/S 到时均为空的记录。
    """
    key = (limit, seed, train_n)
    if key in _STEAD_NOISE_CACHE:
        return _STEAD_NOISE_CACHE[key]
    csv = os.path.join(STEAD, 'chunk1.csv')
    df = pd.read_csv(csv, low_memory=False)
    if 'trace_category' in df.columns:
        df = df[df['trace_category'].astype(str).str.lower() == 'noise']
    if 'p_arrival_sample' in df.columns:
        df = df[df['p_arrival_sample'].isna()]
    if 's_arrival_sample' in df.columns:
        df = df[df['s_arrival_sample'].isna()]
    df = df.reset_index(drop=True)
    n_tr = min(int(train_n), len(df))
    trained = set(df.sample(n=n_tr, random_state=seed)['trace_name'].astype(str))
    held = df[~df['trace_name'].astype(str).isin(trained)].reset_index(drop=True)
    # 固定种子**随机**抽样（避免取文件靠前的记录 -> 时段偏置）
    n_take = int(limit) if limit else min(5000, len(held))
    names = held.sample(n=n_take, random_state=12345)['trace_name'].astype(str).tolist()
    print(f'  [STEAD-DET] chunk1 噪声: 总 {len(df):,} 条, 训练用 {n_tr:,} 条, '
          f'本测试用 {len(names):,} 条（互斥 ✓）')
    _STEAD_NOISE_CACHE[key] = names
    return names


def iter_stead_det_noise(which='noise', limit=0):
    """STEAD chunk1 的未参与训练纯噪声（检测的负类）。"""
    names = _stead_noise_names(limit)
    with h5py.File(os.path.join(STEAD, 'chunk1.hdf5'), 'r') as f:
        for nm in names:
            d = f.get('data/' + str(nm))
            if d is None:
                continue
            yield (str(nm), _fix(d[()], nm), np.nan, np.nan)


def iter_stead_det(which='events', limit=0):
    """STEAD 域内检测测试集：正类 = chunk5 无泄漏事件；负类 = chunk1 未训练噪声。"""
    if which == 'events':
        yield from iter_stead_ch5c('events', limit)
    else:
        yield from iter_stead_det_noise(which, limit)


DATASETS = {
    'INSTANCE': lambda w, lim: iter_instance(w, lim),
    'CREW': lambda w, lim: iter_crew(w, lim),
    'EQS-D-0dB': lambda w, lim: iter_bench('0db', w, lim),
    'EQS-D-imp': lambda w, lim: iter_bench('impulsive', w, lim),
    'STEAD-CH5': lambda w, lim: iter_stead_ch5(w, lim),
    'STEAD-CH5C': lambda w, lim: iter_stead_ch5c(w, lim),
    'STEAD-DET': lambda w, lim: iter_stead_det(w, lim),
}


# ============================ 拾取评估 ============================
def _batched(seq, bs):
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) >= bs:
            yield buf
            buf = []
    if buf:
        yield buf


def eval_picking(model, dataset, split, limit, bs, pre=0.0, save=None):
    """window-level 拾取。

    * P 锚定窗 [p-500, p+2500)：评估 P；窗内若含真值 S 则同时给出 s_err
    * S 锚定窗 [s-500, s+2500)：独立评估 S 拾取质量（s_err_iso）
    """
    W = model.window
    rows = []
    t0 = time.time()
    pack_p, pack_s = [], []

    def flush(pack, kind):
        if not pack:
            return []
        return _run_pick_batch(model, pack, W, kind)

    for name, x, p, s in DATASETS[dataset](split, limit):
        if not _passes_filter(name):
            continue
        T = x.shape[0]
        if T < W:
            continue
        if np.isfinite(p):
            st = max(0, min(int(round(p - 500)), T - W))
            pack_p.append((name, x[st:st + W], st, float(p), float(s)))
        if np.isfinite(s):
            st = max(0, min(int(round(s - 500)), T - W))
            pack_s.append((name, x[st:st + W], st, float(p), float(s)))
        if len(pack_p) >= bs * 4:
            rows += flush(pack_p, 'p')
            pack_p = []
        if len(pack_s) >= bs * 4:
            rows += flush(pack_s, 's')
            pack_s = []
    rows += flush(pack_p, 'p')
    rows += flush(pack_s, 's')

    # 合并：以 P 锚定行为主，把 S 锚定的误差并进来
    df = pd.DataFrame(rows)
    if len(df):
        dp = df[df['kind'] == 'p'].drop(columns=['kind']).copy()
        ds = df[df['kind'] == 's'][['trace', 's_err', 's_pred', 's_true']].copy()
        ds = ds.rename(columns={'s_err': 's_err_iso', 's_pred': 's_pred_iso',
                                's_true': 's_true_iso'})
        out = dp.merge(ds, on='trace', how='left')
    else:
        out = df
    if save and len(out):
        out.to_csv(save, index=False)
    return out, time.time() - t0


def _run_pick_batch(model, items, W, kind='p'):
    if not items:
        return []
    prep = getattr(model, 'prep', None)
    wins = np.stack([(prep(it[1]) if prep else preprocess_40(it[1])) for it in items])
    out = model.score(model.to_input(wins))
    rows = []
    for i, (name, _, start, p, s) in enumerate(items):
        pk = start + int(np.argmax(out['P'][i]))
        sk = (start + int(np.argmax(out['S'][i]))) if 'S' in out else np.nan
        rows.append(dict(kind=kind, trace=name, p_true=p, p_pred=float(pk),
                         p_err=float(pk) - p,
                         s_true=s, s_pred=float(sk),
                         s_err=(float(sk) - s) if np.isfinite(s) else np.nan,
                         s_in_win=bool(np.isfinite(s) and 0 <= (s - start) < W),
                         p_peak=float(out['P'][i].max())))
    return rows


def sum_pick(df, col='p_err', tag='P'):
    if not len(df):
        return {}
    e = df[col].dropna().abs().values
    if not len(e):
        return {}
    return {f'{tag}_n': len(e), f'{tag}_MAE': e.mean(), f'{tag}_med': np.median(e),
            f'{tag}_<=10': (e <= TOL).mean() * 100,
            f'{tag}_<=20': (e <= 20).mean() * 100,
            f'{tag}_<=50': (e <= 50).mean() * 100,
            f'{tag}_>500': (e > 500).mean() * 100}


# ============================ 检测评估 ============================
def _score_traces(model, pack, spans, prep):
    """返回每条记录的 (最大检测分, 该分所在的绝对采样点)。"""
    wins = np.stack([(prep(it[1]) if prep else preprocess_40(it[1])) for it in pack])
    out = model.score(model.to_input(wins))
    # A 判据的分数来自"有无事件"通道；B 判据的到时来自 **P 相位通道**（若有）
    det_chan = out['det'] if 'det' in out else out['P']
    p_chan = out['P'] if 'P' in out else det_chan
    # 双头检测器（如 BiMamba 检测器有 P/S 两个方波头）的自然记录级读出：
    # 逐点取两头最大值，再取记录的全局最大。
    comb_chan = (np.maximum(det_chan, out['detS'])
                 if 'detS' in out else det_chan)
    scores, scores_comb, times, i = [], [], [], 0
    for n in spans:
        sd = det_chan[i:i + n].reshape(-1)        # (n*W,)
        sc = comb_chan[i:i + n].reshape(-1)
        sp = p_chan[i:i + n].reshape(-1)
        jd = int(np.argmax(sd))
        jp = int(np.argmax(sp))
        wd, pd_ = divmod(jd, det_chan.shape[1])
        wp, pp_ = divmod(jp, p_chan.shape[1])
        scores.append(float(sd[jd]))
        scores_comb.append(float(sc.max()))
        times.append(int(pack[i + wp][2]) + int(pp_))
        i += n
    return scores, times, scores_comb


def eval_detection(model, dataset, limit, bs, stride=1000):
    """滑窗扫整条记录；每条记录取检测通道的全局最大值与其位置。

    返回 list of dict(trace, split, p_true, score, t_abs)
      split='events' -> 正样本（该记录含事件，p_true 为其 P 到时）
      split='noise'  -> 负样本（纯噪声，无到时）
    """
    W = model.window
    prep = getattr(model, 'prep', None)
    rec = []
    for split in ('events', 'noise'):
        pack, spans, meta = [], [], []
        for name, x, p, s in DATASETS[dataset](split, limit):
            if not _passes_filter(name):
                continue
            T = x.shape[0]
            if T < W:
                continue
            starts = list(range(0, max(1, T - W + 1), stride)) or [0]
            for st in starts:
                pack.append((name, x[st:st + W], st))
            spans.append(len(starts))
            meta.append((name, float(p) if np.isfinite(p) else np.nan, split))
            if len(pack) >= bs * 4:
                sc, tm, scomb = _score_traces(model, pack, spans, prep)
                rec += [dict(trace=m[0], p_true=m[1], split=m[2], score=s,
                             score_comb=sb, t_abs=tt)
                        for m, s, sb, tt in zip(meta, sc, scomb, tm)]
                pack, spans, meta = [], [], []
        if pack:
            sc, tm, scomb = _score_traces(model, pack, spans, prep)
            rec += [dict(trace=m[0], p_true=m[1], split=m[2], score=s,
                         score_comb=sb, t_abs=tt)
                    for m, s, sb, tt in zip(meta, sc, scomb, tm)]
    return rec


def _crossfit_threshold(rec, tol=10, seed=0):
    """事件级 2 折交叉拟合：在另一折选阈值，在本折评估（无泄漏）。

    返回与 _sweep_oracle 同名的一组键（A_*/B_*），值为两折平均。
    """
    ev = [r for r in rec if r['split'] == 'events']
    nz = [r for r in rec if r['split'] == 'noise']
    if len(ev) < 4 or len(nz) < 4:
        return {}
    sc_e = np.array([r['score'] for r in ev], float)
    sc_n = np.array([r['score'] for r in nz], float)
    te_e = np.array([r['t_abs'] for r in ev], float)
    pt_e = np.array([r['p_true'] for r in ev], float)
    err = np.abs(te_e - pt_e)
    ok_t = np.isfinite(err) & (err <= tol)

    def _eid(x):
        return str(x).split('.')[0]

    g = np.array([_eid(r['trace']) for r in ev])
    ug = np.unique(g)
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    gi = {k: i for i, k in enumerate(ug)}
    ef = np.array([gi[k] % 2 for k in g])
    nf = rng.randint(0, 2, size=len(nz))

    def _prf(tp, fp, fn):
        P = tp / max(tp + fp, 1e-9); R = tp / max(tp + fn, 1e-9)
        return P, R, 2 * P * R / max(P + R, 1e-9)

    def _best(sc, ok, sn, use_ok):
        best = (0.0, 0.0, 0.0, 0.0)
        for th in np.unique(np.r_[sc, sn]):
            ke = sc >= th
            kk = (ke & ok) if use_ok else ke
            fp = int((sn >= th).sum())
            P, R, F = _prf(int(kk.sum()), fp, int((~kk).sum()))
            if F > best[2]:
                best = (P, R, F, float(th))
        return best

    acc = {k: [] for k in ('A_F1', 'A_P', 'A_R', 'A_thr', 'B_F1', 'B_P', 'B_R', 'B_thr')}
    for k in (0, 1):
        # 选阈值：另一折（!= k）；评估：本折（== k）
        sc_sel, ok_sel, sn_sel = sc_e[ef != k], ok_t[ef != k], sc_n[nf != k]
        sc_evl, ok_evl, sn_evl = sc_e[ef == k], ok_t[ef == k], sc_n[nf == k]
        if not len(sc_sel) or not len(sn_sel) or not len(sc_evl) or not len(sn_evl):
            continue
        for use_ok, tag in ((False, 'A'), (True, 'B')):
            _P, _R, _F, th = _best(sc_sel, ok_sel, sn_sel, use_ok)
            ke = sc_evl >= th
            kk = (ke & ok_evl) if use_ok else ke
            P, R, F = _prf(int(kk.sum()), int((sn_evl >= th).sum()), int((~kk).sum()))
            acc[tag + '_F1'].append(F); acc[tag + '_P'].append(P)
            acc[tag + '_R'].append(R); acc[tag + '_thr'].append(th)
    out = {}
    for k, v in acc.items():
        out[k] = float(np.mean(v)) if v else np.nan
    return out


def sweep(rec, tol=10, mode=None):
    """无泄漏包装：默认 mode='crossfit'（事件级 2 折交叉拟合冻结阈值）。

    A_*/B_* 为所选口径的值；oracle 值始终保留在 A_*_oracle / B_*_oracle。
    """
    mode = mode or os.environ.get('EVAL_THRESHOLD_MODE', 'crossfit')
    base = _sweep_oracle(rec, tol)
    if not base:
        return base
    out = dict(base)
    for k in ('A_F1', 'A_P', 'A_R', 'A_thr', 'B_F1', 'B_P', 'B_R', 'B_thr'):
        out[k + '_oracle'] = base.get(k, np.nan)
    out['threshold_mode'] = mode
    if mode == 'oracle':
        return out
    if mode == 'fixed':
        out.update(A_F1=base.get('A_F1_05', np.nan), A_P=base.get('A_P_05', np.nan),
                   A_R=base.get('A_R_05', np.nan), A_thr=0.5,
                   B_F1=base.get('B_F1_05', np.nan), B_P=base.get('B_P_05', np.nan),
                   B_R=base.get('B_R_05', np.nan), B_thr=0.5)
        return out
    cross = _crossfit_threshold(rec, tol)
    if cross:
        out.update(cross)
    return out


def _sweep_oracle(rec, tol=10):
    """A = 记录级二分类；B = 时间匹配（|检测到时-真值| <= tol）。"""
    ev = [r for r in rec if r['split'] == 'events']
    nz = [r for r in rec if r['split'] == 'noise']
    if not ev or not nz:
        return {}
    sc_e = np.array([r['score'] for r in ev])
    te_e = np.array([r['t_abs'] for r in ev], float)
    pt_e = np.array([r['p_true'] for r in ev], float)
    te_err = np.abs(te_e - pt_e)
    ok_t = np.isfinite(te_err) & (te_err <= tol)
    sc_n = np.array([r['score'] for r in nz])

    def _prf(tp, fp, fn):
        P = tp / max(tp + fp, 1e-9); R = tp / max(tp + fn, 1e-9)
        return P, R, 2 * P * R / max(P + R, 1e-9)

    ths = np.unique(np.concatenate([sc_e, sc_n]))
    bestA = dict(F1=-1); bestB = dict(F1=-1)
    for th in ths:
        ke = sc_e >= th
        A = _prf(int(ke.sum()), int((sc_n >= th).sum()), int((~ke).sum()))
        kB = ke & ok_t
        B = _prf(int(kB.sum()), int((sc_n >= th).sum()), int((~kB).sum()))
        if A[2] > bestA['F1']:
            bestA = dict(F1=A[2], P=A[0], R=A[1], thr=float(th))
        if B[2] > bestB['F1']:
            bestB = dict(F1=B[2], P=B[0], R=B[1], thr=float(th))
    # 固定 0.5
    ke = sc_e >= 0.5
    A5 = _prf(int(ke.sum()), int((sc_n >= 0.5).sum()), int((~ke).sum()))
    kB = ke & ok_t
    B5 = _prf(int(kB.sum()), int((sc_n >= 0.5).sum()), int((~kB).sum()))
    # 在 A 的最优阈值处看 B（时间不匹配的惩罚）
    ke = sc_e >= bestA['thr']
    kB = ke & ok_t
    BatA = _prf(int(kB.sum()), int((sc_n >= bestA['thr']).sum()), int((~kB).sum()))
    return dict(
        n_pos=len(ev), n_neg=len(nz),
        A_F1=bestA['F1'], A_P=bestA['P'], A_R=bestA['R'], A_thr=bestA['thr'],
        B_F1=bestB['F1'], B_P=bestB['P'], B_R=bestB['R'], B_thr=bestB['thr'],
        A_F1_05=A5[2], A_P_05=A5[0], A_R_05=A5[1],
        B_F1_05=B5[2], B_P_05=B5[0], B_R_05=B5[1],
        B_at_Athr_F1=BatA[2], B_at_Athr_P=BatA[0], B_at_Athr_R=BatA[1],
        time_med_err=float(np.nanmedian(te_err)),
        time_ok_pct=float(np.nanmean(ok_t) * 100),
        pos_mean=float(sc_e.mean()), neg_mean=float(sc_n.mean()),
    )


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['INSTANCE', 'CREW', 'EQS-D-0dB', 'EQS-D-imp'])
    ap.add_argument('--models', nargs='+',
                    default=['BiMamba-Picker', 'PhaseNet', 'EQTransformer',
                             'PhaseMamba'])
    ap.add_argument('--limit', type=int, default=0, help='每个数据集只用前 N 条（0=全部）')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--detection', action='store_true', help='同时跑 sliding 检测评估')
    ap.add_argument('--keep-csv', default=None,
                    help='只评估该 CSV 里 trace 列的样本（白名单）')
    ap.add_argument('--exclude-csv', default=None,
                    help='剔除该 CSV 里 trace 列的样本（黑名单）')
    ap.add_argument('--skip-picking', action='store_true',
                    help='跳过拾取评估（检测器没有 P 通道时用）')
    ap.add_argument('--stride', type=int, default=1000)
    ap.add_argument('--threshold-mode', choices=['crossfit', 'oracle', 'fixed'],
                    default=os.environ.get('EVAL_THRESHOLD_MODE', 'crossfit'),
                    help='检测阈值口径：crossfit=事件级2折交叉拟合冻结阈值（默认，无泄漏）；'
                         'oracle=测试集选最优（乐观上界）；fixed=固定0.5')
    ap.add_argument('--out', default=os.path.join(HERE, 'eval_all'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  out={a.out}')

    global _KEEP, _DROP
    if a.keep_csv:
        _KEEP = _load_names(a.keep_csv)
        print(f'[filter] 白名单 {len(_KEEP)} 条 <- {a.keep_csv}')
    if a.exclude_csv:
        _DROP = _load_names(a.exclude_csv)
        print(f'[filter] 黑名单 {len(_DROP)} 条 <- {a.exclude_csv}')

    builders = {'BiMamba-Picker': BiMambaPicker, 'PhaseNet': PhaseNetWrap,
                'PhaseMamba': PhaseMambaWrap,
                'EQTransformer': EQTWrap, 'BiMamba-Detector': BiMambaDetector}
    rows = []
    for mname in a.models:
        model = builders[mname](device)
        print(f'\n===== {mname} (window={model.window}) =====', flush=True)
        for ds in ([] if a.skip_picking else a.datasets):
            df, dt = eval_picking(model, ds, 'events', a.limit, a.batch_size,
                                  save=os.path.join(a.out, f'{ds}_{mname}_picking.csv'))
            r = sum_pick(df, 'p_err', 'P')
            rs = sum_pick(df[df['s_in_win']] if len(df) else df, 's_err', 'S')
            rsi = sum_pick(df, 's_err_iso', 'Siso') if 's_err_iso' in df else {}
            cov = (df['s_in_win'].mean() * 100) if len(df) else 0.0
            rec = dict(model=mname, dataset=ds, **r, **rs, **rsi, S_in_win_pct=cov)
            rows.append(rec)
            print(f'  {ds:12s} P: MAE={r.get("P_MAE", np.nan):8.2f} '
                  f'med={r.get("P_med", np.nan):6.1f} <=10={r.get("P_<=10", np.nan):5.1f}% '
                  f'n={r.get("P_n", 0)} | S(窗内比例{cov:.0f}%): '
                  f'MAE={rs.get("S_MAE", np.nan):8.2f} n={rs.get("S_n", 0)} | '
                  f'S(iso): MAE={rsi.get("Siso_MAE", np.nan):8.2f} '
                  f'med={rsi.get("Siso_med", np.nan):6.1f} '
                  f'<=10={rsi.get("Siso_<=10", np.nan):5.1f}% n={rsi.get("Siso_n", 0)} | {dt:.0f}s',
                  flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, 'summary_picking.csv'),
                                      index=False)
        if a.detection:
            if mname == 'BiMamba-Picker':
                print('  [det] 跳过 BiMamba-Picker：其输出是窗内概率密度，'
                      '不能当逐点检测概率（检测用 BiMamba-Detector）', flush=True)
            for ds in ([] if mname == 'BiMamba-Picker' else
                       [d for d in a.datasets if d != 'CREW']):
                rec = eval_detection(model, ds, a.limit, a.batch_size, a.stride)
                s = sweep(rec, tol=10)
                if s:
                    pd.DataFrame(rec).to_csv(
                        os.path.join(a.out, f'detscore_{ds}_{mname}.csv'), index=False)
                    print(f'  [det]{ds:12s} A: F1={s["A_F1"]:.4f} P={s["A_P"]:.4f} '
                          f'R={s["A_R"]:.4f} [@{(s.get("threshold_mode") or "?").upper()} thr={s.get("A_thr", float("nan")):.2e}] '
                          f'| oracle: F1={s.get("A_F1_oracle", float("nan")):.4f} '
                          f'| B(0.1s): F1={s["B_F1"]:.4f} '
                          f'P={s["B_P"]:.4f} R={s["B_R"]:.4f} | '
                          f'B@Athr F1={s["B_at_Athr_F1"]:.4f} | '
                          f'时间误差中位={s["time_med_err"]:.0f} '
                          f'<=10点占比={s["time_ok_pct"]:.1f}%', flush=True)
                    f = os.path.join(a.out, 'summary_detection_AB.csv')
                    pd.DataFrame([dict(model=mname, dataset=ds, **s)]).to_csv(
                        f, mode='a', header=not os.path.exists(f), index=False)
    print('\n完成。结果目录:', a.out)


if __name__ == '__main__':
    main()

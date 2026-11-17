# -*- coding: utf-8 -*-
"""门控 + 滑窗投票定位（Gate → Vote/Hough）。

用户提出的流水线：
  1. 检测器判"该记录是否存在事件"（记录级二值判决，带噪声拒绝）
  2. 若存在 -> 对该窗口做**滑窗投票/Hough** 定位（不用检测器的单点锚点）

与旧的两阶段（检测器定锚 -> 拾取器单窗精化）的区别：
  - 旧法：anchor 一个点 -> 窗 [anchor-500, anchor+2500)，若锚点偏就漏 P ✗
  - 新法：检测器只当**门**；定位交给跨窗共识 ✓（锚点质量不再是瓶颈 ✓）

为了一次前向同时评估两种区域口径：
  - FULL : 全网记录投票（= 纯 vote/Hough，检测器仅当门）
  - WIN  : 只在检测器命中区域 [t_abs-BACK, t_abs+FWD] 内投票（用户的设计）

环境变量：
  EVAL_PICKER_CKPT    拾取器权重
  EVAL_DETSCORE_DIR   检测分数目录
  EVAL_TAG            输出后缀
  GATE_TAU            门控阈值（不设则用"交叉拟合冻结阈值"自动估）
  BACK / FWD          窗口模式的回看/前瞻余量（默认 4000 / 2000）
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_evsplit_models as E   # noqa: E402

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


OUT = os.path.join(HERE, 'eval_all')
DETDIR = os.environ.get('EVAL_DETSCORE_DIR', OUT)
TAG = os.environ.get('EVAL_TAG', '')
BACK = int(os.environ.get('BACK', 4000))
FWD = int(os.environ.get('FWD', 2000))
STRIDE = int(os.environ.get('STRIDE', 250))
BINW = int(os.environ.get('BINW', 10))
BS = int(os.environ.get('BS', 256))


def prf(tp, fp, fn):
    P = tp / max(tp + fp, 1e-9); R = tp / max(tp + fn, 1e-9)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def best_thr(ev, nz):
    best = (0.0, 0.0)
    for th in np.unique(np.r_[ev, nz]):
        _p, _r, F = prf(int((ev >= th).sum()), int((nz >= th).sum()), int((ev < th).sum()))
        if F > best[0]:
            best = (F, float(th))
    return best[1]


def eid(t):
    return str(t).split('.')[0]


def crossfit_tau(scores, splits, traces):
    """事件级 2 折交叉拟合的门控阈值（无泄漏）。"""
    ev = scores[splits == 'events']; nz = scores[splits == 'noise']
    if not len(ev) or not len(nz):
        return np.nan, 0.0
    g = np.array([eid(t) for t in traces[splits == 'events']])
    ug = np.unique(g); rng = np.random.RandomState(0); rng.shuffle(ug)
    gs = {x: i for i, x in enumerate(ug)}
    ef = np.array([gs[x] % 2 for x in g])
    nf = rng.randint(0, 2, size=len(nz))
    ths = []
    for k in (0, 1):
        ea, na = ev[ef == k], nz[nf == k]
        eb, nb = ev[ef != k], nz[nf != k]
        if not len(ea) or not len(na):
            continue
        th = best_thr(ea, na)
        ths.append(th)
    th_o = best_thr(ev, nz)
    _P, _R, F_o = prf(int((ev >= th_o).sum()), int((nz >= th_o).sum()),
                      int((ev < th_o).sum()))
    return (float(np.mean(ths)) if ths else float(th_o)), float(F_o)


def one_record(picker, x, region, device):
    """滑窗投票：返回 (FULL 众数, FULL 集中度, WIN 众数, WIN 集中度)。"""
    W = picker.window
    T = x.shape[0]
    starts = list(range(0, max(1, T - W + 1), STRIDE)) or [0]
    pos, wins, pos_s = [], [], []
    for i in range(0, len(starts), BS):
        idx = starts[i:i + BS]
        wins = np.stack([E.preprocess_40(x[s:s + W]) for s in idx])
        out = picker.score(picker.to_input(wins))
        am = np.asarray(out['P']).argmax(axis=1)
        pos.extend((np.array(idx) + am).tolist())
        if 'S' in out:
            am_s = np.asarray(out['S']).argmax(axis=1)
            pos_s.extend((np.array(idx) + am_s).tolist())
    pos = np.array(pos, dtype=np.int64)
    pos_s = np.array(pos_s, dtype=np.int64)
    if not len(pos):
        return None, np.nan, None, np.nan, None, None, None, None

    def mode(v):
        """返回 (箱中心, 集中度, 亚箱精化位置)。

        亚箱精化：取众数箱内所有票的**中位数**，把定位分辨率从 BINW 降到 1 采样点。
        """
        if not len(v):
            return None, np.nan, None
        lo = int(v.min())
        b = (v - lo) // BINW
        h = np.bincount(b)
        bi = int(h.argmax())
        centre = lo + bi * BINW + BINW // 2
        sel = v[b == bi]
        refined = float(np.median(sel)) if len(sel) else float(centre)
        return centre, float(h[bi] / len(v)), refined

    full_p, full_c, full_ex = mode(pos)
    lo_b, hi_b = region
    sub = pos[(pos >= lo_b) & (pos <= hi_b)]
    win_p, win_c, win_ex = mode(sub)
    # ---- S 波：同一套投票，仅换成 S 通道 ----
    s_full, s_full_c, s_full_ex = mode(pos_s) if len(pos_s) else (None, np.nan, None)
    sub_s = pos_s[(pos_s >= lo_b) & (pos_s <= hi_b)] if len(pos_s) else pos_s
    win_s, win_s_c, win_s_ex = mode(sub_s) if len(sub_s) else (None, np.nan, None)
    return (full_p, full_c, win_p, win_c, full_ex, win_ex,
            s_full_ex, win_s_ex)


def run(ds, picker, device, tau=None):
    _KEY = os.environ.get('EVAL_DET_KEY', 'BiMamba-Detector')
    f = os.path.join(DETDIR, f'detscore_{ds}_{_KEY}.csv')
    if not os.path.exists(f):
        print('缺文件', f); return None, None
    det = pd.read_csv(f)
    info = {str(r['trace']): (float(r['score']), int(r['t_abs']), r['split'],
                              float(r['p_true']) if np.isfinite(r['p_true']) else np.nan)
            for _, r in det.iterrows()}
    sc = det['score'].values
    sp = det['split'].values
    tr = det['trace'].values
    if tau is None:
        tau, F_or = crossfit_tau(sc, sp, tr)
        print(f'  [门控阈值] 交叉拟合 τ={tau:.3e}（oracle F1={F_or:.4f}）')

    rows = []
    for split in ('events', 'noise'):
        for name, x, p, s in E.DATASETS[ds](split, 0):
            k = str(name)
            if k not in info:
                continue
            score, anc, sp_, pt = info[k]
            passed = score >= tau
            rec = dict(trace=k, split=sp_, score=score, passed=bool(passed),
                       anchor=anc, p_true=pt)
            if passed and x.shape[0] >= picker.window:
                lo = max(0, anc - BACK)
                hi = min(x.shape[0], anc + FWD)
                fp_, fc_, wp_, wc_, fx_, wx_, s_fx_, s_wx_ = one_record(picker, x, (lo, hi), device)
                rec.update(full_pick=fp_, full_conc=fc_, win_pick=wp_, win_conc=wc_,
                           full_pick_ex=fx_, win_pick_ex=wx_,
                           s_full_pick=s_fx_, s_win_pick=s_wx_,
                           s_true=float(s) if np.isfinite(s) else np.nan)
            rows.append(rec)
        print(f'    {split} 完成（{len(rows)} 条）', flush=True)
    d = pd.DataFrame(rows)

    def summarize(mode_col):
        # mode_col 传入 *_ex（亚箱精化）
        ev = d[(d.split == 'events') & d.passed & d[mode_col].notna()]
        nz = d[(d.split == 'noise') & d.passed]
        err = np.abs(ev[mode_col].values - ev['p_true'].values)
        return dict(n_detect_pos=int(d.passed.sum()), n_ev=len(ev), n_fp=len(nz),
                    med=float(np.nanmedian(err)) if len(err) else np.nan,
                    ma=float(np.nanmean(err)) if len(err) else np.nan,
                    ok10=float(np.nanmean(err <= 10) * 100) if len(err) else np.nan,
                    ok50=float(np.nanmean(err <= 50) * 100) if len(err) else np.nan,
                    ok100=float(np.nanmean(err <= 100) * 100) if len(err) else np.nan)

    ev_a = d[d.split == 'events']; nz_a = d[d.split == 'noise']
    ke = ev_a.passed.values
    A = prf(int(ke.sum()), int(nz_a.passed.sum()), int((~ke).sum()))
    res = dict(dataset=ds, tau=tau, n_pos=len(ev_a), n_neg=len(nz_a),
               A_P=A[0], A_R=A[1], A_F1=A[2])
    for c, tag in (('full_pick_ex', 'FULL'), ('win_pick_ex', 'WIN')):
        s = summarize(c)
        for k2, v in s.items():
            res[f'{tag}_{k2}'] = v
    return res, d


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)
    _PM = os.environ.get('PICKER_MODEL', 'bimamba').lower()
    _CLS = {'bimamba': 'BiMambaPicker', 'phasenet': 'PhaseNetWrap',
            'eqt': 'EQTWrap', 'eqtransformer': 'EQTWrap',
            'phasemamba': 'PhaseMambaWrap'}.get(_PM, 'BiMambaPicker')
    print(f'[拾取器] {_CLS}')
    picker = getattr(E, _CLS)(device)
    print(f'[配置] 门控余量 BACK={BACK} FWD={FWD} stride={STRIDE} binw={BINW}')
    rows = []
    _DS = os.environ.get('EVAL_DATASETS', 'INSTANCE,EQS-D-0dB,EQS-D-imp')
    for ds in [x.strip() for x in _DS.split(',') if x.strip()]:
        r, d = run(ds, picker, device)
        if r is None:
            continue
        d.to_csv(os.path.join(OUT, f'gatevote_{ds}{TAG}.csv'), index=False)
        rows.append(r)
        print(f"  {ds:11s} A: F1={r['A_F1']:.4f} P={r['A_P']:.4f} R={r['A_R']:.4f} | "
              f"FULL 中位={r['FULL_med']:7.1f} ≤10={r['FULL_ok10']:5.1f}% | "
              f"WIN 中位={r['WIN_med']:7.1f} ≤10={r['WIN_ok10']:5.1f}%", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, f'summary_gatevote{TAG}.csv'), index=False)
    print('\n保存:', os.path.join(OUT, f'summary_gatevote{TAG}.csv'))

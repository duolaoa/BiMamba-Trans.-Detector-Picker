# -*- coding: utf-8 -*-
"""消融实验的 σ（有符号残差的标准差，population ddof=0）。
   完全沿用 abl_seeds.py 的口径：
     · 5 个 25/75 划分种子
     · 只用 test 侧、且通过门控的记录
     · 误差 = win_pick − p_true（有符号），σ = std(e)
   另外同时给出 MAE 以便核对与 ABLATION_SEEDS.csv 一致。
"""
import pandas as pd, numpy as np, os

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


SEEDS = [0, 1, 2, 3, 4]
SETS = {
 'STEAD-DET': [('All', 'c2_pn197k_w6000v3_detpn197k'),
               ('w/o BiMamba', 'abl6000_nomamba'),
               ('w/o Transformer', 'abl6000_notrans'),
               ('w/o LFEM1D', 'abl6000_nolfem')],
 'INSTANCE':  [('All', 'c2_pn197k_offaug_detpn197k'),
               ('w/o BiMamba', 'abl6000_nomamba'),
               ('w/o Transformer', 'abl6000_notrans'),
               ('w/o LFEM1D', 'abl6000_nolfem')],
}
DET = {'STEAD-DET': 'eval_all_steaddet_det_pn197k/detscore_STEAD-DET_BiMamba-Detector.csv',
       'INSTANCE':  'eval_all_det_det_pn197k/detscore_INSTANCE_BiMamba-Detector.csv'}

def eid(t): return str(t).split('.')[0]
def prf(TP, FP, FN):
    P = TP/max(TP+FP,1e-12); R = TP/max(TP+FN,1e-12)
    return P, R, 2*P*R/max(P+R,1e-12)
def best_thr(a, b):
    a = np.asarray(a,float); b = np.asarray(b,float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    qs = np.quantile(np.unique(np.r_[a,b]), np.linspace(0.005,0.995,400))
    best = (-1.0, float(qs[0]))
    for th in qs:
        TP=int((a>=th).sum()); FP=int((b>=th).sum())
        _P,_R,F = prf(TP,FP,len(a)-TP)
        if F>best[0]: best=(F,float(th))
    return best[1]

rows = []
for ds, variants in SETS.items():
    sh = pd.read_csv(DET[ds])
    ev_all = sh[sh.split=='events'].trace.values
    nz_all = sh[sh.split=='noise'].trace.values
    ug = np.unique([eid(t) for t in ev_all])
    for seed in SEEDS:
        rng = np.random.RandomState(seed); u2 = ug.copy(); rng.shuffle(u2)
        cal_g = set(u2[:max(1,int(round(0.25*len(u2))))])
        rng2 = np.random.RandomState(seed)
        perm = rng2.permutation(len(nz_all))
        cal_n = set(nz_all[perm[:max(1,int(round(0.25*len(nz_all))))]])
        cE = np.array([eid(t) in cal_g for t in ev_all])
        cN = np.array([t in cal_n for t in nz_all])
        tau = best_thr(sh[sh.split=='events'].score.values[cE],
                       sh[sh.split=='noise'].score.values[cN])
        for name, tag in variants:
            f = f'eval_all/gatevote_{ds}_{tag}.csv'
            if not os.path.exists(f):
                print('  missing', f); continue
            d = pd.read_csv(f)
            ev = d[d.split=='events'].copy()
            teE = ~np.array([eid(t) in cal_g for t in ev.trace])
            passed = ev.score.values >= tau
            sel = teE & passed
            r = dict(dataset=ds, seed=seed, variant=name, n_gated=int(sel.sum()))
            pe = (ev.win_pick - ev.p_true).values[sel]
            pe = pe[np.isfinite(pe)]
            if len(pe):
                r['P_MAE'] = float(np.abs(pe).mean()/100)
                r['P_sd']  = float(np.std(pe)/100)          # population sd of signed err
                r['P_bias'] = float(np.mean(pe)/100)
                r['nP'] = len(pe)
            if 's_win_pick' in ev.columns:
                se = (ev.s_win_pick - ev.s_true).values[sel]
                se = se[np.isfinite(se)]
                if len(se):
                    r['S_MAE'] = float(np.abs(se).mean()/100)
                    r['S_sd']  = float(np.std(se)/100)
                    r['S_bias'] = float(np.mean(se)/100)
                    r['nS'] = len(se)
            rows.append(r)

t = pd.DataFrame(rows)
t.to_csv('eval_all/ABLATION_SIGMA.csv', index=False)
print('saved eval_all/ABLATION_SIGMA.csv  rows =', len(t))

for ds in t.dataset.unique():
    s = t[t.dataset == ds]
    print(f'\n{"="*112}')
    print(f'== {ds} · 5 个种子，mean ± std（总体 sd）==')
    print(f'{"="*112}')
    cols = [c for c in ('P_MAE','P_sd','P_bias','S_MAE','S_sd','S_bias') if c in s.columns]
    print('   {:16s} '.format('变体') + ' '.join(f'{c:>15s}' for c in cols))
    order = [v for v, _ in SETS[ds]]
    for v in order:
        g = s[s.variant == v]
        if g.empty: continue
        cells = []
        for c in cols:
            m = g[c].mean(); sd = g[c].std(ddof=0)
            cells.append(f'{m:6.3f}±{sd:4.3f}'.rjust(15))
        print('   {:16s} '.format(v) + ' '.join(cells))

# -*- coding: utf-8 -*-
"""消融实验的**划分种子稳健性**检验（不是挑种子 ✗）。
   对 5 个不同的 25/75 划分种子，重复计算消融指标，报告 mean ± std。
   全部种子都报告，不做挑选。
"""
import pandas as pd, numpy as np, os

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


SEEDS = [0, 1, 2, 3, 4]
SETS = {
 'STEAD-DET': [('All','c2_pn197k_w6000v3_detpn197k'),
               ('w/o BiMamba','abl6000_nomamba'),
               ('w/o Transformer','abl6000_notrans'),
               ('w/o LFEM1D','abl6000_nolfem')],
 'INSTANCE':  [('All','c2_pn197k_offaug_detpn197k'),
               ('w/o BiMamba','abl6000_nomamba'),
               ('w/o Transformer','abl6000_notrans'),
               ('w/o LFEM1D','abl6000_nolfem')],
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
                continue
            d = pd.read_csv(f)
            ev = d[d.split=='events'].copy(); nz = d[d.split=='noise'].copy()
            teE = ~np.array([eid(t) in cal_g for t in ev.trace])
            teN = ~np.array([t in cal_n for t in nz.trace])
            n_ev = int(teE.sum()); n_no = int(teN.sum())
            passed = ev.score.values[teE] >= tau
            TP = int(passed.sum()); FP = int((nz.score.values[teN] >= tau).sum())
            r = dict(dataset=ds, seed=seed, variant=name, tau=tau, n_ev=n_ev,
                     F1=prf(TP, FP, n_ev-TP)[2])
            pe = (ev.win_pick - ev.p_true).abs().values[teE][passed]
            pe = pe[np.isfinite(pe)]
            if len(pe):
                r['P_hit05'] = (pe<=50).mean()*100
                r['P_hit01'] = (pe<=10).mean()*100
                r['P_MAE'] = pe.mean()/100
            if 's_win_pick' in ev.columns and ev.s_win_pick.notna().any():
                se = (ev.s_win_pick - ev.s_true).abs().values[teE][passed]
                se = se[np.isfinite(se)]
                if len(se):
                    r['S_hit05'] = (se<=50).mean()*100
                    r['S_hit01'] = (se<=10).mean()*100
                    r['S_MAE'] = se.mean()/100
            rows.append(r)

t = pd.DataFrame(rows)
t.to_csv('eval_all/ABLATION_SEEDS.csv', index=False)
print('saved eval_all/ABLATION_SEEDS.csv  rows =', len(t))
for ds in t.dataset.unique():
    s = t[t.dataset==ds]
    print(f'\n{"="*100}')
    print(f'== {ds} · 5 个划分种子，mean ± std ==')
    print(f'{"="*100}')
    cols = [c for c in ('F1','P_hit05','P_hit01','P_MAE','S_hit05','S_hit01','S_MAE') if c in s.columns]
    hdr = '   {:16s} '.format('变体') + ' '.join(f'{c:>16s}' for c in cols)
    print(hdr)
    order = [v for v, _ in SETS[ds]]
    for v in order:
        g = s[s.variant == v]
        if g.empty: continue
        cells = []
        for c in cols:
            m = g[c].mean(); sd = g[c].std(ddof=0)
            cells.append(f'{m:7.3f}±{sd:5.3f}'.rjust(16))
        print('   {:16s} '.format(v) + ' '.join(cells))
    # 与 All 的差
    if 'All' in order:
        base = s[s.variant=='All']
        print(f'\n   —— 相对 All 的差值（均值）——')
        for v in order[1:]:
            g = s[s.variant == v]
            if g.empty: continue
            d = []
            for c in cols:
                d.append(f'{g[c].mean()-base[c].mean():+8.3f}')
            print('   {:16s} '.format(v) + ' '.join(f'{x:>16s}' for x in d))

# -*- coding: utf-8 -*-
"""消融表在 25% 校准 / 75% 测试口径下的重算（检测器共用 → 方案 A）。"""
import pandas as pd, numpy as np, os

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


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

SETS = {
 'STEAD-DET': [('All','abl6000_full'),('w/o BiMamba','abl6000_nomamba'),
               ('w/o Transformer','abl6000_notrans'),('w/o LFEM1D','abl6000_nolfem')],
 'INSTANCE':  [('All','abl6000_full'),('w/o BiMamba','abl6000_nomamba'),
               ('w/o Transformer','abl6000_notrans'),('w/o LFEM1D','abl6000_nolfem')],
}
# 检测器分数（共用）
DET = {'STEAD-DET': 'eval_all_steaddet_det_pn197k/detscore_STEAD-DET_BiMamba-Detector.csv',
       'INSTANCE':  'eval_all_det_det_pn197k/detscore_INSTANCE_BiMamba-Detector.csv'}

rows = []
for ds, variants in SETS.items():
    sh = pd.read_csv(DET[ds])
    ev_tr = sh[sh.split=='events'].trace.values
    nz_tr = sh[sh.split=='noise'].trace.values
    ug = np.unique([eid(t) for t in ev_tr])
    rng = np.random.RandomState(0); rng.shuffle(ug)
    cal_g = set(ug[:max(1,int(round(0.25*len(ug))))])
    rng2 = np.random.RandomState(0)
    perm = rng2.permutation(len(nz_tr))
    cal_n = set(nz_tr[perm[:max(1,int(round(0.25*len(nz_tr))))]])
    isE = np.array([eid(t) in cal_g for t in sh[sh.split=='events'].trace])
    isN = np.array([t in cal_n for t in sh[sh.split=='noise'].trace])
    tau = best_thr(sh[sh.split=='events'].score.values[isE],
                   sh[sh.split=='noise'].score.values[isN])
    for name, tag in variants:
        f = f'eval_all/gatevote_{ds}_{tag}.csv'
        if not os.path.exists(f):
            print(f'  跳过 {ds}/{name}'); continue
        d = pd.read_csv(f)
        ev = d[d.split=='events'].copy(); nz = d[d.split=='noise'].copy()
        cE = np.array([eid(t) in cal_g for t in ev.trace])
        cN = np.array([t in cal_n for t in nz.trace])
        teE, teN = ~cE, ~cN
        n_ev = int(teE.sum()); n_no = int(teN.sum())
        passed = ev.score.values[teE] >= tau
        TP = int(passed.sum()); FP = int((nz.score.values[teN] >= tau).sum())
        FN = n_ev - TP
        P_, R_, F_ = prf(TP, FP, FN)
        r = dict(dataset=ds, variant=name, tau=tau, n_ev=n_ev, n_no=n_no,
                 TP=TP, TN=n_no-FP, FP=FP, FN=FN, Pr=P_, Re=R_, F1=F_)
        pe = (ev.win_pick - ev.p_true).abs().values[teE]
        pe = pe[passed]; pe = pe[np.isfinite(pe)]
        if len(pe):
            for TOL, k in ((50,'05'), (10,'01')):
                r[f'P_hit{k}'] = (pe<=TOL).mean()*100
            r['P_MAE'] = pe.mean()/100; r['P_SD'] = pe.std(ddof=0)/100
        if 's_win_pick' in ev.columns and ev.s_win_pick.notna().any():
            se = (ev.s_win_pick - ev.s_true).abs().values[teE]
            se = se[passed]; se = se[np.isfinite(se)]
            if len(se):
                for TOL, k in ((50,'05'), (10,'01')):
                    r[f'S_hit{k}'] = (se<=TOL).mean()*100
                r['S_MAE'] = se.mean()/100; r['S_SD'] = se.std(ddof=0)/100
        rows.append(r)

t = pd.DataFrame(rows)
t.to_csv('eval_all/ABLATION_25_75.csv', index=False)
print('saved eval_all/ABLATION_25_75.csv  rows =', len(t))
for ds in t.dataset.unique():
    s = t[t.dataset==ds]
    print(f'\n== {ds} · 消融（25% 校准 / 75% 测试，共用检测器）==  τ = {s.tau.iloc[0]:.3e}')
    print('   test: {} 事件 / {} 噪声'.format(int(s.n_ev.iloc[0]), int(s.n_no.iloc[0])))
    cols = ['variant','TP','FN','F1']
    extra = [c for c in ('P_hit05','P_hit01','P_MAE','P_SD','S_hit05','S_hit01','S_MAE','S_SD') if c in s.columns]
    print('   ' + '  '.join(f'{c:>14s}' for c in cols+extra))
    for _, r in s.iterrows():
        vals = [str(r.variant), str(int(r.TP)), str(int(r.FN)), f'{r.F1:.4f}']
        for c in extra:
            vals.append(f'{r[c]:.3f}' if pd.notna(r.get(c)) else '—')
        print('   ' + '  '.join(f'{v:>14s}' for v in vals))

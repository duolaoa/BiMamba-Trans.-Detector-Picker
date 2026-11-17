# -*- coding: utf-8 -*-
"""跨域阈值敏感性：按「门控通过率」遍历 τ（避免分数饱和导致的退化网格）。
   τ 由噪声侧的通过率反推：给定噪声通过率 p_n，取 τ = quantile(noise, 1-p_n)。
输出 TAU_SWEEP_Q.csv 与两幅子图。
"""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


SETS = [('STEAD-DET', 'eval_all/gatevote_STEAD-DET_c2_pn197k_w6000v3_detpn197k.csv'),
        ('INSTANCE',  'eval_all/gatevote_INSTANCE_c2_pn197k_offaug_detpn197k.csv'),
        ('EQS-D-0dB', 'eval_all/gatevote_EQS-D-0dB_c2_pn197k_w6000v3_detpn197k.csv'),
        ('EQS-D-imp', 'eval_all/gatevote_EQS-D-imp_c2_pn197k_w6000v3_detpn197k.csv')]
PNOISE = [0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0001]

def prf(TP, FP, FN):
    P = TP/max(TP+FP,1e-12); R = TP/max(TP+FN,1e-12)
    return P, R, 2*P*R/max(P+R,1e-12)
def f1tol(err, TOL, FP):
    TP = int((err <= TOL).sum()); FN = len(err)-TP
    P = TP/max(TP+FP,1e-12); R = TP/max(TP+FN,1e-12)
    return 2*P*R/max(P+R,1e-12)

rows = []
for ds, f in SETS:
    if not os.path.exists(f):
        print('skip', ds); continue
    d = pd.read_csv(f)
    ev = d[d.split=='events'].copy(); nz = d[d.split=='noise'].copy()
    sc_ev, sc_nz = ev.score.values, nz.score.values
    has_s = 's_win_pick' in d.columns
    p_err = (ev.win_pick - ev.p_true).abs()
    s_err = (ev.s_win_pick - ev.s_true).abs() if has_s else None
    for pn in PNOISE:
        th = float(np.quantile(sc_nz, 1.0 - pn))
        passed = sc_ev >= th
        TP = int(passed.sum()); FP = int((sc_nz >= th).sum()); FN = len(sc_ev)-TP
        P_, R_, F_ = prf(TP, FP, FN)
        pe = p_err[passed].dropna().values
        r = dict(dataset=ds, noise_pass=pn, tau=th, Pr=P_, Re=R_, F1=F_, n_pass=TP, FP=FP,
                 P_F1_05=f1tol(pe, 50, FP) if len(pe) else np.nan,
                 P_F1_01=f1tol(pe, 10, FP) if len(pe) else np.nan,
                 P_MAE=pe.mean()/100 if len(pe) else np.nan)
        if has_s:
            se = s_err[passed].dropna().values
            r['S_F1_05'] = f1tol(se, 50, FP) if len(se) else np.nan
            r['S_F1_01'] = f1tol(se, 10, FP) if len(se) else np.nan
            r['S_MAE'] = se.mean()/100 if len(se) else np.nan
        rows.append(r)

t = pd.DataFrame(rows)
t.to_csv('eval_all/TAU_SWEEP_Q.csv', index=False)

for ds in t.dataset.unique():
    s = t[t.dataset == ds]
    print(f'\n■ {ds}')
    print('   {:>9s} {:>10s} {:>7s} {:>7s} {:>7s} {:>7s} | {:>8s} {:>8s}'.format(
        '噪声通过率', 'τ', 'Pr', 'Re', 'F1', 'FP', 'P_F1@.5', 'P_F1@.1'), end='')
    if 'S_F1_05' in s.columns and s.S_F1_05.notna().any():
        print(' {:>8s} {:>8s}'.format('S_F1@.5', 'S_F1@.1'))
    else:
        print()
    for _, r in s.iterrows():
        line = '   {:9.4f} {:10.3e} {:7.4f} {:7.4f} {:7.4f} {:7d} | {:8.4f} {:8.4f}'.format(
            r.noise_pass, r.tau, r.Pr, r.Re, r.F1, r.FP, r.P_F1_05, r.P_F1_01)
        if 'S_F1_05' in s.columns and s.S_F1_05.notna().any():
            line += ' {:8.4f} {:8.4f}'.format(r.S_F1_05, r.S_F1_01)
        print(line)

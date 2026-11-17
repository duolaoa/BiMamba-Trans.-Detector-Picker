# -*- coding: utf-8 -*-
"""跨域门控阈值 τ 敏感性分析：从已保存的逐条 score 重新施加不同 τ。"""
import pandas as pd, numpy as np, os

SETS = [('STEAD-DET', 'eval_all/gatevote_STEAD-DET_c2_pn197k_w6000v3_detpn197k.csv'),
        ('INSTANCE',  'eval_all/gatevote_INSTANCE_c2_pn197k_offaug_detpn197k.csv'),
        ('EQS-D-0dB', 'eval_all/gatevote_EQS-D-0dB_c2_pn197k_w6000v3_detpn197k.csv'),
        ('EQS-D-imp', 'eval_all/gatevote_EQS-D-imp_c2_pn197k_w6000v3_detpn197k.csv')]

def prf(TP, FP, FN):
    P = TP/max(TP+FP, 1e-12); R = TP/max(TP+FN, 1e-12)
    return P, R, 2*P*R/max(P+R, 1e-12)

def f1tol(err, TOL, FP):
    TP = int((err <= TOL).sum()); FN = len(err) - TP
    P = TP/max(TP+FP, 1e-12); R = TP/max(TP+FN, 1e-12)
    return 2*P*R/max(P+R, 1e-12)

os.makedirs('eval_all/figs', exist_ok=True)
ALL = {}
for ds, f in SETS:
    if not os.path.exists(f):
        print(f'  跳过 {ds}'); continue
    d = pd.read_csv(f)
    ev = d[d.split == 'events'].copy(); nz = d[d.split == 'noise'].copy()
    sc_ev, sc_nz = ev.score.values, nz.score.values
    has_s = 's_win_pick' in d.columns
    p_err = (ev.win_pick - ev.p_true).abs()
    s_err = (ev.s_win_pick - ev.s_true).abs() if has_s else None

    th_best, f1_best = None, -1
    for th in np.quantile(np.unique(np.r_[sc_ev, sc_nz]), np.linspace(0.01, 0.99, 400)):
        TP = int((sc_ev >= th).sum()); FP = int((sc_nz >= th).sum())
        _P, _R, F = prf(TP, FP, len(sc_ev) - TP)
        if F > f1_best:
            f1_best, th_best = F, float(th)
    grid = th_best * np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0, 10.0, 20.0])

    rows = []
    for th in grid:
        passed = sc_ev >= th
        TP = int(passed.sum()); FP = int((sc_nz >= th).sum()); FN = len(sc_ev) - TP
        P_, R_, F_ = prf(TP, FP, FN)
        pe = p_err[passed].dropna().values
        r = dict(dataset=ds, tau=float(th), tau_rel=float(th/th_best), Pr=P_, Re=R_, F1=F_,
                 n_pass=TP, n_noise_pass=FP,
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
    t.to_csv(f'eval_all/TAU_SWEEP_{ds}.csv', index=False)
    ALL[ds] = t
    print(f'\n■ {ds}   最优 τ = {th_best:.4e}   (S 可用: {has_s})')
    hdr = '   {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>8s} {:>8s}'.format(
        'τ/τ*', 'Pr', 'Re', 'F1', 'n通过', 'P_F1@.5', 'P_F1@.1')
    if has_s:
        hdr += ' {:>8s} {:>8s}'.format('S_F1@.5', 'S_F1@.1')
    print(hdr)
    for _, r in t.iterrows():
        line = '   {:7.2f} {:7.4f} {:7.4f} {:7.4f} {:7d} {:8.4f} {:8.4f}'.format(
            r.tau_rel, r.Pr, r.Re, r.F1, r.n_pass, r.P_F1_05, r.P_F1_01)
        if has_s:
            line += ' {:8.4f} {:8.4f}'.format(r.S_F1_05, r.S_F1_01)
        if abs(r.tau_rel - 1.0) < 1e-9:
            line += '  ← 最优'
        print(line)

print()
print('=' * 100)
print('τ 在最优值附近扰动时，各指标的最大波动')
print('=' * 100)
print('   {:12s} {:>12s} {:>12s} {:>14s} {:>14s}'.format(
    '数据集', 'F1±20%', 'F1±50%', 'P_F1@.5 ±20%', 'P_F1@.5 ±50%'))
for ds, t in ALL.items():
    for a, b, lab in ((0.8, 1.2, '20'), (0.5, 1.5, '50')):
        sub = t[(t.tau_rel >= a) & (t.tau_rel <= b)]
        pass
    s20 = t[(t.tau_rel >= 0.8) & (t.tau_rel <= 1.2)]
    s50 = t[(t.tau_rel >= 0.5) & (t.tau_rel <= 1.5)]
    print('   {:12s} {:>12.4f} {:>12.4f} {:>14.4f} {:>14.4f}'.format(
        ds, s20.F1.max()-s20.F1.min(), s50.F1.max()-s50.F1.min(),
        s20.P_F1_05.max()-s20.P_F1_05.min(), s50.P_F1_05.max()-s50.P_F1_05.min()))

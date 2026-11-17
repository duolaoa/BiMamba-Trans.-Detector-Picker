# -*- coding: utf-8 -*-
"""方案 A + B：25% 校准 τ，75% 测试。A=共用检测器；B=各模型自有检测分数。"""
import pandas as pd, numpy as np, os, glob

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


DS = ['INSTANCE', 'EQS-D-0dB', 'EQS-D-imp']
MODELS = [
    ('BiMamba-Trans.', 'eval_all_pm_w6000_all',
     {'INSTANCE':  'eval_all/gatevote_INSTANCE_c2_pn197k_offaug_detpn197k.csv',
      'EQS-D-0dB': 'eval_all/gatevote_EQS-D-0dB_c2_pn197k_w6000v3_detpn197k.csv',
      'EQS-D-imp': 'eval_all/gatevote_EQS-D-imp_c2_pn197k_w6000v3_detpn197k.csv'}),
    ('PhaseNet',      'eval_all_pm_phasenet6000',
     {d: f'eval_all/gatevote_{d}_base_phasenet6000.csv' for d in DS}),
    ('EQTransformer', 'eval_all_pm_eqt',
     {d: f'eval_all/gatevote_{d}_base_eqt.csv' for d in DS}),
    ('PhaseMamba',    'eval_all_pm_phasemamba6000',
     {d: f'eval_all/gatevote_{d}_base_phasemamba6000.csv' for d in DS}),
]
SHARED = {d: f'eval_all_det_det_pn197k/detscore_{d}_BiMamba-Detector.csv' for d in DS}

def eid(t): return str(t).split('.')[0]
def prf(TP, FP, FN):
    P = TP/max(TP+FP, 1e-12); R = TP/max(TP+FN, 1e-12)
    return P, R, 2*P*R/max(P+R, 1e-12)
def best_thr(sc_ev, sc_nz):
    sc_ev = np.asarray(sc_ev, float); sc_nz = np.asarray(sc_nz, float)
    sc_ev = sc_ev[np.isfinite(sc_ev)]; sc_nz = sc_nz[np.isfinite(sc_nz)]
    if not len(sc_ev) or not len(sc_nz): return np.nan
    qs = np.quantile(np.unique(np.r_[sc_ev, sc_nz]), np.linspace(0.005, 0.995, 400))
    best = (-1.0, float(qs[0]))
    for th in qs:
        TP = int((sc_ev >= th).sum()); FP = int((sc_nz >= th).sum())
        _P, _R, F = prf(TP, FP, len(sc_ev) - TP)
        if F > best[0]: best = (F, float(th))
    return best[1]
def f1(err, TOL, FP):
    TP = int((err <= TOL).sum()); FN = len(err) - TP
    P = TP/max(TP+FP, 1e-12); R = TP/max(TP+FN, 1e-12)
    return 2*P*R/max(P+R, 1e-12)

rows = []
for ds in DS:
    gv0 = pd.read_csv(MODELS[0][2][ds])
    tr_ev = gv0[gv0.split == 'events'].trace.values
    tr_nz = gv0[gv0.split == 'noise'].trace.values
    ug = np.unique([eid(t) for t in tr_ev])
    rng = np.random.RandomState(0); rng.shuffle(ug)
    cal_g = set(ug[: max(1, int(round(0.25 * len(ug))))])
    rng2 = np.random.RandomState(0)
    perm = rng2.permutation(len(tr_nz))
    cal_n = set(tr_nz[perm[: max(1, int(round(0.25 * len(tr_nz))))]])

    # 方案 A 的 τ（共用检测器）
    sh = pd.read_csv(SHARED[ds])
    A_ev = sh[sh.split == 'events'].copy(); A_nz = sh[sh.split == 'noise'].copy()
    A_mE = np.array([eid(t) in cal_g for t in A_ev.trace])
    A_mN = np.array([t in cal_n for t in A_nz.trace])
    tauA = best_thr(A_ev.score.values[A_mE], A_nz.score.values[A_mN])

    for name, pdir, gvmap in MODELS:
        f = gvmap[ds]
        if not os.path.exists(f):
            continue
        gv = pd.read_csv(f)
        g_ev = gv[gv.split == 'events'].copy()
        g_nz = gv[gv.split == 'noise'].copy()
        isE = np.array([eid(t) in cal_g for t in g_ev.trace])
        isN = np.array([t in cal_n for t in g_nz.trace])

        pe = (g_ev.win_pick - g_ev.p_true).abs().values
        has_s = 's_win_pick' in g_ev.columns and g_ev.s_win_pick.notna().any()
        se = (g_ev.s_win_pick - g_ev.s_true).abs().values if has_s else None

        # 方案 B 的分数（模型自有 p_peak）
        _g = glob.glob(f'{pdir}/{ds}_*_picking.csv')
        if not _g:
            continue
        pk = pd.read_csv(_g[0])[['trace', 'p_peak']]
        B_ev = g_ev.merge(pk, on='trace', how='left')
        B_nz = g_nz.merge(pk, on='trace', how='left')
        scB_ev = B_ev.p_peak.values.astype(float)
        scB_nz = B_nz.p_peak.values.astype(float)
        tauB = best_thr(scB_ev[isE], scB_nz[isN])

        for scheme, th, scE, scN in (('A', tauA, g_ev.score.values.astype(float),
                                      g_nz.score.values.astype(float)),
                                     ('B', tauB, scB_ev, scB_nz)):
            teE = ~isE; teN = ~isN
            passed = scE[teE] >= th
            TP = int(passed.sum()); FP = int((scN[teN] >= th).sum())
            FN = int(teE.sum()) - TP
            P_, R_, F_ = prf(TP, FP, FN)
            r = dict(dataset=ds, model=name, scheme=scheme, tau=th,
                     n_test_ev=int(teE.sum()), n_test_no=int(teN.sum()),
                     TP=TP, TN=int(teN.sum())-FP, FP=FP, FN=FN, Pr=P_, Re=R_, F1=F_)
            pte = pe[teE][passed]; pte = pte[np.isfinite(pte)]
            if len(pte):
                r['P_F1_05'] = f1(pte, 50, FP); r['P_F1_01'] = f1(pte, 10, FP)
                r['P_MAE'] = pte.mean()/100; r['P_SD'] = pte.std(ddof=0)/100
            if has_s and se is not None:
                ste = se[teE][passed]; ste = ste[np.isfinite(ste)]
                if len(ste):
                    r['S_F1_05'] = f1(ste, 50, FP); r['S_F1_01'] = f1(ste, 10, FP)
                    r['S_MAE'] = ste.mean()/100; r['S_SD'] = ste.std(ddof=0)/100
            rows.append(r)

t = pd.DataFrame(rows)
t.to_csv('eval_all/CALIB25_TESTS75_AB.csv', index=False)
print('saved eval_all/CALIB25_TESTS75_AB.csv  行数', len(t))

for ds in DS:
    for scheme in ('A', 'B'):
        s = t[(t.dataset == ds) & (t.scheme == scheme)]
        if s.empty: continue
        print(f'\n■ {ds}  方案{scheme}')
        print('   {:16s} {:>10s} {:>7s} {:>7s} {:>7s} {:>7s} {:>8s} {:>8s}'.format(
            '模型', 'τ', 'Pr', 'Re', 'F1', 'TP', 'P_F1@.5', 'P_F1@.1'))
        for _, r in s.iterrows():
            print('   {:16s} {:10.3e} {:7.4f} {:7.4f} {:7.4f} {:7d} {:8.4f} {:8.4f}'.format(
                r.model, r.tau, r.Pr, r.Re, r.F1, r.TP,
                r.get('P_F1_05', np.nan), r.get('P_F1_01', np.nan)))

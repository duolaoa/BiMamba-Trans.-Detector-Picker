# -*- coding: utf-8 -*-
"""方案 B（修正）：MAE/σ 改用**端到端** win_pick / s_win_pick（与方案 A、与论文 Table 9 一致）。
   之前误用了 picking CSV 的 p_err / s_err（窗口级、真值摆窗），导致 MAE 偏小。
"""
import pandas as pd, numpy as np, os, glob

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


DS = ['INSTANCE', 'EQS-D-0dB', 'EQS-D-imp']
MODELS = [
    ('BiMamba-Trans.', {'INSTANCE':  'eval_all/gatevote_INSTANCE_c2_pn197k_offaug_detpn197k.csv',
                        'EQS-D-0dB': 'eval_all/gatevote_EQS-D-0dB_c2_pn197k_w6000v3_detpn197k.csv',
                        'EQS-D-imp': 'eval_all/gatevote_EQS-D-imp_c2_pn197k_w6000v3_detpn197k.csv'},
     {d: f'eval_all_det_det_pn197k/detscore_{d}_BiMamba-Detector.csv' for d in DS}),
    ('PhaseNet', {d: f'eval_all/gatevote_{d}_base_phasenet6000.csv' for d in DS},
     {d: f'eval_all_pm_phasenet6000/detscore_{d}_PhaseNet.csv' for d in DS}),
    ('EQTransformer', {d: f'eval_all/gatevote_{d}_base_eqt.csv' for d in DS},
     {d: f'eval_all_pm_eqt/detscore_{d}_EQTransformer.csv' for d in DS}),
    ('PhaseMamba', {d: f'eval_all/gatevote_{d}_base_phasemamba6000.csv' for d in DS},
     {d: f'eval_all_pm_phasemamba6000/detscore_{d}_PhaseMamba.csv' for d in DS}),
]
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
for ds in DS:
    ref = pd.read_csv(MODELS[0][1][ds])
    ug = np.unique([eid(t) for t in ref[ref.split=='events'].trace])
    nz_tr = ref[ref.split=='noise'].trace.values
    rng = np.random.RandomState(0); rng.shuffle(ug)
    cal_g = set(ug[:max(1,int(round(0.25*len(ug))))])
    rng2 = np.random.RandomState(0)
    perm = rng2.permutation(len(nz_tr))
    cal_n = set(nz_tr[perm[:max(1,int(round(0.25*len(nz_tr))))]])

    for name, gvmap, dmap in MODELS:
        f = gvmap[ds]
        if not os.path.exists(f) or not os.path.exists(dmap[ds]):
            continue
        d = pd.read_csv(f)
        ev = d[d.split=='events'].copy(); nz = d[d.split=='noise'].copy()
        cE = np.array([eid(t) in cal_g for t in ev.trace])
        cN = np.array([t in cal_n for t in nz.trace])
        teE, teN = ~cE, ~cN
        n_ev, n_no = int(teE.sum()), int(teN.sum())
        # 模型自有检测分数
        sc = pd.read_csv(dmap[ds])
        s_ev = sc[sc.split=='events'].set_index('trace').score
        s_nz = sc[sc.split=='noise'].set_index('trace').score
        ev_sc = s_ev.reindex(ev.trace.values).values
        nz_sc = s_nz.reindex(nz.trace.values).values
        tau = best_thr(ev_sc[cE], nz_sc[cN])
        passed = ev_sc[teE] >= tau
        TP = int(passed.sum()); FP = int((nz_sc[teN] >= tau).sum()); FN = n_ev - TP
        P_, R_, F_ = prf(TP, FP, FN)
        r = dict(dataset=ds, model=name, tau=tau, n_ev=n_ev, n_no=n_no,
                 TP=TP, TN=n_no-FP, FP=FP, FN=FN, Pr=P_, Re=R_, F1=F_)
        # 端到端误差（与论文一致）
        pe = (ev.win_pick - ev.p_true).abs().values[teE][passed]
        pe = pe[np.isfinite(pe)]
        if len(pe):
            r['P_TP50'] = int((pe<=50).sum()); r['P_TP10'] = int((pe<=10).sum())
            r['P_MAE'] = pe.mean()/100; r['P_SD'] = pe.std(ddof=0)/100
        if 's_win_pick' in ev.columns and ev.s_win_pick.notna().any():
            se = (ev.s_win_pick - ev.s_true).abs().values[teE][passed]
            n_s = int(np.isfinite(se).sum()); se = se[np.isfinite(se)]
            if len(se):
                r['S_n'] = n_s
                r['S_TP50'] = int((se<=50).sum()); r['S_TP10'] = int((se<=10).sum())
                r['S_MAE'] = se.mean()/100; r['S_SD'] = se.std(ddof=0)/100
        rows.append(r)

t = pd.DataFrame(rows)
t.to_csv('eval_all/CALIB25_B_PS_CM_FIX.csv', index=False)
print('saved eval_all/CALIB25_B_PS_CM_FIX.csv  rows =', len(t))
for ds in t.dataset.unique():
    s = t[t.dataset==ds]
    print(f'\n== {ds} ==  test {int(s.n_ev.iloc[0])} 事件 / {int(s.n_no.iloc[0])} 噪声')
    print('   {:16s} {:>10s} {:>6s} {:>5s} {:>6s} {:>9s} {:>8s} {:>8s} {:>9s} {:>8s} {:>8s}'.format(
        '模型','τ','TP','FP','FN','P_F1@.5','P_MAE','P_SD','S_F1@.5','S_MAE','S_SD'))
    for _, r in s.sort_values('P_MAE').iterrows():
        pTP = r.get('P_TP50', np.nan)
        pf1 = 2*(pTP/max(pTP+r.FP,1e-12))*(pTP/max(pTP+r.FN,1e-12))/max((pTP/max(pTP+r.FP,1e-12))+(pTP/max(pTP+r.FN,1e-12)),1e-12) if pd.notna(pTP) else np.nan
        sTP = r.get('S_TP50', np.nan); sFN = (r.S_n - sTP) if pd.notna(sTP) else np.nan
        sf1 = 2*(sTP/max(sTP+r.FP,1e-12))*(sTP/max(sTP+sFN,1e-12))/max((sTP/max(sTP+r.FP,1e-12))+(sTP/max(sTP+sFN,1e-12)),1e-12) if pd.notna(sTP) else np.nan
        print('   {:16s} {:10.3e} {:6d} {:5d} {:6d} {:9.4f} {:8.3f} {:8.3f} {:9.4f} {:8.3f} {:8.3f}'.format(
            r.model, r.tau, int(r.TP), int(r.FP), int(r.FN), pf1,
            r.get('P_MAE', np.nan), r.get('P_SD', np.nan), sf1,
            r.get('S_MAE', np.nan), r.get('S_SD', np.nan)))

# -*- coding: utf-8 -*-
"""消融表改成**系统级 Recall**（分母 = 全部测试事件），5 个划分种子，mean ± std。
同时给出 hit（门控口径）用于对比，说明差异约 1 点、排序不变。
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
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    qs=np.quantile(np.unique(np.r_[a,b]), np.linspace(0.005,0.995,400))
    best=(-1.0,float(qs[0]))
    for th in qs:
        TP=int((a>=th).sum()); FP=int((b>=th).sum())
        _P,_R,F=prf(TP,FP,len(a)-TP)
        if F>best[0]: best=(F,float(th))
    return best[1]

rows=[]
for ds, variants in SETS.items():
    sh = pd.read_csv(DET[ds])
    ev_all = sh[sh.split=='events'].trace.values
    nz_all = sh[sh.split=='noise'].trace.values
    ug = np.unique([eid(t) for t in ev_all])
    for seed in SEEDS:
        rng=np.random.RandomState(seed); u2=ug.copy(); rng.shuffle(u2)
        cal_g=set(u2[:max(1,int(round(0.25*len(u2))))])
        rng2=np.random.RandomState(seed); perm=rng2.permutation(len(nz_all))
        cal_n=set(nz_all[perm[:max(1,int(round(0.25*len(nz_all))))]])
        cE=np.array([eid(t) in cal_g for t in ev_all])
        cN=np.array([t in cal_n for t in nz_all])
        tau=best_thr(sh[sh.split=='events'].score.values[cE],
                     sh[sh.split=='noise'].score.values[cN])
        for name, tag in variants:
            f=f'eval_all/gatevote_{ds}_{tag}.csv'
            if not os.path.exists(f): continue
            d=pd.read_csv(f)
            ev=d[d.split=='events'].copy()
            teE=~np.array([eid(t) in cal_g for t in ev.trace])
            n_ev=int(teE.sum())
            passed=ev.score.values[teE]>=tau
            n_g=int(passed.sum())
            pe=(ev.win_pick-ev.p_true).abs().values[teE]
            pe_g=pe[passed]; pe_g=pe_g[np.isfinite(pe_g)]
            r=dict(dataset=ds, seed=seed, variant=name, n_ev=n_ev, n_gated=n_g,
                   P_TP=int(np.nansum((pe<=50))),
                   P_Re50=float(np.nansum((pe<=50))/n_ev*100),
                   P_Re10=float(np.nansum((pe<=10))/n_ev*100),
                   P_Hit50=float((pe_g<=50).mean()*100) if len(pe_g) else np.nan)
            hasS = 's_win_pick' in ev.columns and ev.s_win_pick.notna().any()
            if hasS:
                st=ev.s_true.values[teE]; sw=ev.s_win_pick.values[teE]
                m=np.isfinite(st)
                se=np.abs(sw-st)
                nS=int(m.sum())
                r['S_n']=nS
                r['S_Re50']=float(np.nansum((se<=50)&m)/nS*100)
                r['S_Re10']=float(np.nansum((se<=10)&m)/nS*100)
                seg=se[m & passed]; seg=seg[np.isfinite(seg)]
                r['S_Hit50']=float((seg<=50).mean()*100) if len(seg) else np.nan
            rows.append(r)

t=pd.DataFrame(rows); t.to_csv('eval_all/ABLATION_RECALL.csv', index=False)
print('saved eval_all/ABLATION_RECALL.csv  rows =', len(t))
for ds in t.dataset.unique():
    s=t[t.dataset==ds]
    print(f'\n{"="*104}\n== {ds} · 5 种子均值 ± std ==\n{"="*104}')
    order=[v for v,_ in SETS[ds]]
    cols=[c for c in ('P_Re50','P_Re10','P_Hit50','S_Re50','S_Re10','S_Hit50') if c in s.columns]
    print('   {:16s} '.format('变体') + ' '.join(f'{c:>16s}' for c in cols))
    for v in order:
        g=s[s.variant==v]
        if g.empty: continue
        cells=[f'{g[c].mean():7.2f}±{g[c].std(ddof=0):5.2f}'.rjust(16) for c in cols]
        print('   {:16s} '.format(v) + ' '.join(cells))
    print(f'\n   —— 系统级 Recall 相对 All 的差值（均值）——')
    base=s[s.variant=='All']
    for v in order[1:]:
        g=s[s.variant==v]
        if g.empty: continue
        d1=[f'{g[c].mean()-base[c].mean():+8.2f}' for c in cols]
        print('   {:16s} '.format(v) + ' '.join(f'{x:>16s}' for x in d1))
    # 与 hit 的差
    if 'P_Hit50' in s.columns:
        print(f'\n   —— Hit(门控) − Re(系统) 的差（均值，应约 +1 点）——')
        for v in order:
            g=s[s.variant==v]
            if g.empty: continue
            print(f'   {v:16s}  P: {g.P_Hit50.mean()-g.P_Re50.mean():+.2f} 点'
                  + (f'   S: {g.S_Hit50.mean()-g.S_Re50.mean():+.2f} 点' if 'S_Hit50' in s.columns else ''))

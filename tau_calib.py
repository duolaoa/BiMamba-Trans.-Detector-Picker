# -*- coding: utf-8 -*-
"""方案：用独立的「目标域校准集」确定 τ，剩余样本作测试。

  1) 按 event id 划分 校准/测试（同事件不跨集）
  2) 在校准集上扫 τ，取 F1 最优
  3) 在测试集上评估（完全未参与调参）
  与现有「2 折交叉拟合」对比，并给出不同校准比例的影响。
"""
import numpy as np, pandas as pd, os

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


DET = [('STEAD-DET', 'eval_all_steaddet_det_pn197k'),
       ('INSTANCE',  'eval_all_det_det_pn197k'),
       ('EQS-D-0dB', 'eval_all_det_det_pn197k'),
       ('EQS-D-imp', 'eval_all_det_det_pn197k')]

def eid(t): return str(t).split('.')[0]
def prf(TP, FP, FN):
    P = TP/max(TP+FP,1e-9); R = TP/max(TP+FN,1e-9)
    return P, R, 2*P*R/max(P+R,1e-9)
def best_thr(ev, nz):
    best = (-1.0, 0.0)
    for th in np.unique(np.r_[ev, nz]):
        P,R,F = prf(int((ev>=th).sum()), int((nz>=th).sum()), int((ev<th).sum()))
        if F > best[0]: best = (F, float(th))
    return best[1]

print('=' * 116)
print('τ 的确定方式对比：① 2 折交叉拟合（现状）  ② 独立校准集（按 event id 划分）')
print('=' * 116)

rows = []
for ds, d in DET:
    f = os.path.join(d, f'detscore_{ds}_BiMamba-Detector.csv')
    if not os.path.exists(f):
        print(f'  {ds}: 缺'); continue
    df = pd.read_csv(f)
    sc, sp, tr = df['score'].values, df['split'].values, df['trace'].values
    ev, nz = sc[sp=='events'], sc[sp=='noise']
    g = np.array([eid(t) for t in tr[sp=='events']])
    ug = np.unique(g); rng = np.random.RandomState(0); rng.shuffle(ug)

    # 交叉拟合（现状）
    gs = {x:i for i,x in enumerate(ug)}
    ef = np.array([gs[x] % 2 for x in g])
    nf = rng.randint(0, 2, size=len(nz))
    thA = float(np.mean([best_thr(ev[ef==k], nz[nf==k]) for k in (0,1)]))
    PA, RA, FA = prf(int((ev>=thA).sum()), int((nz>=thA).sum()), int((ev<thA).sum()))

    print(f'\n■ {ds}')
    print(f'   ① 交叉拟合 τ = {thA:.4e}   Pr={PA:.4f} Re={RA:.4f} F1={FA:.4f}  （全部样本）')

    for frac in (0.25, 0.33, 0.50):
        nc = int(len(ug) * frac)
        cal_g = set(ug[:nc])
        m_cal = np.array([x in cal_g for x in g])
        # 噪声按同比例随机分（噪声无事件标签，只按比例）
        rng2 = np.random.RandomState(0)
        nz_perm = rng2.permutation(len(nz))
        nc_n = int(len(nz) * frac)
        cal_n = np.zeros(len(nz), bool); cal_n[nz_perm[:nc_n]] = True

        th = best_thr(ev[m_cal], nz[cal_n])
        te_e = ev[~m_cal]; te_n = nz[~cal_n]
        P2, R2, F2 = prf(int((te_e>=th).sum()), int((te_n>=th).sum()), int((te_e<th).sum()))
        print(f'   ② 校准 {frac*100:2.0f}%（{m_cal.sum():5d} 事件 / {cal_n.sum():4d} 噪声）'
              f'  τ={th:.4e}  →  测试集（{len(te_e):5d} 事件 / {len(te_n):4d} 噪声）'
              f'  Pr={P2:.4f} Re={R2:.4f} F1={F2:.4f}')
        rows.append(dict(dataset=ds, scheme=f'calib{int(frac*100)}', tau=th,
                         Pr=P2, Re=R2, F1=F2, n_test=len(te_e)))
    rows.append(dict(dataset=ds, scheme='crossfit', tau=thA, Pr=PA, Re=RA, F1=FA, n_test=len(ev)))

t = pd.DataFrame(rows)
t.to_csv('eval_all/TAU_CALIB_VS_CROSSFIT.csv', index=False)
print()
print('  已保存 TAU_CALIB_VS_CROSSFIT.csv')
print()
print('   --- 各方案 F1 汇总（便于看差异）---')
piv = t.pivot_table(index='dataset', columns='scheme', values='F1')
print(piv.to_string(float_format=lambda v: f'{v:.4f}'))
print()
print('   --- τ 的差异 ---')
piv2 = t.pivot_table(index='dataset', columns='scheme', values='tau')
print(piv2.to_string(float_format=lambda v: f'{v:.3e}'))

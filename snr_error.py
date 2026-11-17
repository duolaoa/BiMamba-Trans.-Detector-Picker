# -*- coding: utf-8 -*-
"""误差随信噪比的变化：STEAD-CH5C 与 INSTANCE。

SNR 定义（逐条记录、逐波）：
  噪声段 = 该波到时刻前 1,000~100 采样点
  信号段 = 该波到时刻起 300 采样点（P）或 S−P 差的一半（S，上限 1,000）
  SNR_dB = 20 log10(RMS(信号段) / RMS(噪声段))
三通道取功率平均。
"""
import os, numpy as np, pandas as pd, json
import eval_evsplit_models as E

# --- output dir (run from the repository root) ---
os.makedirs('eval_all', exist_ok=True)
os.makedirs(os.path.join('eval_all', 'figs'), exist_ok=True)


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64)**2)) + 1e-12)

def snr_of(x, t, half=300):
    """x: (T, C)；t: 到时刻（采样点）"""
    T = x.shape[0]
    t = int(round(float(t)))
    n0, n1 = max(0, t-1000), max(0, t-100)
    s0, s1 = int(t), min(T, int(t)+half)
    if n1 - n0 < 50 or s1 - s0 < 20: return np.nan
    nse = rms(x[n0:n1]); sig = rms(x[s0:s1])
    return 20*np.log10(sig/nse)

def run(ds, pick_dir, tag_name):
    g = [f for f in os.listdir(pick_dir) if f.startswith(ds) and f.endswith('_picking.csv')]
    if not g:
        print(f'  {ds}: 无 picking'); return None
    d = pd.read_csv(os.path.join(pick_dir, g[0]))
    err = {}
    for _, r in d.iterrows():
        err[str(r['trace'])] = (r.get('p_err'), r.get('s_err'), r.get('p_true'), r.get('s_true'))
    out = []
    n = 0
    for name, x, p, s in E.DATASETS[ds]('events', 0):
        k = str(name)
        if k not in err: continue
        pe, se, pt, st = err[k]
        if not np.isfinite(p): continue
        snr_p = snr_of(x, p, 300)
        half_s = 300
        if np.isfinite(s) and s > p:
            half_s = max(150, min(1000, int((s - p) / 2)))
        snr_s = snr_of(x, s if np.isfinite(s) else p, half_s) if np.isfinite(s) else np.nan
        out.append(dict(trace=k, snr_p=snr_p, snr_s=snr_s,
                        p_err=abs(pe) if pe == pe and np.isfinite(pe) else np.nan,
                        s_err=abs(se) if se == se and np.isfinite(se) else np.nan))
        n += 1
        if n >= 10000: break
        if n % 2000 == 0: print(f'    {ds}: {n} 条…', flush=True)
    df = pd.DataFrame(out)
    df.to_csv(f'eval_all/SNR_{ds}.csv', index=False)
    print(f'  {ds}: 完成 {len(df)} 条 → SNR_{ds}.csv')
    return df

if __name__ == '__main__':
    res = {}
    res['STEAD-CH5C'] = run('STEAD-CH5C', 'eval_all_pm_w6000_all', 'w6000')
    res['INSTANCE'] = run('INSTANCE', 'eval_all_pm_w6000_all', 'w6000')
    print()
    for ds, df in res.items():
        if df is None or not len(df): continue
        print('=' * 96)
        print(f'{ds} · 误差随 SNR 的变化（6,000 窗完整模型）')
        print('=' * 96)
        for wave, sk, ek in (('P', 'snr_p', 'p_err'), ('S', 'snr_s', 's_err')):
            sub = df[[sk, ek]].dropna()
            if not len(sub): continue
            print(f'\n--- {wave} 波（n={len(sub)}）')
            print('   {:>12s} {:>6s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}'.format(
                'SNR(dB)', '条数', '≤0.1s%', '≤0.5s%', 'MAE(点)', 'Median(点)', 'P95(点)'))
            BINS = [-20, -5, 0, 5, 10, 15, 20, 25, 30, 40, 60]
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                s2 = sub[(sub[sk] >= lo) & (sub[sk] < hi)]
                if len(s2) < 15: continue
                e = s2[ek].values
                print('   {:>5d}–{:<6d} {:>6d} {:>9.2f} {:>9.2f} {:>9.1f} {:>9.1f} {:>9.1f}'.format(
                    lo, hi, len(e), (e <= 10).mean()*100, (e <= 50).mean()*100,
                    e.mean(), np.median(e), np.percentile(e, 95)))

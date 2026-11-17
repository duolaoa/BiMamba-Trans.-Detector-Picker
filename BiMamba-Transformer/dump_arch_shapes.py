# -*- coding: utf-8 -*-
"""逐配置 dump 模块内部参数 + 输入/输出尺寸（用 forward hook 实测，非推算）。

产出：ARCH_DETAIL_SHAPES.md（拾取器 + 检测器两部分）
"""
import glob, os, sys
import torch, torch.nn as nn
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_bimamba_evsplit as P    # 拾取器
import PS_Detection_evsplit as D     # 检测器

FALLBACK = {
    'best_lightweight_model_3C_ctrl813k': dict(proj_dim=96, hidden_dim=48),
    'best_detection_model_3C_wofocal':    dict(proj_dim=96, hidden_dim=48),
}

WATCH = ['lfem_block1', 'pool1', 'lfem_block2', 'pool2', 'lfem_block3', 'proj',
         'mamba', 'transformer', 'fusion', 'regressor', 's_regressor',
         'p_det_head', 's_det_head']
ORDER_P = ['lfem_block1', 'pool1', 'lfem_block2', 'pool2', 'lfem_block3', 'proj',
           'mamba', 'transformer', 'fusion', 'regressor', 's_regressor']
ORDER_D = ['lfem_block1', 'pool1', 'lfem_block2', 'pool2', 'lfem_block3', 'proj',
           'mamba', 'transformer', 'fusion']
CSV = []


def n(m):
    return sum(p.numel() for p in m.parameters())


def shape_str(t):
    if isinstance(t, (tuple, list)):
        t = t[0] if len(t) else None
    return '×'.join(str(x) for x in tuple(t.shape)) if torch.is_tensor(t) else '—'


def desc(name, mod):
    if name.startswith('lfem') and hasattr(mod, 'branch0'):
        h = mod.conv1x1.out_channels
        return (f"LFEM1D: Conv1d(3→{h},k=1)+BN+SiLU → [b0 Conv1d(k={mod.branch0.kernel_size[0]}) | "
                f"b1 DilatedConv(k={mod.branch1.dconv.kernel_size[0]},d={mod.branch1.dconv.dilation[0]}) | "
                f"b2 DepthWise(k={mod.branch2.dwconv.kernel_size[0]})] → shuffle"
                f"{'✓' if getattr(mod,'use_shuffle',False) else '✗'} → Conv1d({h}→3,k=1)+BN → SiLU(+残差)")
    if name.startswith('lfem'):
        c0 = mod[0] if isinstance(mod, nn.Sequential) else mod
        return f"普通 Conv1d(3→3,k={c0.kernel_size[0]})+BN+ReLU（非 LFEM1D）"
    if name.startswith('pool'):
        return f"MaxPool1d(kernel={mod.kernel_size}, stride={mod.stride})"
    if name == 'proj':
        return f"Conv1d({mod.in_channels}→{mod.out_channels}, k={mod.kernel_size[0]}, bias={mod.bias is not None})"
    if name == 'mamba':
        if isinstance(mod, nn.Identity):
            return "nn.Identity（mamba_mode='none'）"
        m = mod.layers[0].fwd_mamba
        return (f"BidirectionalMamba: d_model={m.d_model}, d_state={m.d_state}, d_conv={m.d_conv}, "
                f"expand={m.expand}, layers={len(mod.layers)}；Mamba实例={2*len(mod.layers)+2}，"
                f"LayerNorm={len(mod.layers)}；末层 cat(fwd,bwd)→2×{m.d_model}")
    if name == 'transformer':
        if mod is None:
            return "—"
        L = mod.transformer_encoder.layers[0]
        d = L.self_attn.embed_dim
        return (f"TransformerEncoder: Linear(→{d}) + 正弦PE(10000×{d}, buffer) + Dropout(0.1)；"
                f"layers={len(mod.transformer_encoder.layers)}, nhead={L.self_attn.num_heads}, "
                f"d_k={d//L.self_attn.num_heads}, FFN={L.linear1.out_features}")
    if name == 'fusion':
        lin = mod[0]
        return f"Linear({lin.in_features}→{lin.out_features}) + ReLU + Dropout(0.1)"
    if name in ('regressor', 's_regressor'):
        return (f"Linear({mod[0].in_features}→{mod[0].out_features}) + ReLU + Dropout(0.2) + "
                f"Linear({mod[3].in_features}→{mod[3].out_features})")
    return type(mod).__name__


def run(kind, files, module, order, winlen_default):
    out = []
    for f in files:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        base = os.path.basename(f).replace('.pth', '')
        if not (isinstance(ck, dict) and 'cfg' in ck):
            continue
        c = dict(ck['cfg'])
        if 'proj_dim' not in c:
            if base not in FALLBACK:
                continue
            c.update(FALLBACK[base])          # 旧权重：用已知量回填
            c.setdefault('use_lfem', kind == 'picker')
            c.setdefault('window_size', winlen_default)
        kwargs = dict(input_dim=3,
                      use_transformer=bool(c.get('use_transformer', True)),
                      mamba_mode=str(c.get('mamba_mode', 'bidirectional')),
                      # 检测器的类默认 use_lfem=False；拾取器为 True
                      use_lfem=bool(c.get('use_lfem', kind == 'picker')),
                      use_cross_channel_attention=bool(c.get('use_cross_channel_attention', False)),
                      proj_dim=int(c.get('proj_dim', 96)), hidden_dim=int(c.get('hidden_dim', 48)),
                      mamba_layers=int(c.get('mamba_layers', 3)), d_state=int(c.get('d_state', 16)),
                      expand=int(c.get('expand', 2)))
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        winlen = int(c.get('window_size', winlen_default))
        m = module.LightweightPWaveNet(**kwargs).to(dev).eval()
        shapes = {}
        hs = []
        for nm, sub in m.named_children():
            if nm not in order:
                continue
            def mk(k):
                def hk(mod, inp, outp):
                    shapes[k] = (shape_str(inp[0]) if inp else '—',
                                 shape_str(outp))
                return hk
            hs.append(sub.register_forward_hook(mk(nm)))
        with torch.no_grad():
            y = m(torch.zeros(1, winlen, 3, device=dev))
        for h in hs:
            h.remove()
        tag = os.path.basename(f).replace('.pth', '')
        total = n(m)
        out.append(f"\n### `{tag}` — 合计 **{total:,}** 参数  ·  输入 **(1, {winlen}, 3)**\n")
        out.append("| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |")
        out.append("|---|---|---|---|---:|---:|")
        for i, nm in enumerate(order, 1):
            mod = getattr(m, nm, None)
            if mod is None:
                continue
            ins, outs = shapes.get(nm, ('—', '—'))
            cnt = n(mod)
            out.append(f"| {i} | `{nm}` | **{ins} → {outs}** | {desc(nm, mod)} | {cnt:,} | {cnt/total*100:.1f}% |")
            CSV.append(dict(kind=kind, tag=tag, module=nm, in_shape=ins, out_shape=outs,
                            params=cnt, pct=round(cnt/total*100, 2), total=total))
        if isinstance(y, (tuple, list)):
            ydesc = ' + '.join(shape_str(t) for t in y)
        else:
            ydesc = shape_str(y)
        out.append(f"| | **输出** | **{ydesc}** | | | |")
        out.append(f"| | **合计** | | | **{total:,}** | 100% |")
    return '\n'.join(out)


pk = sorted(glob.glob('best_lightweight_model_3C*.pth'))
dt = sorted(glob.glob('best_detection_model_3C_det*.pth')) + \
     sorted(glob.glob('best_detection_model_3C_wofocal*.pth'))
md = ["# BiMamba 逐配置 · 逐模块内部参数 + 输入/输出尺寸",
      "",
      "> 输入/输出尺寸由 **forward hook 实测**（拾取器输入 `(1,3000,3)`，检测器输入 `(1,6000,3)`）。",
      "", "---", "", "## 一、拾取器（picker）", ""]
md.append(run('picker', pk, P, ORDER_P, 3000))
md += ["", "---", "", "## 二、检测器（detector）", ""]
md.append(run('detector', dt, D, ORDER_D, 6000))
txt = '\n'.join(md)
open('ARCH_DETAIL_SHAPES.md', 'w', encoding='utf-8').write(txt)
pd.DataFrame(CSV).to_csv('ARCH_DETAIL_SHAPES.csv', index=False)
print(txt)

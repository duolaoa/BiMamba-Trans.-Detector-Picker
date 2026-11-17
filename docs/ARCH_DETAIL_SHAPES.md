# Tensor shapes through the network

Shape trace of the forward pass, produced by `dump_arch_shapes.py`. Use it to
confirm the token counts quoted in the paper:

| stage | picker (Z=2 pools) | detector (Z=3 pools) |
|---|---|---|
| input | (B, 3, 6000) | (B, 3, 6000) |
| after LFEM1D + pooling | (B, 16, 1500) → 1500 tokens | (B, 16, 750) → 750 tokens |
| after BiMamba (concat fwd/bwd) | (B, 32, 1500) | (B, 32, 750) |
| after Transformer encoder | (B, 80, 1500) | (B, 80, 750) |
| after fusion | (B, 80, 1500) | (B, 80, 750) |
| output heads | 2 × (B, 1, 1500) | 2 × (B, 1, 750) |

Note that the number of tokens differs between the two networks at the same
input length, which is why the picker and the detector use different stride
conventions (250 vs. 1000 samples) in the decoder.

Regenerate with:

```bash
python dump_arch_shapes.py
```

> The annotations in the dump below are in Chinese; layer names and shapes are
> the authoritative part.

---

# BiMamba 逐配置 · 逐模块内部参数 + 输入/输出尺寸

> 输入/输出尺寸由 **forward hook 实测**（拾取器输入 `(1,3000,3)`，检测器输入 `(1,6000,3)`）。

---

## 一、拾取器（picker）


### `best_lightweight_model_3C_abl199k_nolfem` — 合计 **198,532** 参数  ·  输入 **(1, 3000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×1500 → 1×3×750** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×750 → 1×3×750** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×16×750** | Conv1d(3→16, k=1, bias=True) | 64 | 0.0% |
| 7 | `mamba` | **1×750×16 → 1×750×32** | BidirectionalMamba: d_model=16, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×16 | 26,976 | 13.6% |
| 8 | `transformer` | **1×750×16 → 1×750×80** | TransformerEncoder: Linear(→80) + 正弦PE(10000×80, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=20, FFN=320 | 157,040 | 79.1% |
| 9 | `fusion` | **1×750×112 → 1×750×80** | Linear(112→80) + ReLU + Dropout(0.1) | 9,040 | 4.6% |
| 10 | `regressor` | **1×750×80 → 1×750×1** | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| 11 | `s_regressor` | **1×750×80 → 1×750×1** | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| | **输出** | **1×3000 + 1×3000** | | | |
| | **合计** | | | **198,532** | 100% |

### `best_lightweight_model_3C_abl199k_nomamba` — 合计 **198,562** 参数  ·  输入 **(1, 3000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×3000 → 1×3×3000** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `pool1` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×1500 → 1×3×1500** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 4 | `pool2` | **1×3×1500 → 1×3×750** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×750 → 1×3×750** | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×188×750** | Conv1d(3→188, k=1, bias=True) | 752 | 0.4% |
| 7 | `mamba` | **1×750×188 → 1×750×188** | nn.Identity（mamba_mode='none'） | 0 | 0.0% |
| 8 | `transformer` | **1×750×188 → 1×750×80** | TransformerEncoder: Linear(→80) + 正弦PE(10000×80, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=20, FFN=320 | 170,800 | 86.0% |
| 9 | `fusion` | **1×750×268 → 1×750×80** | Linear(268→80) + ReLU + Dropout(0.1) | 21,520 | 10.8% |
| 10 | `regressor` | **1×750×80 → 1×750×1** | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| 11 | `s_regressor` | **1×750×80 → 1×750×1** | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| | **输出** | **1×3000 + 1×3000** | | | |
| | **合计** | | | **198,562** | 100% |

### `best_lightweight_model_3C_abl199k_notrans` — 合计 **198,078** 参数  ·  输入 **(1, 3000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×3000 → 1×3×3000** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `pool1` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×1500 → 1×3×1500** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 4 | `pool2` | **1×3×1500 → 1×3×750** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×750 → 1×3×750** | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×54×750** | Conv1d(3→54, k=1, bias=True) | 216 | 0.1% |
| 7 | `mamba` | **1×750×54 → 1×750×108** | BidirectionalMamba: d_model=54, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×54 | 194,724 | 98.3% |
| 9 | `fusion` | **1×750×108 → 1×750×16** | Linear(108→16) + ReLU + Dropout(0.1) | 1,744 | 0.9% |
| 10 | `regressor` | **1×750×16 → 1×750×1** | Linear(16→32) + ReLU + Dropout(0.2) + Linear(32→1) | 577 | 0.3% |
| 11 | `s_regressor` | **1×750×16 → 1×750×1** | Linear(16→32) + ReLU + Dropout(0.2) + Linear(32→1) | 577 | 0.3% |
| | **输出** | **1×3000 + 1×3000** | | | |
| | **合计** | | | **198,078** | 100% |

### `best_lightweight_model_3C_eqt466k` — 合计 **463,266** 参数  ·  输入 **(1, 3000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×3000 → 1×3×3000** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `pool1` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×1500 → 1×3×1500** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 4 | `pool2` | **1×3×1500 → 1×3×750** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×750 → 1×3×750** | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×56×750** | Conv1d(3→56, k=1, bias=True) | 224 | 0.0% |
| 7 | `mamba` | **1×750×56 → 1×750×112** | BidirectionalMamba: d_model=56, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×56 | 207,312 | 44.8% |
| 8 | `transformer` | **1×750×56 → 1×750×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 229,152 | 49.5% |
| 9 | `fusion` | **1×750×208 → 1×750×96** | Linear(208→96) + ReLU + Dropout(0.1) | 20,064 | 4.3% |
| 10 | `regressor` | **1×750×96 → 1×750×1** | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 0.7% |
| 11 | `s_regressor` | **1×750×96 → 1×750×1** | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 0.7% |
| | **输出** | **1×3000 + 1×3000** | | | |
| | **合计** | | | **463,266** | 100% |

### `best_lightweight_model_3C_pm322k` — 合计 **328,498** 参数  ·  输入 **(1, 3000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×3000 → 1×3×3000** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `pool1` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×1500 → 1×3×1500** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 4 | `pool2` | **1×3×1500 → 1×3×750** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×750 → 1×3×750** | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×32×750** | Conv1d(3→32, k=1, bias=True) | 128 | 0.0% |
| 7 | `mamba` | **1×750×32 → 1×750×64** | BidirectionalMamba: d_model=32, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×32 | 79,552 | 24.2% |
| 8 | `transformer` | **1×750×32 → 1×750×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 226,848 | 69.1% |
| 9 | `fusion` | **1×750×160 → 1×750×96** | Linear(160→96) + ReLU + Dropout(0.1) | 15,456 | 4.7% |
| 10 | `regressor` | **1×750×96 → 1×750×1** | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 1.0% |
| 11 | `s_regressor` | **1×750×96 → 1×750×1** | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 1.0% |
| | **输出** | **1×3000 + 1×3000** | | | |
| | **合计** | | | **328,498** | 100% |

### `best_lightweight_model_3C_pn197k` — 合计 **198,610** 参数  ·  输入 **(1, 3000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×3000 → 1×3×3000** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `pool1` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×1500 → 1×3×1500** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 4 | `pool2` | **1×3×1500 → 1×3×750** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×750 → 1×3×750** | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×16×750** | Conv1d(3→16, k=1, bias=True) | 64 | 0.0% |
| 7 | `mamba` | **1×750×16 → 1×750×32** | BidirectionalMamba: d_model=16, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×16 | 26,976 | 13.6% |
| 8 | `transformer` | **1×750×16 → 1×750×80** | TransformerEncoder: Linear(→80) + 正弦PE(10000×80, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=20, FFN=320 | 157,040 | 79.1% |
| 9 | `fusion` | **1×750×112 → 1×750×80** | Linear(112→80) + ReLU + Dropout(0.1) | 9,040 | 4.6% |
| 10 | `regressor` | **1×750×80 → 1×750×1** | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| 11 | `s_regressor` | **1×750×80 → 1×750×1** | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| | **输出** | **1×3000 + 1×3000** | | | |
| | **合计** | | | **198,610** | 100% |

---

## 二、检测器（detector）


### `best_detection_model_3C_det_eqt466k` — 合计 **463,188** 参数  ·  输入 **(1, 6000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×6000 → 1×3×6000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×6000 → 1×3×3000** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×56×750** | Conv1d(3→56, k=1, bias=True) | 224 | 0.0% |
| 7 | `mamba` | **1×750×56 → 1×750×112** | BidirectionalMamba: d_model=56, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×56 | 207,312 | 44.8% |
| 8 | `transformer` | **1×750×56 → 1×750×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 229,152 | 49.5% |
| 9 | `fusion` | **1×750×208 → 1×750×96** | Linear(208→96) + ReLU + Dropout(0.1) | 20,064 | 4.3% |
| | **输出** | **1×6000 + 1×6000** | | | |
| | **合计** | | | **463,188** | 100% |

### `best_detection_model_3C_det_pm322k` — 合计 **328,420** 参数  ·  输入 **(1, 6000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×6000 → 1×3×6000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×6000 → 1×3×3000** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×32×750** | Conv1d(3→32, k=1, bias=True) | 128 | 0.0% |
| 7 | `mamba` | **1×750×32 → 1×750×64** | BidirectionalMamba: d_model=32, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×32 | 79,552 | 24.2% |
| 8 | `transformer` | **1×750×32 → 1×750×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 226,848 | 69.1% |
| 9 | `fusion` | **1×750×160 → 1×750×96** | Linear(160→96) + ReLU + Dropout(0.1) | 15,456 | 4.7% |
| | **输出** | **1×6000 + 1×6000** | | | |
| | **合计** | | | **328,420** | 100% |

### `best_detection_model_3C_det_pm322k_w3000` — 合计 **328,420** 参数  ·  输入 **(1, 3000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×1500 → 1×3×750** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×750 → 1×3×750** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×375 → 1×32×375** | Conv1d(3→32, k=1, bias=True) | 128 | 0.0% |
| 7 | `mamba` | **1×375×32 → 1×375×64** | BidirectionalMamba: d_model=32, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×32 | 79,552 | 24.2% |
| 8 | `transformer` | **1×375×32 → 1×375×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 226,848 | 69.1% |
| 9 | `fusion` | **1×375×160 → 1×375×96** | Linear(160→96) + ReLU + Dropout(0.1) | 15,456 | 4.7% |
| | **输出** | **1×3000 + 1×3000** | | | |
| | **合计** | | | **328,420** | 100% |

### `best_detection_model_3C_det_pn197k` — 合计 **198,532** 参数  ·  输入 **(1, 6000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×6000 → 1×3×6000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×6000 → 1×3×3000** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×16×750** | Conv1d(3→16, k=1, bias=True) | 64 | 0.0% |
| 7 | `mamba` | **1×750×16 → 1×750×32** | BidirectionalMamba: d_model=16, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×16 | 26,976 | 13.6% |
| 8 | `transformer` | **1×750×16 → 1×750×80** | TransformerEncoder: Linear(→80) + 正弦PE(10000×80, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=20, FFN=320 | 157,040 | 79.1% |
| 9 | `fusion` | **1×750×112 → 1×750×80** | Linear(112→80) + ReLU + Dropout(0.1) | 9,040 | 4.6% |
| | **输出** | **1×6000 + 1×6000** | | | |
| | **合计** | | | **198,532** | 100% |

### `best_detection_model_3C_detabl328k_lfem` — 合计 **328,498** 参数  ·  输入 **(1, 6000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×6000 → 1×3×6000** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `pool1` | **1×3×6000 → 1×3×3000** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×3000 → 1×3×3000** | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 4 | `pool2` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×1500 → 1×3×1500** | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×32×750** | Conv1d(3→32, k=1, bias=True) | 128 | 0.0% |
| 7 | `mamba` | **1×750×32 → 1×750×64** | BidirectionalMamba: d_model=32, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×32 | 79,552 | 24.2% |
| 8 | `transformer` | **1×750×32 → 1×750×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 226,848 | 69.1% |
| 9 | `fusion` | **1×750×160 → 1×750×96** | Linear(160→96) + ReLU + Dropout(0.1) | 15,456 | 4.7% |
| | **输出** | **1×6000 + 1×6000** | | | |
| | **合计** | | | **328,498** | 100% |

### `best_detection_model_3C_detabl328k_nomamba` — 合计 **328,508** 参数  ·  输入 **(1, 6000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×6000 → 1×3×6000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×6000 → 1×3×3000** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×454×750** | Conv1d(3→454, k=1, bias=True) | 1,816 | 0.6% |
| 7 | `mamba` | **1×750×454 → 1×750×454** | nn.Identity（mamba_mode='none'） | 0 | 0.0% |
| 8 | `transformer` | **1×750×454 → 1×750×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 267,360 | 81.4% |
| 9 | `fusion` | **1×750×550 → 1×750×96** | Linear(550→96) + ReLU + Dropout(0.1) | 52,896 | 16.1% |
| | **输出** | **1×6000 + 1×6000** | | | |
| | **合计** | | | **328,508** | 100% |

### `best_detection_model_3C_detabl328k_notrans` — 合计 **328,672** 参数  ·  输入 **(1, 6000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×6000 → 1×3×6000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×6000 → 1×3×3000** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×70×750** | Conv1d(3→70, k=1, bias=True) | 280 | 0.1% |
| 7 | `mamba` | **1×750×70 → 1×750×140** | BidirectionalMamba: d_model=70, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×70 | 308,420 | 93.8% |
| 9 | `fusion` | **1×750×140 → 1×750×96** | Linear(140→96) + ReLU + Dropout(0.1) | 13,536 | 4.1% |
| | **输出** | **1×6000 + 1×6000** | | | |
| | **合计** | | | **328,672** | 100% |

### `best_detection_model_3C_wofocal` — 合计 **813,412** 参数  ·  输入 **(1, 6000, 3)**

| # | 模块 | 输入 → 输出 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---|---:|---:|
| 1 | `lfem_block1` | **1×3×6000 → 1×3×6000** | 普通 Conv1d(3→3,k=7)+BN+ReLU（非 LFEM1D） | 72 | 0.0% |
| 2 | `pool1` | **1×3×6000 → 1×3×3000** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 3 | `lfem_block2` | **1×3×3000 → 1×3×3000** | 普通 Conv1d(3→3,k=5)+BN+ReLU（非 LFEM1D） | 54 | 0.0% |
| 4 | `pool2` | **1×3×3000 → 1×3×1500** | MaxPool1d(kernel=2, stride=2) | 0 | 0.0% |
| 5 | `lfem_block3` | **1×3×1500 → 1×3×1500** | 普通 Conv1d(3→3,k=3)+BN+ReLU（非 LFEM1D） | 36 | 0.0% |
| 6 | `proj` | **1×3×750 → 1×96×750** | Conv1d(3→96, k=1, bias=True) | 384 | 0.0% |
| 7 | `mamba` | **1×750×96 → 1×750×192** | BidirectionalMamba: d_model=96, d_state=16, d_conv=4, expand=2, layers=3；Mamba实例=8，LayerNorm=3；末层 cat(fwd,bwd)→2×96 | 545,856 | 67.1% |
| 8 | `transformer` | **1×750×96 → 1×750×96** | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1)；layers=2, nhead=4, d_k=24, FFN=384 | 232,992 | 28.6% |
| 9 | `fusion` | **1×750×288 → 1×750×96** | Linear(288→96) + ReLU + Dropout(0.1) | 27,744 | 3.4% |
| | **输出** | **1×6000 + 1×6000** | | | |
| | **合计** | | | **813,412** | 100% |
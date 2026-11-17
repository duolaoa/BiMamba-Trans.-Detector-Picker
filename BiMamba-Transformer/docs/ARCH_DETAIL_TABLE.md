# Layer-by-layer architecture (per parameter preset)

Reference dump of the sub-modules and parameter counts of `LightweightPWaveNet`
for every width preset used in the paper:

| preset | used for | total trainable |
|---|---|---:|
| `pn197k` | BiMamba picker / detector (paper configuration) | 198,610 / 198,532 |
| `pm322k` | parameter-matched to PhaseMamba | — |
| `eqt466k` | parameter-matched to EQTransformer | 463,266 |
| `ctrl813k` | wider control | — |
| `abl_nomamba` | ablation: BiMamba removed | 198,562 |
| `abl_notrans` | ablation: Transformer removed | 198,078 |
| `abl_nolfem` | ablation: LFEM1D replaced by plain conv stem | 198,532 |

Regenerate with:

```bash
python dump_arch_shapes.py     # writes ARCH_DETAIL_SHAPES.csv / docs/ARCH_DETAIL_SHAPES.md
```

> The module descriptions in the tables below were emitted by the original
> annotation script and are in Chinese; module names and parameter counts are
> the authoritative part. See README §1 for the English description of the same
> architecture.

---

### `_abl199k_nolfem` — 合计 **198,532** 参数

| # | 模块 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---:|---:|
| 1 | `lfem_block1` | Conv1d(3→3,k=7)+BN+ReLU  (非 LFEM1D) | 72 | 0.0% |
| 2 | `lfem_block2` | Conv1d(3→3,k=5)+BN+ReLU  (非 LFEM1D) | 54 | 0.0% |
| 3 | `lfem_block3` | Conv1d(3→3,k=3)+BN+ReLU  (非 LFEM1D) | 36 | 0.0% |
| 4 | `proj` | Conv1d(3→16, k=1, bias=True) | 64 | 0.0% |
| 5 | `mamba` | BidirectionalMamba: d_model=16, d_state=16, d_conv=4, expand=2, layers=3 (每层 fwd+bwd+LayerNorm) + final fwd/bwd 拼接 → 2×16; Mamba 实例=8，LayerNorm=3 | 26,976 | 13.6% |
| 6 | `transformer` | TransformerEncoder: Linear(→80) + 正弦PE(10000×80, buffer) + Dropout(0.1) → 4 头, d_k=20, FFN=320, layers=2, dropout=0.1 | 157,040 | 79.1% |
| 7 | `fusion` | Linear(112→80) + ReLU + Dropout(0.1) | 9,040 | 4.6% |
| 8 | `regressor` | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| 9 | `s_regressor` | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| | **合计** | | **198,532** | 100% |

### `_abl199k_nomamba` — 合计 **198,562** 参数

| # | 模块 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---:|---:|
| 1 | `lfem_block1` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `lfem_block2` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 3 | `lfem_block3` | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 4 | `proj` | Conv1d(3→188, k=1, bias=True) | 752 | 0.4% |
| 5 | `mamba` | nn.Identity  (mamba_mode='none'，无参数) | 0 | 0.0% |
| 6 | `transformer` | TransformerEncoder: Linear(→80) + 正弦PE(10000×80, buffer) + Dropout(0.1) → 4 头, d_k=20, FFN=320, layers=2, dropout=0.1 | 170,800 | 86.0% |
| 7 | `fusion` | Linear(268→80) + ReLU + Dropout(0.1) | 21,520 | 10.8% |
| 8 | `regressor` | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| 9 | `s_regressor` | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| | **合计** | | **198,562** | 100% |

### `_abl199k_notrans` — 合计 **198,078** 参数

| # | 模块 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---:|---:|
| 1 | `lfem_block1` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `lfem_block2` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 3 | `lfem_block3` | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 4 | `proj` | Conv1d(3→54, k=1, bias=True) | 216 | 0.1% |
| 5 | `mamba` | BidirectionalMamba: d_model=54, d_state=16, d_conv=4, expand=2, layers=3 (每层 fwd+bwd+LayerNorm) + final fwd/bwd 拼接 → 2×54; Mamba 实例=8，LayerNorm=3 | 194,724 | 98.3% |
| 7 | `fusion` | Linear(108→16) + ReLU + Dropout(0.1) | 1,744 | 0.9% |
| 8 | `regressor` | Linear(16→32) + ReLU + Dropout(0.2) + Linear(32→1) | 577 | 0.3% |
| 9 | `s_regressor` | Linear(16→32) + ReLU + Dropout(0.2) + Linear(32→1) | 577 | 0.3% |
| | **合计** | | **198,078** | 100% |

### `_eqt466k` — 合计 **463,266** 参数

| # | 模块 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---:|---:|
| 1 | `lfem_block1` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `lfem_block2` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 3 | `lfem_block3` | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 4 | `proj` | Conv1d(3→56, k=1, bias=True) | 224 | 0.0% |
| 5 | `mamba` | BidirectionalMamba: d_model=56, d_state=16, d_conv=4, expand=2, layers=3 (每层 fwd+bwd+LayerNorm) + final fwd/bwd 拼接 → 2×56; Mamba 实例=8，LayerNorm=3 | 207,312 | 44.8% |
| 6 | `transformer` | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1) → 4 头, d_k=24, FFN=384, layers=2, dropout=0.1 | 229,152 | 49.5% |
| 7 | `fusion` | Linear(208→96) + ReLU + Dropout(0.1) | 20,064 | 4.3% |
| 8 | `regressor` | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 0.7% |
| 9 | `s_regressor` | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 0.7% |
| | **合计** | | **463,266** | 100% |

### `_pm322k` — 合计 **328,498** 参数

| # | 模块 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---:|---:|
| 1 | `lfem_block1` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `lfem_block2` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 3 | `lfem_block3` | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 4 | `proj` | Conv1d(3→32, k=1, bias=True) | 128 | 0.0% |
| 5 | `mamba` | BidirectionalMamba: d_model=32, d_state=16, d_conv=4, expand=2, layers=3 (每层 fwd+bwd+LayerNorm) + final fwd/bwd 拼接 → 2×32; Mamba 实例=8，LayerNorm=3 | 79,552 | 24.2% |
| 6 | `transformer` | TransformerEncoder: Linear(→96) + 正弦PE(10000×96, buffer) + Dropout(0.1) → 4 头, d_k=24, FFN=384, layers=2, dropout=0.1 | 226,848 | 69.1% |
| 7 | `fusion` | Linear(160→96) + ReLU + Dropout(0.1) | 15,456 | 4.7% |
| 8 | `regressor` | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 1.0% |
| 9 | `s_regressor` | Linear(96→32) + ReLU + Dropout(0.2) + Linear(32→1) | 3,137 | 1.0% |
| | **合计** | | **328,498** | 100% |

### `_pn197k` — 合计 **198,610** 参数

| # | 模块 | 内部结构参数 | 参数量 | 占比 |
|---|---|---|---:|---:|
| 1 | `lfem_block1` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=7) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 78 | 0.0% |
| 2 | `lfem_block2` | LFEM1D: Conv1d(3→3,k=1)+BN+SiLU → [b0 Conv1d(k=5) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✓ → Conv1d(3→3,k=1)+BN → SiLU(+残差) | 72 | 0.0% |
| 3 | `lfem_block3` | LFEM1D: Conv1d(3→4,k=1)+BN+SiLU → [b0 Conv1d(k=3) | b1 DilatedConv(k=3,d=2) | b2 DepthWise(k=3)] → shuffle✗ → Conv1d(4→3,k=1)+BN → SiLU(+残差) | 90 | 0.0% |
| 4 | `proj` | Conv1d(3→16, k=1, bias=True) | 64 | 0.0% |
| 5 | `mamba` | BidirectionalMamba: d_model=16, d_state=16, d_conv=4, expand=2, layers=3 (每层 fwd+bwd+LayerNorm) + final fwd/bwd 拼接 → 2×16; Mamba 实例=8，LayerNorm=3 | 26,976 | 13.6% |
| 6 | `transformer` | TransformerEncoder: Linear(→80) + 正弦PE(10000×80, buffer) + Dropout(0.1) → 4 头, d_k=20, FFN=320, layers=2, dropout=0.1 | 157,040 | 79.1% |
| 7 | `fusion` | Linear(112→80) + ReLU + Dropout(0.1) | 9,040 | 4.6% |
| 8 | `regressor` | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| 9 | `s_regressor` | Linear(80→32) + ReLU + Dropout(0.2) + Linear(32→1) | 2,625 | 1.3% |
| | **合计** | | **198,610** | 100% |
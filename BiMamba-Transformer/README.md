# BiMamba-Transformer

Reference implementation of **BiMamba-Transformer**, a compact hybrid
sequence model for seismic event detection and P/S phase picking, and of the
**gated sliding-window voting** decoder that turns it into an end-to-end
detection-plus-picking system.

Two networks share the same backbone width (≈ 198 k trainable parameters each):

| Network | Output | Role |
|---|---|---|
| **BiMamba picker** | two Gaussian-labelled P/S density channels | accurate arrival times inside a window |
| **BiMamba detector** | two rectangular P/S *presence* channels | record-level event gate + coarse anchor |

The picker alone is a strong in-domain picker; on long (cross-domain) records
the detector gate plus majority voting over overlapping picker windows is what
recovers the P/S arrivals. Both stages are in this repository.

---

## 1. Architecture

```
input  (B, 3, T)  [Z, N, E] , per-channel z-score
  │
  ├─ LFEM1D × 3        (k = 7 / 5 / 3, three parallel branches, channel-shuffle, residual)
  │    + MaxPool × 2 for the picker      → T/4
  │    + MaxPool × 3 for the detector    → T/8
  │
  ├─ Conv1d(3 → 16, k = 1)                    proj
  │
  ├─ BidirectionalMamba × 3                   d_model = 16, d_state = 16, expand = 2
  │    forward ∥ backward, LayerNorm, concat  → 32
  │
  ├─ Transformer encoder (80-d, 4 heads, 2 layers, FFN 320, sinusoidal PE)
  │
  ├─ concat [ 32 , 80 ] → 112
  ├─ Linear(112 → 80) + ReLU + Dropout(0.1)   fusion
  └─ two heads: Linear(80 → 32) + ReLU + Dropout + Linear(32 → 1)
```

Parameter budget (verified by instantiating the released checkpoints):

| Component | Params |
|---|---:|
| LFEM1D × 3 (picker) | 240 |
| `proj` | 64 |
| BidirectionalMamba × 3 | 26,976 |
| Transformer encoder | 157,040 |
| `fusion` | 9,040 |
| two output heads | 5,250 |
| **picker total (trainable)** | **198,610** |
| **detector total (trainable)** | **198,532** |

The sinusoidal positional encoding is stored as a non-trainable buffer
(10,000 × 80 = 800,000 entries), which is why a raw `state_dict` looks ~1 M.
`--preset` switches the width to match each baseline's parameter count
(`pn197k`, `pm322k`, `eqt466k`, `ctrl813k`) and provides the three ablation
variants (`abl_nomamba`, `abl_notrans`, `abl_nolfem`).

Losses:

* picker — Gaussian focal loss (`kl_weight=1.0`, `soft_argmax_weight=5.0`,
  `focal_weight=2.0`) plus a joint P/S term (`p=1.0`, `s=0.5`);
* detector — `BCEWithLogits` on the two rectangular presence channels with
  `pos_weight=3.0` and an explicit all-zero supervision on noise windows.

---

## 2. Repository layout

Everything is flat so that the module imports (`import eval_evsplit_models as E`,
`from event_split import ...`, `import train_phasenet_evsplit as PN`) work
without installing a package.

### Models and training

| File | Purpose |
|---|---|
| `PS_Detection_evsplit.py` | all model components: `LFEM1D`, `BidirectionalMamba`, `TransformerEncoder`, `LightweightPWaveNet`, `DetectionOnlyLoss`, augmentations, dataset |
| `train_bimamba_evsplit.py` | BiMamba picker training (this is the paper's picker) |
| `train_detection_evsplit.py` | BiMamba detector training (rectangular P/S presence heads) |
| `train_phasenet_evsplit.py` | PhaseNet baseline |
| `train_eqt_evsplit.py` | EQTransformer baseline |
| `train_phasemamba_evsplit.py` | PhaseMamba baseline |

### Data splitting

| File | Purpose |
|---|---|
| `event_split.py` | event-level train/val/test split (all models share one split table) |
| `verify_split.py` | verifies zero event leakage; can also report the leakage of the naive trace-level split |

### Evaluation

| File | Purpose |
|---|---|
| `eval_evsplit_models.py` | unified window-level picking + sliding-window detection evaluation for all models and datasets |
| `eval_gate_vote.py` | **gate → anchor → region of interest → majority voting** decoder |
| `eval_gate_vote_subbin.py` | same, with sub-bin refinement of the modal bin |
| `eval_e2e_anchors.py` | compares anchor read-outs (earliest above τ vs. global maximum) |

### Threshold calibration

| File | Purpose |
|---|---|
| `tau_calib.py` | compares cross-fitted τ against a held-out-calibration-set τ |
| `tau_sweep_cross.py` | τ sensitivity sweep on the cross-domain sets |
| `tau_sweep_q.py` | quantile-based τ sweep (operating-point analysis) |
| `calib_75.py`, `calib_75_b_fix.py` | 25 % calibration / 75 % test protocol used for all cross-domain numbers |

### Ablation and analyses

| File | Purpose |
|---|---|
| `abl_seeds.py` | ablation over 5 **split seeds** (all seeds reported, mean ± std) |
| `abl_recall.py` | system-level recall vs. gated hit rate (end-to-end bookkeeping) |
| `abl_sigma.py` | per-variant σ of the signed residual |
| `abl_2575.py` | ablation under the 25/75 protocol |
| `snr_full_table.py`, `snr_all_models.py`, `snr_error.py` | per-SNR-bin accuracy tables |
| `pos_sensitivity.py` | distribution of the true P position inside the training window |
| `onestage_baseline.py` | single-stage (own arrival head) vs. two-stage (gate + voting) comparison |
| `s_region_response.py` | picker response inside the S voting region |
| `window_level_table.py`, `audit_window_level.py` | window-level vs. end-to-end consistency audit |

### Long records (CREW)

| File | Purpose |
|---|---|
| `crew_e2e_2575.py` | CREW under the 25/75 protocol, end-to-end (gate → ROI → voting) |
| `crew_longseq_eval.py` | CREW window-splicing and full-record sliding schemes |
| `crew_e2e_bimamba.py` | two-stage CREW baseline with anchor read-outs |
| `crew_vote_pick.py` | voting-only CREW picking |

### Figures

`plot_snr_bars2.py`, `plot_snr_two_variants.py` (per-SNR accuracy),
`plot_gap_curve.py` (S−P gap vs. hit rate), `plot_tau_sens2.py` (τ sensitivity),
`plot_errdist12.py`, `plot_err_hist.py` (error histograms),
`plot_picking_v2.py` (picking examples), `plot_det_examples.py`,
`plot_det_noise_fp.py` (detection examples, including a false-alarm case),
`dump_arch_shapes.py` (layer-by-layer shapes; writes `docs/ARCH_DETAIL_SHAPES.md`).

---

## 3. Installation

```bash
conda create -n bimamba python=3.10 -y
conda activate bimamba

# CUDA build of PyTorch that matches your driver, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`mamba_ssm` needs `causal-conv1d` and compiles against the installed CUDA
toolkit. Install them **after** torch:

```bash
pip install causal-conv1d
pip install mamba-ssm
```

Reference environment used for the paper: Python 3.10.20, torch 2.9.0+cu128,
`mamba_ssm` 2.3.2.post1, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, h5py 3.16.0.

---

## 4. Data

No data are redistributed here. Download each dataset from its original source:

| Dataset | Source |
|---|---|
| **STEAD** | Mousavi et al., *IEEE Access* 7 (2019) — `stead` HDF5 chunks + metadata CSVs |
| **INSTANCE** | Michelini et al., *Earth Syst. Sci. Data* 13 (2021) |
| **CREW** | Aguilar Suarez & Beroza, *Seismica* 3 (2024) |
| **EQS-Denoiser** | Dahmen, *BSSA* (2025) — used as an extra 0 dB-noise benchmark |

Point the code at your copy with environment variables (no code editing needed):

```bash
export SEISMIC_DATA_ROOT=/path/to/seismic_data
# expect:
#   $SEISMIC_DATA_ROOT/STEAD_data/1Dsignal/{chunk1..chunk4}.hdf5  +  *.csv
#   $SEISMIC_DATA_ROOT/INSTANCE_DATASET/
#   $SEISMIC_DATA_ROOT/CREW_Dataset_files/000_CREW.hdf5
```

Individual overrides: `STEAD_DIR`, `INSTANCE_DIR`, `CREW_DIR`, `CREW_H5`.
Figure output goes to `$FIG_DIR` (default `./figures`), evaluation output to
`./eval_all`.

STEAD chunk roles used throughout:

| chunk | role |
|---|---|
| `chunk1` | noise (detector / PhaseNet / EQTransformer negatives) |
| `chunk2`, `chunk3` | events (train / val / test) |
| `chunk4` | waveform pool for the second-event augmentation, with val/test events removed |

Checkpoints expected in the repository root (see §7 for download links):

```
best_lightweight_model_3C_pn197k_w6000v3.pth   picker
best_detection_model_3C_det_pn197k.pth         detector
best_phasenet_baseline_6000.pth                PhaseNet
best_eqtransformer_baseline.pth                EQTransformer (3,000-sample window)
best_phasemamba_evsplit__phasemamba6000.pth    PhaseMamba
scaler_3C.pkl, scaler_3C_detection.pkl         normalisation statistics
```

---

## 5. Training

### 5.1 Event-level split

Records of the same earthquake must not appear in both training and test sets.
`event_split.py` groups the metadata by event *before* sampling, and merges the
chunk metadata first because STEAD `source_id` is not unique across chunks
(chunk2 and chunk3 share 39,420 events). With a naive trace-level split, 65.1 %
of the test traces belong to events also seen in training; with the event-level
split this is 0 %.

```bash
python event_split.py --data-dir "$SEISMIC_DATA_ROOT/STEAD_data/1Dsignal" \
    --chunks chunk2 chunk3 --pool-chunk chunk4 \
    --ratios 0.8,0.15,0.05 --out event_split_table.csv
python verify_split.py --split-table event_split_table.csv
```

### 5.2 Pickers

The paper's picker (`pn197k` preset, 6,000-sample window):

```bash
python train_bimamba_evsplit.py \
    --split-table event_split_table.csv \
    --preset pn197k --window-size 6000 \
    --tag _pn197k_w6000v3 --epochs 100
```

The detector (rectangular presence heads, same width):

```bash
python train_detection_evsplit.py \
    --split-table event_split_table.csv \
    --train-files "$D/chunk2.hdf5,$D/chunk2.csv" "$D/chunk3.hdf5,$D/chunk3.csv" \
    --noise-files "$D/chunk1.hdf5,$D/chunk1.csv" \
    --pool-files  "$D/chunk4.hdf5,$D/chunk4.csv"
```

Baselines use the same split table, so the comparison is like-for-like:

```bash
python train_phasenet_evsplit.py --split-table event_split_table.csv \
    --window-size 6000 --tag _6000
python train_eqt_evsplit.py      --split-table event_split_table.csv \
    --noise-chunks chunk1 --tag _6000
python train_phasemamba_evsplit.py --split-table event_split_table.csv
```

`python train_bimamba_evsplit.py --dry-run` builds and prints the splits
without training — useful to confirm the data layout first.

`scripts/train_all.sh` runs the whole chain.

---

## 6. Evaluation

### 6.1 Window-level picking and sliding detection

```bash
# picking, all datasets / all models
python eval_evsplit_models.py --datasets INSTANCE STEAD-DET EQS-D-0dB \
    --models BiMamba-Picker PhaseNet EQTransformer PhaseMamba

# sliding-window detection only (writes detscore_*.csv)
python eval_evsplit_models.py --datasets STEAD-DET --models BiMamba-Detector \
    --detection --skip-picking --stride 1000 --out out_det
```

`--threshold-mode crossfit|oracle|fixed` selects how τ is obtained; `--limit`
truncates each dataset for a smoke test.

### 6.2 Gate → anchor → region of interest → voting

The decoder is configured through environment variables:

```bash
PICKER_MODEL=bimamba \
EVAL_PICKER_CKPT=$PWD/best_lightweight_model_3C_pn197k_w6000v3.pth \
EVAL_DETSCORE_DIR=out_det \
EVAL_DET_KEY=BiMamba-Detector \
EVAL_DATASETS=INSTANCE \
EVAL_TAG=_pn197k_w6000v3 \
python eval_gate_vote.py
```

| Variable | Meaning | Default |
|---|---|---|
| `BACK`, `FWD` | region of interest: `[t_a − BACK, t_a + FWD]` | 4000, 2000 samples |
| `STRIDE` | picker stride inside the ROI | 250 |
| `BINW` | histogram bin width for the vote | 10 samples |
| `GATE_TAU` | gate threshold (unset ⇒ cross-fitted) | — |

### 6.3 Threshold protocol

Two regimes are used and reported separately:

* **in-domain (STEAD)** — τ is taken from the held-out split of the training
  pipeline and then frozen; evaluation runs on the full STEAD-DET set
  (10,000 events with P and S labels, 5,000 noise traces).
* **cross-domain (INSTANCE, CREW, EQS-Denoiser)** — τ is calibrated on a
  **25 % split of the target-domain labels, partitioned by event id**, and
  applied unchanged to the remaining 75 %. The evaluation records never
  contribute to threshold selection:

  ```
  τ* = argmax_τ F1( E_cal , N_cal ; τ )
  ```

CREW contains event records only, so it has no ambient-noise traces. Its
negatives are therefore built from the same records: a window lying at least
Δ away from both catalogue arrivals counts as a negative. The resulting τ is
*stricter* than a τ calibrated on genuine noise, so CREW detection figures
should be read as an internal consistency check rather than as an absolute
false-alarm rate.

`calib_75.py` implements this protocol; `tau_calib.py` compares it against
cross-fitting; `tau_sweep_q.py` and `tau_sweep_cross.py` give the sensitivity
of the operating point.

---

## 7. Reproducing the paper

Numbering follows the manuscript; the item names are given as well so the
mapping stays valid if the numbering shifts during revision.

| Paper item | How to obtain it |
|---|---|
| Detector tables (Tables 5–7) | `eval_evsplit_models.py --detection --skip-picking --stride 1000` → `detscore_*.csv`, then `calib_75.py` for the τ of each model |
| End-to-end picking tables (Tables 8–10) | `eval_gate_vote.py` with `BACK=4000 FWD=2000 STRIDE=250 BINW=10`; the MAE/σ columns come from `eval_evsplit_models.py` |
| Ablation tables | `abl_seeds.py` (5 split seeds, mean ± std), `abl_sigma.py`, `abl_recall.py`, `abl_2575.py` |
| Architecture figure | `dump_arch_shapes.py` → `docs/ARCH_DETAIL_SHAPES.md` |
| Per-SNR accuracy figures (STEAD / INSTANCE, P and S) | `snr_full_table.py` → `plot_snr_bars2.py`, `plot_snr_two_variants.py` |
| Picking examples (STEAD + INSTANCE) | `plot_picking_v2.py` |
| Detection examples, incl. a baseline false alarm | `plot_det_examples.py`, `plot_det_noise_fp.py` |
| S−P gap figure (CREW) | `plot_gap_curve.py` |
| τ sensitivity figure | `tau_sweep_q.py` → `plot_tau_sens2.py` |
| Error histograms / distributions | `eval_evsplit_models.py` → `plot_errdist12.py`, `plot_err_hist.py` |
| CREW long-record analysis (Table 13) | `crew_e2e_2575.py`, `crew_longseq_eval.py` |

`scripts/reproduce_tables.sh` chains the common cases.

### Note on comparability

Numbers produced by `eval_evsplit_models.py` are **window-level** (the window is
placed on the true arrival), whereas `eval_gate_vote.py` is **end-to-end**
(detector gate + anchor + voting). They are not interchangeable: on a 300 s CREW
record the same picker that reaches ≈ 80 % of P picks within 0.1 s in a
truth-placed 6,000-sample window reaches ≈ 27 % end-to-end within 0.5 s,
because a mis-placed anchor moves the arrival outside the region of interest.
Always state which protocol a table uses.

---

## 8. Pretrained weights

Weights are not tracked in this repository. See **[WEIGHTS.md](WEIGHTS.md)** for
the exact file list, sizes, SHA-256 prefixes and which script needs which file.

Short version — **7 files, ≈ 12 MB** reproduce Tables 5–10:

```
best_lightweight_model_3C_pn197k_w6000v3.pth    picker,     W = 6000
best_detection_model_3C_det_pn197k.pth          detector,   W = 6000
best_phasenet_baseline_6000.pth                 PhaseNet,   W = 6000
best_eqtransformer_baseline.pth                 EQT,        W = 3000
best_phasemamba_evsplit__phasemamba6000.pth     PhaseMamba, W = 6000
scaler_3C.pkl                                   picker normalisation   (in repo)
scaler_3C_detection.pkl                         detector normalisation (in repo)
```

Drop them in the repository root; these names are the **defaults** in
`eval_evsplit_models.CKPT`, so no environment variable is needed. Every entry
can still be redirected with `EVAL_PICKER_CKPT`, `EVAL_DET_CKPT`,
`EVAL_PHASENET_CKPT`, `EVAL_EQT_CKPT`, `EVAL_PHASEMAMBA_CKPT`,
`EVAL_DET_SCALER`.

The three ablation checkpoints (`..._abl6000_{nomamba,notrans,nolfem}.pth`) are
listed in WEIGHTS.md Tier 2; the parameter-matched controls in Tier 3.

---

## 9. Citation

```bibtex
@article{biMambaTransformer,
  title   = {BiMamba-Transformer: A Compact Hybrid Sequence Model for Seismic
             Event Detection and Phase Picking},
  author  = {Luo, Chenghao and Cheng, Hao and Shen, Junhai and Gao, Xiaoping},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026}
}
```

Please also cite the datasets (STEAD, INSTANCE, CREW, EQS-Denoiser) and the
`mamba-ssm` implementation this code builds on.

---

## 10. License

MIT — see [LICENSE](LICENSE).

## 11. Acknowledgements

The backbone follows the Mamba formulation of Gu & Dao (2023); the PhaseNet,
EQTransformer and PhaseMamba baselines are re-implementations trained here under
the same event-level split for a controlled comparison, and are not the
authors' original code.

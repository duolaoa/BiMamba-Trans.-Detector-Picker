# Checkpoints

The repository ships **no weights** (`.gitignore` excludes `*.pth`), because a
GitHub repository should stay diff-friendly. All checkpoints are attached to the
**release / Zenodo record** of this repository; drop them into the repository
root and the code works without any environment variable, since the defaults in
`eval_evsplit_models.CKPT` are exactly the file names below.

Total: **7 files / ≈ 12 MB** are needed for the main tables; the rest is optional.

---

## Tier 1 — required for Tables 5–10 and the picking/detection figures

| File | Size | Network | Window | Params | Used by |
|---|---:|---|---:|---:|---|
| `best_lightweight_model_3C_pn197k_w6000v3.pth` | 3.97 MB | **BiMamba picker** (paper config) | 6000 | 198,610 | `eval_evsplit_models.py` (`BiMamba-Picker`), `eval_gate_vote.py`, `scripts/*.sh` |
| `best_detection_model_3C_det_pn197k.pth` | 3.95 MB | **BiMamba detector** (paper config) | 6000 | 198,532 | `eval_evsplit_models.py --detection` (`BiMamba-Detector`) |
| `best_phasenet_baseline_6000.pth` | 0.82 MB | PhaseNet baseline | 6000 | 198,117 | `--models PhaseNet` |
| `best_eqtransformer_baseline.pth` | 2.04 MB | EQTransformer baseline | **3000** | 469,067 | `--models EQTransformer` |
| `best_phasemamba_evsplit__phasemamba6000.pth` | 1.33 MB | PhaseMamba baseline | 6000 | 322,003 | `--models PhaseMamba` |
| `scaler_3C.pkl` | 298 B | picker normalisation statistics | — | — | already in the repo |
| `scaler_3C_detection.pkl` | 488 B | detector normalisation statistics | — | — | already in the repo |

**Note on windows.** PhaseNet and PhaseMamba are evaluated with a 6,000-sample
window, EQTransformer with 3,000: its 6,000-sample variant does not converge
(detection F1 drops from 0.998 to 0.948). This is the configuration disclosed
in the paper, and the numbers in Tables 5–10 assume it.

---

## Tier 2 — required for the ablation tables

| File | Size | Variant | Params |
|---|---:|---|---:|
| `best_lightweight_model_3C_abl6000_nomamba.pth` | 4.03 MB | BiMamba removed | 198,562 |
| `best_lightweight_model_3C_abl6000_notrans.pth` | 0.85 MB | Transformer removed | 198,078 |
| `best_lightweight_model_3C_abl6000_nolfem.pth` | 4.05 MB | LFEM1D → plain conv stem | 198,532 |

Used by `abl_seeds.py`, `abl_recall.py`, `abl_sigma.py`, `abl_2575.py`
(variant tags `abl6000_nomamba` / `abl6000_notrans` / `abl6000_nolfem`).
The ablation is evaluated through `eval_gate_vote.py` with

```bash
EVAL_PICKER_CKPT=$PWD/best_lightweight_model_3C_abl6000_notrans.pth \
EVAL_TAG=_abl6000_notrans ...
```

---

## Tier 3 — optional, only needed by specific analysis / figure scripts

| File | Size | Used by |
|---|---:|---|
| `best_eqtransformer_baseline_6000.pth` | 2.04 MB | `plot_picking_v2.py`, `plot_det_examples.py`, `plot_det_noise_fp.py`, `crew_e2e_2575.py`, `onestage_baseline.py` |
| `best_lightweight_model_3C_pn197k.pth` | 4.06 MB | `s_region_response.py` (3,000-sample control) |
| `best_lightweight_model_3C_pm322k.pth` | 5.22 MB | `s_region_response.py` (PhaseMamba-matched width) |
| `best_detection_model_3C_wofocal.pth` | 7.15 MB | `crew_e2e_bimamba.py` (focal term disabled) |

> ⚠️ **Known inconsistency to resolve before submission.** The example-figure
> scripts hardcode `best_eqtransformer_baseline_6000.pth`, whereas the detector
> and picking tables use the 3,000-sample EQTransformer
> (`best_eqtransformer_baseline.pth`). The two are different networks. Either
> regenerate the figures with `best_eqtransformer_baseline.pth` (recommended, so
> figure and table show the same model) or state the difference in the caption.

---

## Loading and verification

Every checkpoint is a plain `dict` loadable with

```python
import torch
ck = torch.load('best_lightweight_model_3C_pn197k_w6000v3.pth',
                map_location='cpu', weights_only=False)
# keys: 'model_state_dict', 'cfg'  (cfg carries proj_dim / hidden_dim /
#       mamba_layers / d_state / expand / window)
```

The model is rebuilt from `cfg` stored inside the checkpoint, so no manual
architecture flags are needed.

Sizes above are the exact byte counts of the released files. To verify a
download, the first 16 hex digits of the SHA-256 are:

| File | SHA-256 (prefix) |
|---|---|
| `best_lightweight_model_3C_pn197k_w6000v3.pth` | `d5c5f853b7ae4a30` |
| `best_detection_model_3C_det_pn197k.pth` | `0b3ea8b77c21a4ab` |
| `best_phasenet_baseline_6000.pth` | `12d5307a758fd28b` |
| `best_eqtransformer_baseline.pth` | `5e7026de25a2ccf4` |
| `best_phasemamba_evsplit__phasemamba6000.pth` | `9c934fe31b492d8c` |
| `best_lightweight_model_3C_abl6000_nomamba.pth` | `c458cc23d210e50f` |
| `best_lightweight_model_3C_abl6000_notrans.pth` | `d637d51786ad62b2` |
| `best_lightweight_model_3C_abl6000_nolfem.pth` | `90adadb7dd21bd7e` |
| `best_eqtransformer_baseline_6000.pth` | `0b74c9039a1a882b` |
| `best_lightweight_model_3C_pn197k.pth` | `42b83d5ffec326ae` |
| `best_lightweight_model_3C_pm322k.pth` | `a44380957f8b67b3` |
| `best_detection_model_3C_wofocal.pth` | `3e269af4832b1411` |

```bash
sha256sum -c <<'EOF'
d5c5f853b7ae4a30...  best_lightweight_model_3C_pn197k_w6000v3.pth
EOF
```

---

## Retraining instead of downloading

Nothing here is required if you retrain — see README §5. The picker and detector
in Tier 1 are produced by

```bash
python train_bimamba_evsplit.py   --preset pn197k --window-size 6000 --tag _pn197k_w6000v3
python train_detection_evsplit.py --train-files ... --noise-files ... --pool-files ...
```

with the seed fixed by `--seed 42` and the shared `event_split_table.csv`;
`train_*_evsplit.py` writes `scaler_3C.pkl` (and the detector writes
`scaler_3C_detection.pkl`) next to the checkpoint, so the normalisation travels
with the weights.

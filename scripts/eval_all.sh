#!/usr/bin/env bash
# Window-level picking + sliding detection for every model on every dataset.
#
#   bash scripts/eval_all.sh
#
# Produces, per dataset and model:
#   <out>/picking_<dataset>_<model>.csv     per-trace P/S errors (window-level)
#   <out>/detscore_<dataset>_<model>.csv    per-record detection score
# Override paths with SEISMIC_DATA_ROOT and EVAL_OUT.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${EVAL_OUT:-eval_all}"
DET_OUT="${EVAL_DET_OUT:-eval_all_det}"
DATASETS="${EVAL_DATASETS:-STEAD-DET INSTANCE EQS-D-0dB EQS-D-imp}"
CK_PICK="${EVAL_PICKER_CKPT:-$PWD/best_lightweight_model_3C_pn197k_w6000v3.pth}"
CK_DET="${EVAL_DET_CKPT:-$PWD/best_detection_model_3C_det_pn197k.pth}"
mkdir -p "$OUT" "$DET_OUT"

echo "== 1/3  window-level picking (all models) =="
EVAL_WINDOW=6000 EVAL_PICKER_CKPT="$CK_PICK" \
python -u eval_evsplit_models.py \
    --datasets $DATASETS \
    --models BiMamba-Picker PhaseNet EQTransformer PhaseMamba \
    --out "$OUT"

echo "== 2/3  sliding-window detection, BiMamba detector =="
EVAL_WINDOW=6000 EVAL_DET_CKPT="$CK_DET" \
python -u eval_evsplit_models.py \
    --datasets $DATASETS --models BiMamba-Detector \
    --detection --skip-picking --stride 1000 --out "$DET_OUT"

echo "== 3/3  sliding-window detection, baselines' own presence heads =="
EVAL_WINDOW=6000 \
python -u eval_evsplit_models.py \
    --datasets $DATASETS \
    --models PhaseNet EQTransformer PhaseMamba \
    --detection --skip-picking --stride 1000 --out "$DET_OUT"

echo "done. picking -> $OUT   detection -> $DET_OUT"

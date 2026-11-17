#!/usr/bin/env bash
# Reproduce the end-to-end (gate -> anchor -> ROI -> voting) numbers and the
# threshold / ablation tables.
#
#   bash scripts/reproduce_tables.sh
#
# Assumes scripts/eval_all.sh has been run, so that detscore_*.csv exist.
set -euo pipefail
cd "$(dirname "$0")/.."

DET_OUT="${EVAL_DET_OUT:-eval_all_det}"
CK_PICK="${EVAL_PICKER_CKPT:-$PWD/best_lightweight_model_3C_pn197k_w6000v3.pth}"
TAG="_pn197k_w6000v3"
mkdir -p eval_all

echo "== 1/4  gate + voting (paper decoder: BACK=4000 FWD=2000 STRIDE=250 BINW=10) =="
for DS in INSTANCE EQS-D-0dB EQS-D-imp STEAD-DET; do
  f="eval_all/gatevote_${DS}${TAG}.csv"
  if [ -f "$f" ]; then echo "  $DS: exists, skip"; continue; fi
  PICKER_MODEL=bimamba \
  EVAL_PICKER_CKPT="$CK_PICK" \
  EVAL_DETSCORE_DIR="$DET_OUT" \
  EVAL_DET_KEY=BiMamba-Detector \
  EVAL_DATASETS="$DS" \
  EVAL_TAG="$TAG" \
  BACK=4000 FWD=2000 STRIDE=250 BINW=10 \
  python -u eval_gate_vote.py
  echo "  $DS: done -> $f"
done

echo "== 2/4  25% calibration / 75% test threshold tables =="
python -u calib_75.py
python -u calib_75_b_fix.py

echo "== 3/4  ablation: 5 split seeds, mean +- std =="
python -u abl_seeds.py
python -u abl_sigma.py
python -u abl_recall.py

echo "== 4/4  tau sensitivity =="
python -u tau_calib.py
python -u tau_sweep_q.py

echo "done. tables in ./eval_all"

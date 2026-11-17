#!/usr/bin/env bash
# Train the detector and all pickers on one shared event-level split table.
#
#   bash scripts/train_all.sh
#
# Override the data location with SEISMIC_DATA_ROOT (see README §4).
# Extra arguments are forwarded to every training script.
set -euo pipefail
cd "$(dirname "$0")/.."

D="${SEISMIC_DATA_ROOT:-$HOME/seismic_data}/STEAD_data/1Dsignal"
TABLE="${EVSPLIT_TABLE:-event_split_table.csv}"
EXTRA="${*:-}"

echo "data dir : $D"
echo "split    : $TABLE"
echo "extra    : $EXTRA"

# 0) event-level split table (idempotent: reuse if present)
if [ -f "$TABLE" ]; then
  echo "[0/5] reusing $TABLE"
else
  echo "[0/5] building $TABLE"
  python event_split.py --data-dir "$D" --chunks chunk2 chunk3 \
      --pool-chunk chunk4 --ratios 0.8,0.15,0.05 --out "$TABLE"
fi
python verify_split.py --split-table "$TABLE" | tail -12

# 1) detector (rectangular P/S presence heads)
echo "[1/5] detector"
python -u train_detection_evsplit.py --split-table "$TABLE" \
    --train-files "$D/chunk2.hdf5,$D/chunk2.csv" "$D/chunk3.hdf5,$D/chunk3.csv" \
    --noise-files "$D/chunk1.hdf5,$D/chunk1.csv" \
    --pool-files  "$D/chunk4.hdf5,$D/chunk4.csv" $EXTRA

# 2) BiMamba picker (paper configuration: pn197k width, 6,000-sample window)
echo "[2/5] BiMamba picker"
python -u train_bimamba_evsplit.py --split-table "$TABLE" \
    --preset pn197k --window-size 6000 --tag _pn197k_w6000v3 $EXTRA

# 3) PhaseNet baseline
echo "[3/5] PhaseNet"
python -u train_phasenet_evsplit.py --split-table "$TABLE" \
    --window-size 6000 --tag _6000 $EXTRA

# 4) EQTransformer baseline (3,000-sample window)
echo "[4/5] EQTransformer"
python -u train_eqt_evsplit.py --split-table "$TABLE" \
    --noise-chunks chunk1 $EXTRA

# 5) PhaseMamba baseline
echo "[5/5] PhaseMamba"
python -u train_phasemamba_evsplit.py --split-table "$TABLE" $EXTRA

echo "done. split table: $TABLE"

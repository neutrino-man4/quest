#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/h5_maker_slim.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

JET_TYPES=("HToBB" "HToCC" "HToGG" "HToWW4Q" "TTBar_" "TTBarLep" "WToQQ" "ZJetsToNuNu" "ZToQQ")
DATASET_TYPES=("train" "val" "test")

for dataset_type in "${DATASET_TYPES[@]}"; do
    for jet_type in "${JET_TYPES[@]}"; do
        log="$LOG_DIR/${dataset_type}_${jet_type}.log"
        echo "[$dataset_type] $jet_type -> $log"
        python "$PYTHON_SCRIPT" --jet_type "$jet_type" --dataset_type "$dataset_type" \
            > "$log" 2>&1
    done
done

echo "All done."

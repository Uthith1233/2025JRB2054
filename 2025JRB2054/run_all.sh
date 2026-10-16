#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=============================================="
echo "ELL8299 Assignment 1 - Repository Check"
echo "=============================================="

for f in \
    src/peft_sweep.py \
    src/sanitize_eval.py \
    src/train_victim.py \
    src/attack_extract.py \
    src/download_adapters.py
do
    if [ -f "$f" ]; then
        echo "OK: $f"
    else
        echo "MISSING: $f"
        exit 1
    fi
done

echo ""
echo "Checking Python syntax..."
python -m py_compile \
    src/peft_sweep.py \
    src/sanitize_eval.py \
    src/train_victim.py \
    src/attack_extract.py \
    src/download_adapters.py

echo "Python syntax: OK"
echo ""
echo "=============================================="
echo "Downloading adapters from Hugging Face"
echo "=============================================="

python src/download_adapters.py

echo ""
echo "Verifying downloaded adapter directories..."
for adapter in best_sanitizer victim_raw victim_san
do
    if [ -d "adapters/$adapter" ] && [ -n "$(find "adapters/$adapter" -type f -print -quit)" ]; then
        echo "OK: adapters/$adapter"
    else
        echo "MISSING or EMPTY: adapters/$adapter"
        exit 1
    fi
done

echo ""
echo "Repository check and adapter download completed successfully."
echo "Note: This script does not run the full training or evaluation pipeline."

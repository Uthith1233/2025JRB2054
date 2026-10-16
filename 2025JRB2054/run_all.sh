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
  src/attack_extract.py
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
  src/attack_extract.py

echo ""
echo "Python syntax: OK"
echo "Repository check completed successfully."

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
BACKBONE="${BACKBONE:-checkpoints/dinov2_converted_512x512.pth}"
export PYTHONNOUSERSITE=1
mkdir -p work_dirs/test

sources=(citys bdd map)
checkpoints=(
  checkpoints/citys_samo_plus_68.15.pth
  checkpoints/bdd_samo_plus_69.34.pth
  checkpoints/map_samo_plus_70.64.pth
)

for i in "${!sources[@]}"; do
  source_name="${sources[i]}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/test.py \
    --config "configs/samo_plus_v3/${source_name}_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py" \
    --checkpoint "${checkpoints[i]}" \
    --backbone "${BACKBONE}" \
    --work-dir "work_dirs/test/${source_name}" \
    --resize-width 1024 \
    --resize-height 512 \
    --crop-size 512 \
    --stride 341 \
    --seed 3407 2>&1 | tee "work_dirs/test/${source_name}.log"
done

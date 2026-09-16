#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
RESUME="${RESUME:-0}"
export PYTHONNOUSERSITE=1

resume_args=()
if [[ "${RESUME}" == "1" ]]; then
  resume_args+=(--resume)
fi

for source_name in citys bdd map; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/train.py \
    --config "configs/samo_plus_v3/${source_name}_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py" \
    --work-dir "work_dirs/train/${source_name}" \
    "${resume_args[@]}"
done

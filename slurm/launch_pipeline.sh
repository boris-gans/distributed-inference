#!/usr/bin/env bash
# Helper to submit the single pipeline-parallel job with fixed configs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="${ROOT_DIR}/slurm/submit.sbatch"

# Host-side paths for configs and outputs (override via env as needed).
# Use scratch for runtime data to follow cluster guidance.
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/${USER}/scratch/group1}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${SCRATCH_ROOT}/pipeline_run}"
EXP_CONFIG_PATH="${EXP_CONFIG_PATH:-${PIPELINE_ROOT}/exp_config.json}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-${PIPELINE_ROOT}/ds_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PIPELINE_ROOT}/outputs}"

mkdir -p "${PIPELINE_ROOT}" "${OUTPUT_DIR}" "${SCRATCH_ROOT}/hpc-runs"

if [[ -z "${APPAINTER_IMAGE:-}" ]]; then
  APPAINTER_IMAGE="${SCRATCH_ROOT}/appainter/appainter.sif"
fi
if [[ ! -f "${APPAINTER_IMAGE}" ]]; then
  echo "APPAINTER_IMAGE is not set or not found at ${APPAINTER_IMAGE}. Export it to your container image before submitting." >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available. Run this on your Slurm login node." >&2
  exit 1
fi

echo "Submitting single pipeline job..."
PIPELINE_ROOT="${PIPELINE_ROOT}" \
SCRATCH_ROOT="${SCRATCH_ROOT}" \
PROJECT_ROOT="${PROJECT_ROOT:-${PIPELINE_ROOT}}" \
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PIPELINE_ROOT}}" \
EXP_CONFIG_PATH="${EXP_CONFIG_PATH}" \
DS_CONFIG_PATH="${DS_CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
APPAINTER_IMAGE="${APPAINTER_IMAGE}" \
sbatch "${SBATCH_SCRIPT}"

echo "Submitted. Check your Slurm queue or output files under ${OUTPUT_DIR}."

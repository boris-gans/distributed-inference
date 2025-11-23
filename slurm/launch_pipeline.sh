#!/usr/bin/env bash
# Helper to submit the single pipeline-parallel job with fixed configs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="${ROOT_DIR}/slurm/submit.sbatch"

# Host-side paths for configs and outputs (override via env as needed).
PIPELINE_ROOT="${PIPELINE_ROOT:-${ROOT_DIR}/experiments/pipeline_run}"
EXP_CONFIG_PATH="${EXP_CONFIG_PATH:-${PIPELINE_ROOT}/exp_config.json}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-${PIPELINE_ROOT}/ds_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PIPELINE_ROOT}/outputs}"

mkdir -p "${PIPELINE_ROOT}" "${OUTPUT_DIR}"

if [[ -z "${APPAINTER_IMAGE:-}" ]]; then
  echo "APPAINTER_IMAGE is not set. Export it to your container image before submitting." >&2
  exit 1
fi

# Create minimal placeholder configs if they don't already exist.
if [[ ! -f "${EXP_CONFIG_PATH}" ]]; then
  cat > "${EXP_CONFIG_PATH}" <<'EOF'
{
  "model_name": "meta-llama/Llama-3.3-70B-Instruct",
  "prompt_variant": "2k",
  "batch_size": 4
}
EOF
  echo "Wrote placeholder exp_config.json to ${EXP_CONFIG_PATH}"
fi

if [[ ! -f "${DS_CONFIG_PATH}" ]]; then
  cat > "${DS_CONFIG_PATH}" <<'EOF'
{
  "train_batch_size": 4,
  "pipeline_parallel_size": 2,
  "tensor_parallel_size": 2
}
EOF
  echo "Wrote placeholder ds_config.json to ${DS_CONFIG_PATH}"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available. Run this on your Slurm login node." >&2
  exit 1
fi

echo "Submitting single pipeline job..."
PIPELINE_ROOT="${PIPELINE_ROOT}" \
PROJECT_ROOT="${PROJECT_ROOT:-${PIPELINE_ROOT}}" \
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PIPELINE_ROOT}}" \
EXP_CONFIG_PATH="${EXP_CONFIG_PATH}" \
DS_CONFIG_PATH="${DS_CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
APPAINTER_IMAGE="${APPAINTER_IMAGE}" \
sbatch "${SBATCH_SCRIPT}"

echo "Submitted. Check your Slurm queue or output files under ${OUTPUT_DIR}."

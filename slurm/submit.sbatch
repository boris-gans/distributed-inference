#!/bin/bash
#SBATCH --job-name=llama_pipeline
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/hpc-runs/%x-%j.out
#SBATCH --error=/home/%u/hpc-runs/%x-%j.err

# submit.sbatch
# Single pipeline-parallel launch with fixed config paths.
# Usage: sbatch submit.sbatch (or via slurm/launch_pipeline.sh)

set -euo pipefail

#############################################
# Paths and image to adjust for your cluster
#############################################

# Shared project root on host (keep /home usage narrow)
PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/hpc-pipeline}"

# Fixed experiment root (host-side)
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/pipeline_run}"

# Appainter image (build/push this separately)
APPAINTER_IMAGE="${APPAINTER_IMAGE:-}"
if [[ -z "${APPAINTER_IMAGE}" ]]; then
  echo "APPAINTER_IMAGE is not set. Export it before calling sbatch." >&2
  exit 1
fi

# Profiling mode for run.sh: "none", "nsys", or "perf"
export PROFILER="${PROFILER:-none}"

export PROJECT_ROOT
export EXPERIMENT_ROOT
export EXP_CONFIG_PATH="${EXP_CONFIG_PATH:-${EXPERIMENT_ROOT}/exp_config.json}"
export DS_CONFIG_PATH="${DS_CONFIG_PATH:-${EXPERIMENT_ROOT}/ds_config.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "/home/${USER}/hpc-runs"

echo "Job ${SLURM_JOB_NAME} starting on nodes: ${SLURM_JOB_NODELIST}"
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Config: ${EXP_CONFIG_PATH}"
echo "DeepSpeed config: ${DS_CONFIG_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Image: ${APPAINTER_IMAGE}"

#############################################
# Launch containerized run.sh on all tasks
#############################################

# srun will start one task per node (because of --ntasks-per-node=1)
# Each task runs a separate Appainter container, calling /app/run.sh inside.
# All tasks see the same /workspace (mounted from PROJECT_ROOT).

srun appainter run \
    --mount "${PROJECT_ROOT}:/workspace" \
    "${APPAINTER_IMAGE}" \
    /app/run.sh

echo "Job ${SLURM_JOB_ID} finished."

echo "You can inspect accounting data with:"
echo "  sacct -j ${SLURM_JOB_ID} --format=JobID,JobName%30,State,Elapsed,MaxRSS"

#!/bin/bash
#SBATCH --job-name=llama_pipeline
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --output=/home/%u/scratch/group1/hpc-runs/%x-%j.out
#SBATCH --error=/home/%u/scratch/group1/hpc-runs/%x-%j.err
# Update the paths above if you change SCRATCH_ROOT.

# submit.sbatch
# Single pipeline-parallel launch with fixed config paths.
# Usage: sbatch submit.sbatch (or via slurm/launch_pipeline.sh)

set -euo pipefail

#############################################
# Paths and image to adjust for your cluster
#############################################

# Shared project root on host (keep /home usage narrow). Use scratch for runtime data.
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/${USER}/scratch/group1}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_ROOT}/pipeline_run}"
CODE_ROOT="${CODE_ROOT:-/home/${USER}/projects/distributed-inference}"

# Container-visible experiment root: use a writable path that already exists.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/tmp/workspace}"

# Apptainer image: prefer the shared PyTorch SIF provided on the cluster.
# Override via APPAINTER_IMAGE if your cluster stores it elsewhere.
DEFAULT_IMAGE="${SCRATCH_ROOT}/appainter/appainter.sif"
if [[ -z "${APPAINTER_IMAGE:-}" ]]; then
  if [[ -f "${DEFAULT_IMAGE}" ]]; then
    APPAINTER_IMAGE="${DEFAULT_IMAGE}"
  else
    APPAINTER_IMAGE="/project/containers/pytorch-2.3.1-cuda11.8.sif"
  fi
fi
if [[ ! -f "${APPAINTER_IMAGE}" ]]; then
  echo "Apptainer image not found at ${APPAINTER_IMAGE}. Build env/apptainer.def into ${DEFAULT_IMAGE} or override APPAINTER_IMAGE." >&2
  exit 1
fi
if [[ "${APPAINTER_IMAGE}" == "/project/containers/pytorch-2.3.1-cuda11.8.sif" ]]; then
  echo "Using shared PyTorch image; slurm/run.sh will self-install DeepSpeed/Transformers into ${EXPERIMENT_ROOT}/.venv."
fi

if [[ ! -d "${CODE_ROOT}" ]]; then
  echo "CODE_ROOT ${CODE_ROOT} does not exist. Point CODE_ROOT to the repo with slurm/run.sh." >&2
  exit 1
fi

# Profiling mode for run.sh: "none", "nsys", or "perf"
# Default to none because the shared PyTorch SIF does not include nsys/perf.
export PROFILER="${PROFILER:-none}"

export PROJECT_ROOT
export EXPERIMENT_ROOT
export SCRATCH_ROOT
export EXP_CONFIG_PATH="${EXP_CONFIG_PATH:-${EXPERIMENT_ROOT}/exp_config.json}"
export DS_CONFIG_PATH="${DS_CONFIG_PATH:-${EXPERIMENT_ROOT}/ds_config.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_ROOT}/hpc-runs"

echo "Job ${SLURM_JOB_NAME} starting on nodes: ${SLURM_JOB_NODELIST}"
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Config: ${EXP_CONFIG_PATH}"
echo "DeepSpeed config: ${DS_CONFIG_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Image: ${APPAINTER_IMAGE}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Code root: ${CODE_ROOT}"
echo "Container workspace: ${EXPERIMENT_ROOT}"

# Pre-compute master address on the host (scontrol may not exist inside the image)
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT
echo "Master addr: ${MASTER_ADDR}"

#############################################
# Launch containerized run.sh on all tasks
#############################################

# srun will start one task per node (because of --ntasks-per-node=1)
# Each task runs a separate Apptainer container, calling /app/slurm/run.sh inside.
# All tasks see the same /workspace (mounted from PROJECT_ROOT).

srun apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${EXPERIMENT_ROOT}" \
    --bind "${CODE_ROOT}:/app" \
    --pwd /app \
    "${APPAINTER_IMAGE}" \
    bash /app/slurm/run.sh

echo "Job ${SLURM_JOB_ID} finished."

echo "You can inspect accounting data with:"
echo "  sacct -j ${SLURM_JOB_ID} --format=JobID,JobName%30,State,Elapsed,MaxRSS"

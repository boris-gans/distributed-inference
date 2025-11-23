#!/usr/bin/env bash
# run.sh
#
# Single pipeline-parallel Slurm task launcher (runs inside the container).
# It wires Slurm env → torch.distributed, and invokes your fixed DeepSpeed entrypoint.
# Assumes /workspace is a shared path mounted by submit.sbatch.

set -euo pipefail

############################
# Basic configuration
############################

# Shared experiment root (mounted from host). One fixed pipeline run.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/workspace/pipeline_run}"

# Fixed config paths for the single strategy
EXP_CONFIG_PATH="${EXP_CONFIG_PATH:-${EXPERIMENT_ROOT}/exp_config.json}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-${EXPERIMENT_ROOT}/ds_config.json}"

# Output directory per job (shared across ranks)
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs}"

# Profiling mode: "none", "nsys", or "perf"
PROFILER="${PROFILER:-none}"

mkdir -p "${OUTPUT_DIR}"

############################
# Slurm → torch.distributed/env wiring
############################

# Number of ranks = total Slurm tasks
WORLD_SIZE="${WORLD_SIZE:-${SLURM_NTASKS:-1}}"
RANK="${RANK:-${SLURM_PROCID:-0}}"
LOCAL_RANK="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"

# Pick master address as the first node in the job
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT

# NCCL backend tuning
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0

# Reasonable default for CPU threads
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

############################
# Base command: PyTorch + DeepSpeed Inference entrypoint
############################

BASE_CMD=(
    python /app/run_distributed_inference.py
    --exp_config "${EXP_CONFIG_PATH}"
    --ds_config "${DS_CONFIG_PATH}"
    --output_dir "${OUTPUT_DIR}"
)

# Each rank’s log file (inside shared storage)
RANK_LOG="${OUTPUT_DIR}/rank_${RANK}.log"

############################
# Wrap with profiler if requested
############################

case "${PROFILER}" in
  nsys)
    echo "[rank ${RANK}] Running under Nsight Systems..." | tee -a "${RANK_LOG}"
    # Adjust Nsight options as needed
    nsys profile \
      --force-overwrite=true \
      --output="${OUTPUT_DIR}/nsys_rank_${RANK}" \
      --capture-range=nvtx \
      --capture-range-end=stop \
      "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    ;;

  perf)
    echo "[rank ${RANK}] Running under perf..." | tee -a "${RANK_LOG}"
    perf stat -d -d -d \
      --output="${OUTPUT_DIR}/perf_rank_${RANK}.txt" \
      "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    ;;

  none|*)
    echo "[rank ${RANK}] Running without profiler..." | tee -a "${RANK_LOG}"
    "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    ;;
esac

echo "[rank ${RANK}] Done." | tee -a "${RANK_LOG}"

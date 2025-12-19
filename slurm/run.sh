#!/bin/bash

# Launches a single pipeline-parallel task inside the container.
# Sets up Slurm environment variables for torch.distributed and runs the DeepSpeed entrypoint.
# Assumes /tmp/workspace is a shared mount provided by submit.sbatch.

set -euo pipefail

############################
# Basic configuration
############################

# Shared experiment root mounted from the host.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/tmp/workspace}"

# Fixed config paths used by this run.
EXP_CONFIG_PATH="${EXP_CONFIG_PATH:-${EXPERIMENT_ROOT}/exp_config.json}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-${EXPERIMENT_ROOT}/ds_config.json}"

# Output directory shared by all ranks.
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs}"

# Optional profiling mode: none, nsys, or perf.
PROFILER="${PROFILER:-none}"

mkdir -p "${OUTPUT_DIR}"

############################
# Slurm → torch.distributed setup
############################

# Map Slurm task information into distributed training variables.
WORLD_SIZE="${WORLD_SIZE:-${SLURM_NTASKS:-1}}"
RANK="${RANK:-${SLURM_PROCID:-0}}"
LOCAL_RANK="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"

# Default master address/port if not provided.
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT

# NCCL behavior tuned for typical multi-GPU runs.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0

# Basic thread count for CPU-heavy sections.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

############################
# Python environment and dependencies
############################

# Ensure a working CA bundle inside the container in case host defaults are not visible.
CERT_FILE="$(python - <<'PY'
import certifi
try:
    print(certifi.where())
except Exception:
    print("")
PY
)"
if [[ -z "${CERT_FILE}" || ! -f "${CERT_FILE}" ]]; then
  if [[ -f "/etc/ssl/certs/ca-certificates.crt" ]]; then
    CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  else
    echo "[rank ${RANK}] WARNING: No CA bundle found; pip may fail." >&2
  fi
fi
if [[ -n "${CERT_FILE:-}" && -f "${CERT_FILE}" ]]; then
  export SSL_CERT_FILE="${CERT_FILE}"
  export REQUESTS_CA_BUNDLE="${CERT_FILE}"
  export PIP_CERT="${CERT_FILE}"
  echo "[rank ${RANK}] Using CA bundle at ${CERT_FILE}"
fi

PYTHON_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

PYTHON_SITE="${EXPERIMENT_ROOT}/.venv/lib/python${PYTHON_VERSION}/site-packages"
PYTHON_BIN="${EXPERIMENT_ROOT}/.venv/bin"

# Add site-packages and local tools to the environment.
export PYTHONPATH="${PYTHON_SITE}:/app:${PYTHONPATH:-}"
export PIP_CACHE_DIR="${EXPERIMENT_ROOT}/.pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="${PYTHON_BIN}:${PYTHON_SITE}/bin:${PATH}"

# Avoid large Triton caches hitting slow filesystems.
export TRITON_CACHE_DIR="${EXPERIMENT_ROOT}/.triton-cache"

# Remove any pip-installed torch or CUDA wheels so the container versions are always used.
if [[ -d "${PYTHON_SITE}/torch" || -d "${PYTHON_SITE}/torch-2."* ]]; then
  echo "[rank ${RANK}] Removing local torch/CUDA wheels to use container builds..."
  rm -rf "${PYTHON_SITE}/torch" \
         "${PYTHON_SITE}"/torch-*.dist-info \
         "${PYTHON_SITE}"/nvidia* \
         "${PYTHON_SITE}"/triton*
fi

# Helper to test whether key Python modules are available.
check_python_deps() {
  PYTHONPATH="${PYTHONPATH}" python - <<'PY'
import importlib.util, sys
missing = [m for m in ("deepspeed", "transformers", "tokenizers") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
}

# Install missing Python packages into the shared venv location.
if ! check_python_deps; then
  echo "[rank ${RANK}] Installing Python dependencies..."
  mkdir -p "${PYTHON_SITE}" "${PIP_CACHE_DIR}"
  (
    flock 200
    if ! check_python_deps; then
      DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 DS_SKIP_TORCH_CHECK=1 \
      python -m pip install --no-cache-dir --upgrade \
        --no-deps \
        --target "${PYTHON_SITE}" \
        "deepspeed==0.14.4" \
        "transformers==4.43.3" \
        "tokenizers==0.19.1" \
        "sentencepiece==0.2.0" \
        "accelerate==0.30.1" \
        "huggingface-hub==0.23.4" \
        "hjson==3.1.0" \
        "tqdm>=4.66" \
        "psutil>=5.9" \
        "pyyaml" \
        "packaging" \
        "ninja>=1.11" \
        "requests>=2.31" \
        "filelock>=3.12" \
        "regex>=2024.4" \
        "safetensors>=0.4.1" \
        "fsspec>=2023.5.0" \
        "pydantic==2.12.5" \
        "typing-inspection>=0.4.2" \
        "py-cpuinfo>=9.0.0" \
        "typing_extensions>=4.9" \
        "certifi>=2024.2.0" \
        "charset-normalizer>=3.3.0" \
        "idna>=3.6" \
        "urllib3>=2.1.0" \
        "numpy>=1.23"
    fi
  ) 200>"${EXPERIMENT_ROOT}/.pip-install.lock"
fi

# Ensure ninja is present for torch extension builds.
if ! command -v ninja >/dev/null 2>&1; then
  if [[ -x "${PYTHON_SITE}/bin/ninja" ]]; then
    export PATH="${PYTHON_SITE}/bin:${PATH}"
  elif [[ -x "${PYTHON_BIN}/ninja" ]]; then
    export PATH="${PYTHON_BIN}:${PATH}"
  else
    echo "[rank ${RANK}] Installing ninja..."
    python -m pip install --no-cache-dir --upgrade --target "${PYTHON_SITE}" "ninja>=1.11"
    export PATH="${PYTHON_SITE}/bin:${PATH}"
  fi
fi

# Final check before continuing.
if ! check_python_deps; then
  echo "[rank ${RANK}] Missing required Python modules; aborting." >&2
  exit 1
fi

#########################################
# Base command for distributed inference
#########################################

BASE_CMD=(
    python /app/run_distributed_inference.py
    --exp_config "${EXP_CONFIG_PATH}"
    --ds_config "${DS_CONFIG_PATH}"
    --output_dir "${OUTPUT_DIR}"
)

# Per-rank log file.
RANK_LOG="${OUTPUT_DIR}/rank_${RANK}.log"

####################################################################################
# Profiler wrapper, allowing use of nsys or perf if installed and defined in submit.sbatch
####################################################################################

case "${PROFILER}" in
  nsys)
    if command -v nsys >/dev/null 2>&1; then
      echo "[rank ${RANK}] Running under Nsight Systems..." | tee -a "${RANK_LOG}"
      nsys profile \
        --force-overwrite=true \
        --output="${OUTPUT_DIR}/nsys_rank_${RANK}" \
        --capture-range=nvtx \
        --capture-range-end=stop \
        "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    else
      echo "[rank ${RANK}] Nsight Systems not available; running normally." | tee -a "${RANK_LOG}"
      "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    fi
    ;;

  perf)
    if command -v perf >/dev/null 2>&1; then
      echo "[rank ${RANK}] Running under perf..." | tee -a "${RANK_LOG}"
      perf stat -d -d -d \
        --output="${OUTPUT_DIR}/perf_rank_${RANK}.txt" \
        "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    else
      echo "[rank ${RANK}] perf not available; running normally." | tee -a "${RANK_LOG}"
      "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    fi
    ;;

  none|*)
    echo "[rank ${RANK}] Running without profiling..." | tee -a "${RANK_LOG}"
    "${BASE_CMD[@]}" 2>&1 | tee -a "${RANK_LOG}"
    ;;
esac

echo "[rank ${RANK}] Done." | tee -a "${RANK_LOG}"
## Node and Hardware Configuration

* **Partition:** `gpu-node`
* **Nodes:** 2
* **Tasks per Node:** 1
* **GPUs per Node:** 1
* **CPUs per Task:** 4
* **GPU Model:** Tesla T4 (16 GB)
* **Host NVIDIA Driver:** 550.144.06
* **Host CUDA Version (driver API):** 12.4

## Module Stack (Host)

* CCconfig
* gentoo/2023
* gcccore/.12.3
* gcc/12.3
* hwloc/2.9.1
* ucx/1.14.1
* libfabric/1.18.0
* pmix/4.2.4
* ucc/1.2.0
* openmpi/4.1.5
* flexiblas/3.3.1
* imkl/2023.2.0
* StdEnv/2023
* **apptainer/1.3.5**
* **cuda/12.2**

## Container Runtime

* **Image:** `pytorch-2.3.1-cuda11.8.sif`
* **PyTorch (inside container):** 2.3.1
* **Container CUDA Runtime:** 11.8
* **nvidia-smi inside container:** not available
* **Bindings:**

  * `${PROJECT_ROOT}` → `/tmp/workspace`
  * `${CODE_ROOT}` → `/app`
  * `${SCRATCH_ROOT}/models/openllama-3b` → `/workspace/models/openllama-3b`

## Distributed Runtime

* **Backend:** NCCL
* **Master Address:** first host in `$SLURM_JOB_NODELIST`
* **Master Port:** 29500
* **Environment Variables:**

  * `WORLD_SIZE`, `RANK`, `LOCAL_RANK`
  * `NCCL_DEBUG=INFO`
  * `NCCL_ASYNC_ERROR_HANDLING=1`
  * `NCCL_IB_DISABLE=0`
  * `NCCL_NET_GDR_LEVEL=2`
  * `NCCL_P2P_DISABLE=0`
  * `OMP_NUM_THREADS=4`

## Python Environment

* Local venv created under:

  ```
  /tmp/workspace/.venv/
  ```
* Runtime-installed packages include:
  DeepSpeed 0.14.4, Transformers 4.43.3, Tokenizers 0.19.1, SentencePiece 0.2.0, Accelerate 0.30.1, HuggingFace Hub, TQDM, NumPy, Pydantic, FSSpec, Safetensors, Regex, Ninja, and related utilities.

## Filesystems

* **Scratch Root:** `/home/$USER/scratch/group1`
* **Experiment Root (container-visible):** `/tmp/workspace`
* **SLURM Logs:** `/home/$USER/scratch/group1/hpc-runs`
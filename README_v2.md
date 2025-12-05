# Distributed Inference (v2):

## Overview
Many large language models are too large to fit on a single GPU, so even inference requires multi-GPU, multi-node execution.
This project implements and benchmarks a POC, using a small 3B model and a distributed inference pipeline using **PyTorch** with **DeepSpeed Inference** (which uses **NCCL** for GPU communication). The model we choose was OpenLLaMA v2 3B (https://huggingface.co/openlm-research/open_llama_3b_v2), due to it's small size and permissive license

## Objectives
- Run OpenLLaMA v2 3B inference on 2 GPU nodes under **Slurm**.
- Measure "correctness" of the model through comparing a set of 20 prompts with a baseline 
- Measure strong and weak scaling (throughput / latency / efficiency) from 1 → 2 nodes.
- Conduct a batch size sweep to evaluate the parameter's affect (memory and throughput) 
- Quantize the model to explore the balance between performance and accuracy
- Profile compute vs. communication time using **INSERT HERE**.   
- Produce a short paper describing results and scaling limits.

## Tech Stack
- **PyTorch** — model runtime  
- **DeepSpeed Inference** — Pipeline parallelism  
- **NCCL** — GPU–GPU communication backend  
- **Slurm** — cluster scheduling  
- **Apptainer** — containerization  
- **Nsight / perf / sacct** — profiling and performance analysis (**not up to date**)

## Model Hyperparameters

**might be worth it to elaborate, but only the params that we change during our experiments**

## Cluster Configuration
Our model will implement Pipeline Parallelism (PP) across nodes.

By doing so, we ensure the model fits within a node and we have a clean scaling axis. Our scaling experiments will go as follows:

  | Nodes | GPUs/node | TP | PP | Total GPUs |
  | ----- | --------- | -- | -- | ---------- |
  | 1     | 1         | 2  | 1  | 1          |
  | 2     | 1         | 2  | 2  | 2          |

*Note: If we had access to more GPU nodes, we would perform at least four iterations of the above*

### Cluster Topology:
**OUTDATED; NO TP**


                    ┌─────────────────────────────────────────────────────────────┐
                    │                DeepSeek-70B Inference Cluster               │
                    └─────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                  ┌──────────────────────────────────┐
                                  │        Input Prompt Batch        │
                                  └──────────────────────────────────┘
                                                  │
                              ┌───────────────────┼──────────────────────┐
                              │                   │                      │
                              ▼                   ▼                      ▼

                    ┌────────────────────-─┐   ┌────────────────────-─┐   ┌─────────────────────┐
                    │  Node 0 (Stage 0)    │   │  Node 1 (Stage 1)    │   │  Node 2 (Stage 2)   │
                    │  Layers 0–N/3        │   │  Layers N/3–2N/3     │   │  Layers 2N/3–N      │
                    │  Tensor Parallel = 2 │   │  Tensor Parallel = 2 │   │  Tensor Parallel = 2│
                    └─────────────────────-┘   └────────────────────-─┘   └─────────────────────┘
                              │                   │                      │
                              ▼                   ▼                      ▼

                        ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
                        │ GPU0 ──┐          │      │ GPU2 ──┐          │      │ GPU4 ──┐          │
                        │ GPU1 ──┼─ NVLink ─┼─ TP  │ GPU3 ──┼─ NVLink ─┼─ TP  │ GPU5 ──┼─ NVLink ─┼─ TP
                        │        └─ intra ──┘      │        └─ intra ──┘      │        └─ intra ──┘
                        │       node comm          │       node comm          │       node comm
                        └───────────────────┘      └───────────────────┘      └───────────────────┘
                              │                   │                      │
                              │<────────── Pipeline (activations, KV-cache) ───────────│
                              │                   │                      │
                              ▼                   ▼                      ▼

                      ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
                      │  Partial Out   │   │  Partial Out   │   │   Final Output │
                      └────────────────┘   └────────────────┘   └────────────────┘


## Quick Start
### Local dev (ignore for now):
```bash
git clone https://github.com/boris-gans/distributed-inference.git
cd distributed-inference
source .venv/bin/activate
pip install -r requirements.txt

# Default to --override and all variants with no limit (40 prompts - careful this is a lot of tokens)
python -m src.inference

# Override flag to skip parquet loading (if available) and re-construct all df's
python -m src.inference --override 

# No-override flag to load df's from parquet (if available) and skip df construction
python -m src.inference --no-override 

# Variant flag to load only 2k/4k prompts, limit flag to specify amount of prompts to load from each variant
python -m src.inference --override --variant=2k --limit=1
```

### On the cluster:
``` bash
# Install and authenticate HF cli first: https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
hf download openlm-research/open_llama_3b --include="*" --local-dir models/

rsync -av \
  slurm \
  run_distributed_inference.py \
  <YOUR-USER>@hpcie.labs.faculty.ie.edu:/home/<YOUR-USER>/projects/def-sponsor00/<YOUR-USER>/distributed-inference

# Note: this will take a while and is about 6GB. Do this once per group, not user
rsync -av \
  models/openllama-3b \
  user49@hpcie.labs.faculty.ie.edu:/home/user49/scratch/group1/models/

# Note: I have the hostname configured in ~/.ssh/config to resolve hpcie for user49 at the real host
ssh hpcie 

export APPAINTER_IMAGE=/home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif
export PIPELINE_ROOT=/home/user49/scratch/group1/pipeline_run
export CODE_ROOT=/home/user49/projects/def-sponsor00/user49/distributed-inference
export APPTAINERENV_LD_LIBRARY_PATH=/usr/lib64:/usr/lib


sbatch ${CODE_ROOT}/slurm/submit.sbatch
```

Now that your job has been submitted, you can run the following to inspect / monitor it.

```bash
squeue -j <JOB-ID>

scontrol show job <JOB-ID>
```

To open the logs:
```bash
# Follow logs
tail -f /home/<YOUR-USER>/scratch/group1/hpc-runs/llama_pipeline-<JOB-ID>.out | ts
# Or for error messages
tail -f /home/<YOUR-USER>/scratch/group1/hpc-runs/llama_pipeline-<JOB-ID>.err | ts


# Or just see all (doesnt update)
cat /home/<YOUR-USER>/scratch/group1/hpc-runs/llama_pipeline-<JOB-ID>.out
# Or for error messages
cat /home/<YOUR-USER>/scratch/group1/hpc-runs/llama_pipeline-<JOB-ID>.err
```

Once the job has finished, pull the results back to your local workstation:
```bash
# First ensure ALL logs/info are outside the cluster (per-rank logs aren't binded); do this on the cluster
cp -r /tmp/workspace/outputs ${SCRATCH_ROOT}/pipeline_run/outputs

# Then locally:
rsync -avz --progress \
    <YOUR-USER>@hpcie.labs.faculty.ie.edu:${SCRATCH_ROOT}/pipeline_run/outputs \
    ./outputs
```


## Execution Flow
1) **Local step** — dry-run `python -m src.inference --local_debug` to verify configs/prompts and capture reference outputs.  
2) **Cluster step** — Follow the above steps to upload the code and the model to the cluster (if needed) and run the batch job.
3) **Result sync** — pull `${EXPERIMENT_ROOT}/outputs` (rank logs, completions, profiler traces if enabled) back to your local workstation.  
4) **Post-process** — run the analysis notebook/script to compute metrics across the four experiments; include the local reference outputs when comparing correctness.

## Experiments to Run
- **Strong scaling** — fix total tokens/prompts, vary nodes/GPUs to measure speedup.  
- **Weak scaling** — grow workload with resources so per-GPU load stays constant.  
- **Sensitivity sweep** — sweep one parameter (batch size, preconditioner setting, or partitioning choice) to expose stability/efficiency trends.  
- **Optimization** — apply one change that measurably improves time or throughput (e.g., caching, fused kernels, better partitioning) and rerun the baseline config.

## Metrics Captured (each experiment)
- Wall-clock time (job elapsed), throughput (tokens/sec or prompts/sec), and parallel efficiency.  
- Resource use (GPU/CPU util + memory), I/O time, communication time, and preprocessing cost.  
- Keep profiler outputs (`nsys`/`perf` if enabled), `sacct` accounting, and the per-rank logs for traceability.

## Outputs
- Structured metrics bundle per run (CSV/JSON) with the values above.  
- Plots/tables comparing strong vs. weak scaling, the sensitivity curve, and the optimization delta.  
- Notes on correctness deltas between local and cluster outputs (manual spot checks only).


## Cluster File Structure
**outdated; TO DO**

Professor guidance: keep code tiny in `/home`, persist important assets in `/project`, and put large/temporary runtime data in `/scratch`. On this cluster you have `/home/user49/projects` and `/scratch/user49`. The job should mount `/scratch/user49/pipeline_run` (shared, fast) into `/workspace` for all nodes; that is where configs, outputs, and the Hugging Face model cache live. Only `/slurm`, `env/appainter.sif`, and `run_distributed_inference.py` are staged; the model downloads at runtime into shared storage so all nodes reuse it.

```
Host layout (what exists before sbatch)
/home/user49/projects/distributed-inference/   # code only (small)
├── slurm/                                     # submit.sbatch, run.sh, etc.
├── run_distributed_inference.py               # source copy
└── env/appainter.sif                          # if small; otherwise place in /project/user49/appainter/appainter.sif

/project/user49/appainter/                     # persistent, if available (preferred spot for the SIF)
└── appainter.sif                              # built from env/appainter.def

/scratch/user49/pipeline_run/                  # EXPERIMENT_ROOT (shared across nodes/ranks; mounted to /workspace)
├── exp_config.json                            # you place here before sbatch
├── ds_config.json                             # you place here before sbatch
├── slurm/                                     # optional copy of run.sh if you launch via /workspace/slurm/run.sh
├── outputs/                                   # created at runtime (shared)
│   ├── rank_0.log
│   ├── rank_1.log
│   ├── completions_rank_0.jsonl
│   ├── completions_rank_1.jsonl
│   ├── sacct_<jobid>.txt                      # if sacct available (written by rank 1)
│   ├── nsys_rank_<r>.*                        # if PROFILER=nsys
│   └── perf_rank_<r>.txt                      # if PROFILER=perf
└── hf_cache/                                  # Hugging Face cache; model+tokenizer download happens here once

Inside any node (container view during the job)
/app/
├── run_distributed_inference.py               # baked into the SIF
└── (optionally) run.sh                        # if you baked it; otherwise use the bind-mounted copy
/workspace/                                    # bind-mounted from /scratch/user49/pipeline_run
├── exp_config.json
├── ds_config.json
├── outputs/                                   # shared logs/results/profiles
└── hf_cache/                                  # shared model/tokenizer cache (downloaded once, reused by all nodes)

Per-node local (outside the mount, if any)
/tmp/, /var/tmp/                               # not used by default; no model copies here unless you change HF cache
```


## Resources
**Model:** https://huggingface.co/openlm-research/open_llama_3b_v2
**Docs:**
-   TO DO (apptainer)
-   TO DO (pytorch)
-   TO DO (deepspeed)
-   etc

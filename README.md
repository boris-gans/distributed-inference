# Distributed Inference of Llama 3.3 70B on Multi-GPU Clusters

## Overview
Large language models such as Llama 3.3 70B are too large to fit on a single GPU, so even inference requires multi-GPU, multi-node execution.  
This project implements and benchmarks a distributed inference pipeline using **PyTorch** with **DeepSpeed Inference** (which uses **NCCL** for GPU communication).  
CPU nodes are optionally used for orchestration and preprocessing, allowing us to explore hybrid CPU–GPU performance.

## Objectives
- Run Llama 3.3 70B inference on 2–3 GPU nodes under **Slurm**.
- Measure "correctness" of the model through comparing a set of 20 prompts with a baseline  
- Measure scaling (throughput / latency / efficiency) from 1 → N nodes.
- Profile compute vs. communication time using **Nsight Systems**, **perf**, and **sacct**.  
- Package everything in an **Apptainer** container for full reproducibility.  
- Produce a short paper and EuroHPC proposal describing results and scaling limits.

## Tech Stack
- **PyTorch** — model runtime  
- **DeepSpeed Inference** — tensor + pipeline parallelism  
- **NCCL** — GPU–GPU communication backend  
- **Slurm** — cluster scheduling  
- **Apptainer** — containerization  
- **Nsight / perf / sacct** — profiling and performance analysis

## Model Hyperparameters

* **L, H, D**: `L=80`, `H=64`, `D=128` (hidden size `8192`).
* **KV per token (BF16/FP16)**: `~2.5 MiB`.
  * KV@2k: **5.0 GiB** total; KV@4k: **10.0 GiB** total (batch=1).
  * **Per-GPU KV** = total_KV / (**TP×PP**).
* **Weights** (BF16): 70e9 x 2B = **~130.4 GiB** total = **~140 GB** decimal.
  * **Per-GPU weights** = total_weights / (**TP×PP**).
* **Other activations** (non-KV) budget: **~2 GiB/GPU** (tune with measurements).
* **Overhead** budget: **~6 GiB/GPU** (CUDA/NCCL/allocator/workspaces).
  * **Batch-size > 1** Activation memory (including KV) scales lineary with batch-size


This table shows the **per GPU memory**, assuming 8 GPUs (TPxPP always equals 8, regardless of cluster config).


| Component         | Formula                     | Value (GiB) |
| ----------------- | --------------------------- | ----------- |
| Weights           | 130.4 / (TP×PP) = 130.4 / 8 | **16.30**   |
| KV @ **2k**       | 5.0 / 8                     | **0.625**   |
| KV @ **4k**       | 10.0 / 8                    | **1.25**    |
| Other activations | fixed                       | **2.0**     |
| Overhead          | fixed                       | **6.0**     |
| **Total @ 2k**    | sum                         | **24.93**   |
| **Total @ 4k**    | sum                         | **25.55**   |

<hr>

## Cluster Configuration
Our model will implement two parallelism techniques:

- Tensor Parallelism (TP) across all GPUs within a node
- Pipeline Parallelism (PP) across nodes

By doing so, we ensure the model fits within a node (through TP) and we have a clean scaling axis (through PP). Our scaling experiments will go as follows:

  | Nodes | GPUs/node | TP | PP | Total GPUs |
  | ----- | --------- | -- | -- | ---------- |
  | 1     | 2         | 2  | 1  | 2          |
  | 2     | 2         | 2  | 2  | 4          |
  | 3     | 2         | 2  | 3  | 8          |


### Cluster Topology:

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
**For now (dev):**
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
python -m src.inference --override --variant=2k --limit=10
```

**Outdated:**
```bash
git clone https://github.com/boris-gans/distributed-inference.git
cd deepseek-hpc/env
apptainer build deepseek.sif project.def
sbatch slurm/submit_inference.sbatch

# Local debug workflow (CPU only)
python -m src.inference --input data/sample_inputs.txt --local_debug --output results/local_debug.jsonl

# Orchestrate local workers
python -m src.orchestrator --input data/sample_inputs.txt --local_debug --num_workers 2 --dispatch_size 4
```

## Development Skeleton
- `src/inference.py` exposes the CLI entry point. Pass `--local_debug` to bypass DeepSpeed/NCCL setup and run a lightweight PyTorch-only flow that verifies I/O, batching, and logging.
- `src/orchestrator.py` simulates distributed request handling with threads so you can refine orchestration logic before running under Slurm.
- `src/utils.py` provides shared logging and performance tracking helpers (throughput, latency, CPU timings).
- `results/` is where we store the system performance, feeding directly into the "performance and scaling analysis" section of the paper. It answers speed, effecienty and scalability
- `eval/`is where we perform model quality evaluation. It answers model correctness, and if precision or sharding affect model quality


## Resources
- **Hosting Updated:** https://fireworks.ai/
  - https://app.fireworks.ai/models/fireworks/gpt-oss-20b
  - https://app.fireworks.ai/models/fireworks/gpt-oss-120b

- **Model Choices:**
  - https://huggingface.co/openai/gpt-oss-20b
  - https://huggingface.co/openai/gpt-oss-120b

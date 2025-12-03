# Distributed Inference: Updated Plan (v2)

## What Changed
- Accuracy validation now runs manually: execute locally first, then on the cluster; no Fireworks AI or automated evaluator in the loop.
- Runtime uses the shared PyTorch Apptainer image at `/project/containers/pytorch-2.3.1-cuda11.8.sif` (see `env/modules.txt` for the module stack).
- Experiments are organized as four explicit runs with a fixed metric set captured after results are pulled back.

## Execution Flow
1) **Local step** — dry-run `python -m src.inference --local_debug` to verify configs/prompts and capture reference outputs.  
2) **Cluster step** — submit `slurm/submit.sbatch` (or `launch_pipeline.sh`) pointing `APPAINTER_IMAGE` to the shared PyTorch SIF and `PROJECT_ROOT`/`SCRATCH_ROOT` to the shared scratch workspace.  
3) **Result sync** — pull `${EXPERIMENT_ROOT}/outputs` (rank logs, completions, profiler traces if enabled) back to your workstation.  
4) **Post-process** — run the analysis notebook/script to compute metrics across the four experiments; include the local reference outputs when comparing correctness.

## Experiments to Run
- **Strong scaling** — fix total tokens/prompts, vary nodes/GPUs to measure speedup.  
- **Weak scaling** — grow workload with resources so per-GPU load stays constant.  
- **Sensitivity sweep** — sweep one parameter (batch size, preconditioner setting, or partitioning choice) to expose stability/efficiency trends.  
- **Optimization** — apply one change that measurably improves time or throughput (e.g., caching, fused kernels, better partitioning) and rerun the baseline config.

## Metrics to Capture (each experiment)
- Wall-clock time (job elapsed), throughput (tokens/sec or prompts/sec), and parallel efficiency.  
- Resource use (GPU/CPU util + memory), I/O time, communication time, and preprocessing cost.  
- Keep profiler outputs (`nsys`/`perf` if enabled), `sacct` accounting, and the per-rank logs for traceability.

## Outputs
- Structured metrics bundle per run (CSV/JSON) with the values above.  
- Plots/tables comparing strong vs. weak scaling, the sensitivity curve, and the optimization delta.  
- Notes on correctness deltas between local and cluster outputs (manual spot checks only).

Cluster-only run instructions (uses the shared PyTorch Apptainer image; no local build).

1) Sync the minimal code/assets to the cluster:
```bash
rsync -av \
  slurm \
  src/run_distributed_inference.py \
  <USER>@hpcie.labs.faculty.ie.edu:/home/<USER>/projects/def-sponsor00/<USER>/distributed-inference

# One-time model sync (shared)
rsync -av \
  models/openllama-3b \
  <USER>@hpcie.labs.faculty.ie.edu:/home/<USER>/scratch/group1/models/
```

2) Prepare scratch workspace (seen as `/tmp/workspace` inside the container):
```bash
ssh hpcie

SCRATCH_ROOT=/home/<USER>/scratch/group1
PIPELINE_ROOT=${SCRATCH_ROOT}/pipeline_run
mkdir -p ${PIPELINE_ROOT}/outputs ${SCRATCH_ROOT}/hpc-runs
cp /home/<USER>/projects/def-sponsor00/<USER>/distributed-inference/slurm/exp_config.json ${PIPELINE_ROOT}/exp_config.json
cp /home/<USER>/projects/def-sponsor00/<USER>/distributed-inference/slurm/ds_config.json  ${PIPELINE_ROOT}/ds_config.json
cp /home/<USER>/projects/def-sponsor00/<USER>/distributed-inference/slurm/prompts.jsonl   ${PIPELINE_ROOT}/prompts.jsonl
```

3) Submit the job:
```bash
export APPAINTER_IMAGE=/home/<USER>/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif
export PIPELINE_ROOT=/home/<USER>/scratch/group1/pipeline_run
export CODE_ROOT=/home/<USER>/projects/def-sponsor00/<USER>/distributed-inference
export APPTAINERENV_LD_LIBRARY_PATH=/usr/lib64:/usr/lib

sbatch ${CODE_ROOT}/slurm/submit.sbatch
```
`submit.sbatch` binds `${PIPELINE_ROOT} -> /tmp/workspace` (configs/prompts/outputs) and `${CODE_ROOT} -> /app`. Inside the container, `run.sh` installs missing Python deps to `/tmp/workspace/.venv` and runs `run_distributed_inference.py`.

4) Monitor and collect results:
- Slurm stdout/err: `${SCRATCH_ROOT}/hpc-runs/llama_pipeline-<JOBID>.{out,err}`
- Per-rank logs/results: `${PIPELINE_ROOT}/outputs` (rank_*.log, completions_rank_*.jsonl, optional sacct/perf/nsys files)

Notes:
- Partition/CPU: `submit.sbatch` requests `--nodes=2`, `--ntasks-per-node=1`, `--gres=gpu:1`, `--cpus-per-task=4`, `--partition=gpu-node`.
- If the container cannot create `/workspace`, ensure you are using the updated scripts (they bind to `/tmp/workspace`).

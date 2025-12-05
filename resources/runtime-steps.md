## Run with the shared PyTorch image (no build)

1) Sync the tiny code folder to `/home` (or your project repo path) on the cluster:
```bash
rsync -av \
  slurm \
  run_distributed_inference.py \
  user49@hpcie.labs.faculty.ie.edu:/home/user49/projects/def-sponsor00/user49/distributed-inference

rsync -av \
  slurm \
  run_distributed_inference.py \
  user49@hpcie.labs.faculty.ie.edu:/home/user49

rsync -av \
  models/openllama-3b \
  user49@hpcie.labs.faculty.ie.edu:/home/user49/scratch/group1/models/
```

2) Prepare the shared scratch workspace that all ranks will see as `/tmp/workspace` inside the container:
```bash
ssh hpcie

SCRATCH_ROOT=/home/user49/scratch/group1
PIPELINE_ROOT=${SCRATCH_ROOT}/pipeline_run
™™£™
mkdir -p ${PIPELINE_ROOT}/outputs ${SCRATCH_ROOT}/hpc-runs
# Place prompts + configs under scratch so every node reads the same files
cp /home/user49/projects/def-sponsor00/user49/distributed-inference/slurm/exp_config.json ${PIPELINE_ROOT}/exp_config.json
cp /home/user49/projects/def-sponsor00/user49/distributed-inference/slurm/ds_config.json  ${PIPELINE_ROOT}/ds_config.json
cp /home/user49/projects/def-sponsor00/user49/distributed-inference/slurm/prompts.jsonl   ${PIPELINE_ROOT}/prompts.jsonl
```
*Prompts*: the job reads whatever path is in `exp_config.json["inputs"]["prompt_path"]`. The defaults expect `/tmp/workspace/prompts.jsonl`, which is the same file you copied into `${PIPELINE_ROOT}` on scratch.

3) Point to the shared PyTorch image and submit:
```bash
export APPAINTER_IMAGE=/home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif
export PIPELINE_ROOT=/home/user49/scratch/group1/pipeline_run
export CODE_ROOT=/home/user49/projects/def-sponsor00/user49/distributed-inference
export APPTAINERENV_LD_LIBRARY_PATH=/usr/lib64:/usr/lib


sbatch ${CODE_ROOT}/slurm/submit.sbatch
```
The sbatch script will bind `${PIPELINE_ROOT} -> /tmp/workspace` (for prompts/configs/outputs) and `${CODE_ROOT} -> /app` (for `run.sh` + `run_distributed_inference.py`).
Inside the container `EXPERIMENT_ROOT` is forced to `/tmp/workspace`, so configs/outputs paths should use `/tmp/workspace/...` (as in `exp_config.json`).
The master address is computed on the host before entering the container (no need for `scontrol` inside the image).

4) After the job, check logs/outputs under `${PIPELINE_ROOT}/outputs` and the Slurm stdout/err under `${SCRATCH_ROOT}/hpc-runs`.

Partition/CPU notes:
- GPU nodes live in partition `gpu-node` with 1 GPU and 4 CPUs each (per `sinfo` in notes).
- `submit.sbatch` requests `--nodes=2`, `--ntasks-per-node=1`, `--gres=gpu:1`, `--cpus-per-task=4`, `--partition=gpu-node` to match what is available; adjust only if your site changes these names.

If you see “Error changing the container working directory” the fix is already baked in via `--pwd /app` in `submit.sbatch`; just make sure `CODE_ROOT` points to the repo that contains `slurm/run.sh`.
If you see “cannot create directory ‘/workspace’” it means the container cannot create a root-level mountpoint. The scripts now use `/tmp/workspace`, which exists and is writable inside the image.
If you need runtime stats and can’t use Nsight Systems on the shared image, leave `PROFILER=none` (default). You still get wall-clock, per-prompt throughput in `rank_*.log` and Slurm accounting via `sacct`; you can also run `nvidia-smi dmon -s pucm -d 1` from the host on an interactive run for quick GPU utilization.
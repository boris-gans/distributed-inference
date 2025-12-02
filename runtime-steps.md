1) Build the Apptainer image on scratch so quotas in `projects`/`home` are not exhausted:
```bash
SCRATCH_ROOT=/home/user49/scratch/group1
mkdir -p ${SCRATCH_ROOT}/appainter
apptainer build ${SCRATCH_ROOT}/appainter/appainter.sif env/apptainer.def
```
If you build on a different machine, copy the SIF straight into scratch (not `projects`):
```bash
rsync -av ${SCRATCH_ROOT}/appainter/appainter.sif user49@login1:${SCRATCH_ROOT}/appainter/
```

2) Push only the small code artifacts to the cluster `projects` area (keep it tiny):
```bash
rsync -av \
  slurm \
  run_distributed_inference.py \
  env/apptainer.def \
  user49@login1:/home/user49/projects/def-sponsor00/user49/distributed-inference
```

3) On the cluster, prep the shared scratch workspace (mounted to `/workspace` in the job):
```bash
ssh user49@login1 <<'EOF'
  SCRATCH_ROOT=/home/user49/scratch/group1
  PIPELINE_ROOT=${SCRATCH_ROOT}/pipeline_run

  mkdir -p ${PIPELINE_ROOT}/outputs ${SCRATCH_ROOT}/hpc-runs
  # Copy or create your configs and prompts under scratch so they are shared:
  # cp ~/projects/def-sponsor00/user49/distributed-inference/exp_config.json ${PIPELINE_ROOT}/exp_config.json
  # cp ~/projects/def-sponsor00/user49/distributed-inference/ds_config.json ${PIPELINE_ROOT}/ds_config.json
  # cp ~/projects/def-sponsor00/user49/distributed-inference/prompts.jsonl  ${PIPELINE_ROOT}/prompts.jsonl
EOF
```

4) Submit from the login node using scratch for both the image and runtime data:
```bash
export APPAINTER_IMAGE=/home/user49/scratch/group1/appainter/appainter.sif
export PIPELINE_ROOT=/home/user49/scratch/group1/pipeline_run
sbatch /home/user49/projects/def-sponsor00/user49/distributed-inference/slurm/submit.sbatch
```


**dir notes**

*projects pwd*
/home/user49/projects/def-sponsor00/user49
/home/user49/projects/def-sponsor00/shared

*home pwd*
/home/user49/
or
/home/user49/distributed-inference

*scratch pwd*
/home/user49/scratch/group1

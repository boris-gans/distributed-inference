## Quick Start

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

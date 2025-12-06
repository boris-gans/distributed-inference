```bash
squeue -j <jobid>

scontrol show job <jobid>

sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,AllocCPUS,AllocGRES

watch -n 1 squeue -u $USER




tail -f /home/user49/scratch/group1/hpc-runs/llama_pipeline-4231.out | ts

scancel 


squeue -o "%.18i %.8u %.10P %.8T %.30R"

squeue --sort=u -o "%.8u %.18i %.8T %N"

```


export APPAINTER_IMAGE=/home/user49/projects/def-sponsor00/shared/images/pytorch-2.3.1-cuda11.8.sif
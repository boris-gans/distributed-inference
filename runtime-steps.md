1. Run app locally to populate and save the df's:
``` bash
python -m src.inference --override 
```

2. 
``` bash
rsync -av slurm run_distributed_inference.py user09@hpcie.labs.faculty.ie.edu:/home/user09/distributed-inference/
```

3. Build appainter image
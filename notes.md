# llama_pipeline-4205.err
srun: error: gpu-node2: task 1: Exited with exit code 1
srun: error: gpu-node1: task 0: Exited with exit code 1

# llama_pipeline-4205.out
Dec 03 10:22:16 [rank 0] Running without profiler...
Dec 03 10:22:16 [rank 1] Running without profiler...
Dec 03 10:22:16 2025-12-03 10:22:12,989 [INFO][rank=1] Initialized torch.distributed (rank=1, world_size=2)
Dec 03 10:22:16 2025-12-03 10:22:12,992 [INFO][rank=0] Initialized torch.distributed (rank=0, world_size=2)
Dec 03 10:22:16 2025-12-03 10:22:12,991 [INFO][rank=1] Using Hugging Face cache at /tmp/workspace/hf_cache
Dec 03 10:22:16 2025-12-03 10:22:12,995 [INFO][rank=0] Using Hugging Face cache at /tmp/workspace/hf_cache
Dec 03 10:22:16 [rank0]: Traceback (most recent call last):
Dec 03 10:22:16 [rank0]:   File "/app/run_distributed_inference.py", line 432, in <module>
Dec 03 10:22:16 [rank0]:     main()
Dec 03 10:22:16 [rank0]:   File "/app/run_distributed_inference.py", line 380, in main
Dec 03 10:22:16 [rank0]:     torch.cuda.set_device(device)
Dec 03 10:22:16 [rank0]:   File "/opt/conda/lib/python3.10/site-packages/torch/cuda/__init__.py", line 399, in set_device
Dec 03 10:22:16 [rank0]:     torch._C._cuda_setDevice(device)
Dec 03 10:22:16 [rank0]:   File "/opt/conda/lib/python3.10/site-packages/torch/cuda/__init__.py", line 293, in _lazy_init
Dec 03 10:22:16 [rank0]:     torch._C._cuda_init()
Dec 03 10:22:16 [rank0]: RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
Dec 03 10:22:16 [rank1]: Traceback (most recent call last):
Dec 03 10:22:16 [rank1]:   File "/app/run_distributed_inference.py", line 432, in <module>
Dec 03 10:22:16 [rank1]:     main()
Dec 03 10:22:16 [rank1]:   File "/app/run_distributed_inference.py", line 380, in main
Dec 03 10:22:16 [rank1]:     torch.cuda.set_device(device)
Dec 03 10:22:16 [rank1]:   File "/opt/conda/lib/python3.10/site-packages/torch/cuda/__init__.py", line 399, in set_device
Dec 03 10:22:16 [rank1]:     torch._C._cuda_setDevice(device)
Dec 03 10:22:16 [rank1]:   File "/opt/conda/lib/python3.10/site-packages/torch/cuda/__init__.py", line 293, in _lazy_init
Dec 03 10:22:16 [rank1]:     torch._C._cuda_init()
Dec 03 10:22:16 [rank1]: RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

# driver check
user49@gpu-node1 ~/scratch/group1/pipeline_run $ echo "Driver check:"
Driver check:
user49@gpu-node1 ~/scratch/group1/pipeline_run $ nvidia-smi
Wed Dec  3 10:30:29 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.144.06             Driver Version: 550.144.06     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000001:00:00.0 Off |                  Off |
| N/A   27C    P8             14W /   70W |       1MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
user49@gpu-node1 ~/scratch/group1/pipeline_run $ lsmod | grep nvidia
nvidia_uvm           6762496  2
nvidia_drm            126976  0
drm_ttm_helper         16384  1 nvidia_drm
nvidia_modeset       1359872  1 nvidia_drm
video                  77824  1 nvidia_modeset
nvidia              54439936  15 nvidia_uvm,nvidia_modeset
drm_kms_helper        266240  4 drm_shmem_helper,hyperv_drm,drm_ttm_helper,nvidia_drm
drm                   811008  8 drm_kms_helper,drm_shmem_helper,nvidia,hyperv_drm,drm_ttm_helper,nvidia_drm,ttm
user49@gpu-node1 ~/scratch/group1/pipeline_run $ which nvidia-smi
/usr/bin/nvidia-smi

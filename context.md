# .err:

Lmod has detected the following error: These module(s) or extension(s) exist
but cannot be loaded as requested: "apptainer"
   Try: "module spider apptainer" to see how to load the module(s).



Lmod has detected the following error: These module(s) or extension(s) exist
but cannot be loaded as requested: "cuda"
   Try: "module spider cuda" to see how to load the module(s).



srun: error: gpu-node1: task 0: Exited with exit code 1
/app/slurm/run.sh: line 197: 2985003 Aborted                 (core dumped) "${BASE_CMD[@]}" 2>&1
     2985004 Done                    | tee -a "${RANK_LOG}"
srun: error: gpu-node2: task 1: Exited with exit code 134


# .out:

gpu-node2:2985003:2985024 [0] misc/socket.cc:50 NCCL WARN socketProgress: Connection closed by remote peer gpu-node1.int.hpcie.labs.faculty.ie.edu<45074>
gpu-node2:2985003:2985024 [0] NCCL INFO misc/socket.cc:752 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO transport/net_socket.cc:474 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO transport/net.cc:1298 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO proxy.cc:694 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO proxy.cc:874 -> 6 [Progress Thread]

gpu-node2:2985003:2985024 [0] misc/socket.cc:50 NCCL WARN socketProgress: Connection closed by remote peer gpu-node1.int.hpcie.labs.faculty.ie.edu<45074>
gpu-node2:2985003:2985024 [0] NCCL INFO misc/socket.cc:752 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO transport/net_socket.cc:474 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO transport/net.cc:1298 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO proxy.cc:694 -> 6
gpu-node2:2985003:2985024 [0] NCCL INFO proxy.cc:874 -> 6 [Progress Thread]
gpu-node2:2985003:2985022 [0] NCCL INFO [Service thread] Connection closed by localRank 0
gpu-node2:2985003:2985016 [0] NCCL INFO comm 0x8e01d00 rank 1 nranks 2 cudaDev 0 busId 100000 - Abort COMPLETE
[rank1]:[E ProcessGroupNCCL.cpp:577] [Rank 1] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank1]:[E ProcessGroupNCCL.cpp:583] [Rank 1] To avoid data inconsistency, we are taking the entire process down.
[rank1]:[E ProcessGroupNCCL.cpp:1414] [PG 0 Rank 1] Process group watchdog thread terminated with exception: NCCL error: remote process exited or there was a network error, NCCL version 2.20.5
ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.
Last error:
socketProgress: Connection closed by remote peer gpu-node1.int.hpcie.labs.faculty.ie.edu<45074>
Exception raised from checkForNCCLErrorsInternal at /opt/conda/conda-bld/pytorch_1716905971132/work/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1723 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x14790e6d6897 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::checkForNCCLErrorsInternal(std::shared_ptr<c10d::NCCLComm>&) + 0x220 (0x1478add786a0 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::checkAndSetException() + 0x7c (0x1478add788ec in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::watchdogHandler() + 0x180 (0x1478add7db10 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x10c (0x1478add7ee7c in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #5: <unknown function> + 0xdbbf4 (0x14790e2c7bf4 in /opt/conda/lib/python3.10/site-packages/torch/lib/../../../.././libstdc++.so.6)
frame #6: <unknown function> + 0x94ac3 (0x147916838ac3 in /lib/x86_64-linux-gnu/libc.so.6)
frame #7: <unknown function> + 0x126850 (0x1479168ca850 in /lib/x86_64-linux-gnu/libc.so.6)
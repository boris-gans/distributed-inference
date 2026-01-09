# Performance Analysis

## Strong Scaling

### Configuration (Baseline)
- **Nodes:** 1
- **Num Prompts:** 5
- **Batch Size:** 1

| Metric                    | Value        |
|---------------------------|--------------|
| Speedup                   | 1.0 (baseline) |
| Wall Clock Time (s)       | 16.34        |
| Throughput (tokens/sec)   | 15.42        |
| Parallel Efficiency       | 1.0          |
| I/O Time (s)              | 2.68         |
| Communication Time (s)    | 0            |
| Preprocessing Cost (s)    | 0.66         |

The single-node baseline completed 5 prompts generating 252 total tokens in 16.34 seconds. The aggregate throughput of 15.42 tokens per second reflects the sequential processing of prompts through all 26 transformer layers on a single T4 GPU. Model loading required approximately 2.68 seconds, with tokenizer initialization adding 0.66 seconds of preprocessing overhead.

---

### Configuration
- **Nodes:** 2
- **Num Prompts:** 5
- **Batch Size:** 1

| Metric                    | Value    |
|---------------------------|----------|
| Speedup                   | 1.34x    |
| Wall Clock Time (s)       | 12.22    |
| Throughput (tokens/sec)   | 20.62    |
| Parallel Efficiency       | 0.67     |
| I/O Time (s)              | 3.22     |
| Communication Time (s)    | 1.89     |
| Preprocessing Cost (s)    | 0.68     |

The two-node pipeline configuration achieved a speedup of 1.34x over the single-node baseline, completing the same 5 prompts in 12.22 seconds with an aggregate throughput of 20.62 tokens per second. The parallel efficiency of 0.67 reflects the overhead introduced by inter-node communication during each decoding step. Pipeline partitioning split the 26-layer model evenly, with stage 0 (rank 0) handling the embedding layer and layers 0-12, while stage 1 (rank 1) handled layers 13-25, the final normalization, and the language model head.

---

## Weak Scaling

### Configuration (Baseline)
- **Nodes:** 1
- **Num Prompts:** 5
- **Batch Size:** 1

| Metric                    | Value        |
|---------------------------|--------------|
| Speedup                   | 1.0 (baseline) |
| Wall Clock Time (s)       | 16.78        |
| Throughput (tokens/sec)   | 15.08        |
| Parallel Efficiency       | 1.0          |
| I/O Time (s)              | 2.67         |
| Communication Time (s)    | 0            |
| Preprocessing Cost (s)    | 0.67         |

The weak-scaling baseline mirrors the strong-scaling single-node configuration: 5 prompts processed sequentially on one T4 GPU, generating 253 tokens in 16.78 seconds at 15.08 tokens per second.

---

### Configuration
- **Nodes:** 2
- **Num Prompts:** 10
- **Batch Size:** 1

| Metric                    | Value    |
|---------------------------|----------|
| Speedup                   | 1.36x    |
| Wall Clock Time (s)       | 24.67    |
| Throughput (tokens/sec)   | 20.47    |
| Parallel Efficiency       | 0.68     |
| I/O Time (s)              | 3.23     |
| Communication Time (s)    | 2.12     |
| Preprocessing Cost (s)    | 0.68     |

With doubled workload (10 prompts) and doubled resources (2 nodes), the pipeline completed in 24.67 seconds, generating 505 total tokens at an aggregate throughput of 20.47 tokens per second. The weak-scaling efficiency of 0.68 indicates that per-node throughput decreased slightly compared to the single-node baseline due to pipeline communication overhead. The per-prompt latency averaged 2.47 seconds in pipeline mode compared to 3.36 seconds in single-node mode, demonstrating the benefit of splitting the model across stages.

---

## Sensitivity Sweep: Batch Size

### Configuration (Baseline)
- **Nodes:** 2
- **Num Prompts:** 10
- **Batch Size:** 1

| Metric                    | Value    |
|---------------------------|----------|
| Speedup                   | 1.0 (baseline) |
| Wall Clock Time (s)       | 24.67    |
| Throughput (tokens/sec)   | 20.47    |
| Parallel Efficiency       | 0.68     |
| I/O Time (s)              | 3.23     |
| Communication Time (s)    | 2.13     |
| Preprocessing Cost (s)    | 0.68     |

Under batch size 1, the two-node pipeline processed 10 prompts in 24.67 seconds with an aggregate throughput of 20.47 tokens per second. Per-prompt throughput averaged 20.6 tokens per second across individual prompts, indicating stable pipeline behavior with minimal variance.

---

### Configuration
- **Nodes:** 2
- **Num Prompts:** 10
- **Batch Size:** 2

| Metric                    | Value    |
|---------------------------|----------|
| Speedup                   | 1.57x    |
| Wall Clock Time (s)       | 15.67    |
| Throughput (tokens/sec)   | 32.23    |
| Parallel Efficiency       | 0.79     |
| I/O Time (s)              | 3.23     |
| Communication Time (s)    | 1.78     |
| Preprocessing Cost (s)    | 0.68     |

Increasing the batch size to 2 yielded substantial performance improvements. Wall-clock time dropped to 15.67 seconds and aggregate throughput increased to 32.23 tokens per second, representing a 1.57x speedup over the batch-1 baseline. Per-prompt throughput averaged 32.5 tokens per second, indicating better GPU utilization from reduced kernel launch overhead and improved memory access patterns. The batch efficiency of 0.79 demonstrates favorable scaling characteristics at this batch size.

---

### Configuration
- **Nodes:** 2
- **Num Prompts:** 10
- **Batch Size:** 4

| Metric                    | Value    |
|---------------------------|----------|
| Speedup                   | 1.69x    |
| Wall Clock Time (s)       | 14.56    |
| Throughput (tokens/sec)   | 34.68    |
| Parallel Efficiency       | 0.42     |
| I/O Time (s)              | 3.22     |
| Communication Time (s)    | 1.92     |
| Preprocessing Cost (s)    | 0.68     |

At batch size 4, incremental gains diminish compared to batch size 2. Wall-clock time decreased to 14.56 seconds with throughput reaching 34.68 tokens per second—a 1.69x speedup over batch-1. However, the batch efficiency dropped to 0.42, indicating that the additional batching provides diminishing returns. Per-prompt throughput averaged 35.1 tokens per second. This configuration approaches the memory and compute saturation point of the T4 GPUs, consistent with the OOM issues encountered when attempting larger models.

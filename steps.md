# Distributed Inference Runtime Plan

This document captures the class-based layout and execution flow for the revised pipeline described in `README.md`. It assumes a local orchestration node plus a Slurm-managed GPU cluster.

## 1. Prompt Assets (local CPU)
- **`data/prompts_2k.txt` + `data/prompts_4k.txt`**: Two curated text files, 20 prompts each, targeting ~2k and ~4k token budgets.
- **`src/data/prompts.py`**
  - `PromptRecord`: dataclass with `prompt_id`, `variant` (`2k`/`4k`), `prompt`, `token_budget`.
  - `PromptFileSet`: validates both files exist, enforces 20 prompts per file, attaches IDs (shared across variants).
  - `PromptRepository`: exposes `load_all()` → `list[PromptRecord]`, caches metadata for downstream consumers.

## 2. Prompt Table & Local Baseline
- **`src/data/table.py`**
  - `PromptDataFrameBuilder`: converts `PromptRecord` objects into a pandas `DataFrame`.
  - Columns: `prompt_id`, `variant`, `prompt`, `token_budget`, `source_file`, `local_completion`, `local_latency_ms`, `status`.
- **`src/inference/openai_client.py`**
  - `OpenAICompletionClient`: wraps the OpenAI Python SDK to call the remote Llama 3.3 70B endpoint (hosted outside the cluster).
  - Handles retries, rate limiting, and response normalization.
- **`src/inference/local_runner.py`**
  - `LocalInferenceRunner`: iterates over the DataFrame, calls `OpenAICompletionClient`, and writes `local_completion` + timing back into the DataFrame.
  - Persists the table to `results/baseline_prompts.parquet` for reuse by distributed runs.

## 3. Distributed Experiment Orchestration (Slurm)
- **`src/parallelism/strategy.py`**
  - `ParallelismStrategy` (abstract): defines `name`, `slurm_constraints`, `deepspeed_args`, `postprocess(df)` and `expected_jobs`.
  - Concrete subclasses align with the four README techniques:
    1. `TensorParallelAcrossNodes` (TP-only across nodes).
    2. `PipelineParallelAcrossNodes` (PP-only across nodes).
    3. `TensorParallelSingleNode` (TP-only single node).
    4. `HybridTensorPipeline` (TP intra-node + PP inter-node).
- **`src/slurm/job_factory.py`**
  - `SlurmConfig`: dataclass describing nodes, GPUs, walltime, env, container, output paths.
  - `SlurmJobFactory`: builds configs per strategy run (40 prompts = 20×2 variants). Injects prompt shard paths and output locations.
- **`src/slurm/job_manager.py`**
  - `SlurmJobManager`: submits jobs, monitors status via `sacct`/`squeue`, and triggers callback once logs land locally.
  - Responsible for “dynamic config”: after each job finishes, it asks the relevant strategy for the next config (allowing parameter sweeps or retries).
- **`src/orchestration/pipeline.py`**
  - `DistributedExperimentPipeline`: runs strategies sequentially, ensures each processes both prompt variants, and hands results to the `ResultCollector`.

## 4. Result Collection & Metrics (local CPU)
- **`src/results/collector.py`**
  - `ResultCollector`: loads JSON/Parquet artifacts produced on the cluster, normalizes schema, and merges them back into the master DataFrame under columns `strategy`, `rank`, `gpu_id`, `cluster_completion`, `cluster_latency_ms`, `trace_path`.
- **`src/metrics/correctness.py`**
  - `CorrectnessEvaluator`: after all four strategies complete, computes your “correctness” metric per prompt/strategy pair (uses the baseline completions stored earlier).
- **`src/metrics/report.py`**
  - `RunSummary`: emits CSV/Markdown summaries plus plots (latency vs. correctness, throughput vs. token length).

## 5. Execution Steps
1. **Prepare prompts locally** using `PromptFileSet`. Persist canonical IDs to keep rows aligned between 2k and 4k variants.
2. **Build the prompt DataFrame** (`PromptDataFrameBuilder`). Store to disk before inference so downstream scripts can reload the same ordering.
3. **Run baseline OpenAI inference** with `LocalInferenceRunner`; update `local_completion` + metrics columns.
4. **Launch distributed strategies sequentially** via `DistributedExperimentPipeline`:
   - For each strategy, instantiate `ParallelismStrategy` subclass.
   - `SlurmJobFactory` generates the job script/config dynamically (include specific TP/PP/DP settings described in README).
   - `SlurmJobManager` submits, monitors, and triggers data ingestion when outputs arrive.
5. **Ingest job outputs** with `ResultCollector`, append to the shared DataFrame, and checkpoint to `results/run_<timestamp>.parquet`.
6. **Compute correctness metrics** (outside the cluster) and export a summary (CSV/Markdown) comparing all four strategies.



# Build steps

### PromptFileSet.validate  
*src/data/prompts.py (line 29)*  
Verifies that the 2k/4k prompt files exist, contain exactly 20 records, and have aligned IDs.

### PromptFileSet.iter_records  
*src/data/prompts.py (line 33)*  
Scans both files, attaches `prompt_id`, `variant`, and `token_budget`, and yields deterministic `PromptRecord` objects for downstream sharding.

### PromptRepository.load_all  
*src/data/prompts.py (line 45)*  
Loads and caches the full list of prompt records by invoking the validated file-set iterator exactly once.

### PromptRepository.get_by_id  
*src/data/prompts.py (line 49)*  
Filters the cached prompt records to provide requested prompt variants to strategy-specific shard builders.

### PromptDataFrameBuilder.build  
*src/data/table.py (line 21)*  
Constructs the canonical pandas DataFrame containing all schema fields (prompt metadata plus placeholder completion/latency columns).

### PromptDataFrameBuilder.persist  
*src/data/table.py (line 25)*  
Writes the DataFrame (likely Parquet) so Slurm jobs can reuse it.

### OpenAICompletionClient.complete_prompt / complete_batch / normalize_response  
*src/inference/openai_client.py (lines 16–25)*  
Wraps the OpenAI SDK: authenticates calls, performs single/batch inference with retry/backoff, and normalizes response payloads into fields like `text` and `latency_ms` for the baseline table.

### LocalInferenceRunner  
*src/inference/local_runner.py*  
- **run (line 24)**: Iterates through DataFrame rows, calls the completion client, and stores completions/latencies.  
- **persist (line 28)**: Saves the augmented table (e.g., `results/baseline_prompts.parquet`).  
- **attach_metrics (line 32)**: Adds aggregated timing columns needed by downstream evaluators.

---

# Cluster Runtime & Node Workers

### run_distributed_stub  
*src/inference/runtime.py (line 149)*  
Cluster-branch entry point: should initialize DeepSpeed/NCCL, parse CLI args, and delegate to Slurm/node-runtime helpers.

### Node-level helpers  
*src/inference/node_runtime.py (lines 9–41)*  

- **initialize_distributed_environment**: Configure NCCL/torch.distributed for the chosen parallelism strategy.  
- **configure_parallel_groups**: Form TP/PP/DP process groups per rank.  
- **load_model_weights**: Load the correct tensor shard (likely via DeepSpeed engine initialization).  
- **prepare_prompt_shard**: Retrieve per-rank prompt subset.  
- **run_generation_loop**: Perform inference over local batches.  
- **persist_rank_outputs**: Write JSONL/Parquet outputs plus traces to shared storage.  
- **finalize_rank**: Report metrics/profiling back to the controller.

---

# Strategy Orchestration & Slurm Integration

### Strategy implementations  
*src/parallelism/strategy.py (lines 44–157)*  
All four strategy classes must implement:

- **describe_topology**  
- **slurm_constraints**  
- **deepspeed_args**  
- **expected_jobs**  
- **postprocess**

These must match the configurations described in `README.md`.

### DistributedExperimentPipeline  
*src/orchestration/pipeline.py (lines 17–26)*  
Responsible for orchestrating strategy execution:

- `run`: Execute all strategies in order.  
- `run_strategy`: Submit the job, wait for completion, collect outputs.  
- `collect_results`: Gather all artifacts and expose them to the metrics stage.

### SlurmJobFactory  
*src/slurm/job_factory.py (lines 33–41)*  
Builds Slurm job assets from strategies:

- **create_config**: Convert strategy specs into #SBATCH parameters.  
- **render_script**: Produce executable sbatch shell script (likely running Apptainer + `python -m src.inference.runtime`).  
- **script_path**: Provide deterministic location for the script.

### SlurmJobManager  
*src/slurm/job_manager.py (lines 18–27)*  
Wraps Slurm operations:

- **submit**: Call `sbatch`.  
- **monitor**: Poll `squeue`/`sacct` until job completion.  
- **collect_artifacts**: Trigger result collection when logs and outputs appear.

---

# Result Evaluation & Reporting

### ResultCollector  
*src/results/collector.py (lines 21–34)*  

- **ingest_run**: Read each strategy’s output directory.  
- **merge**: Combine results into unified structures.  
- **to_dataframe**: Convert artifacts to a consolidated DataFrame.  
- **iter_traces**: Yield trace files (e.g., Nsight).

### CorrectnessEvaluator  
*src/metrics/correctness.py (lines 20–28)*  

- **evaluate**: Compare completions vs. baseline references.  
- **score_prompt**: Compute string distance, similarity, etc.  
- **summarize**: Produce aggregate correctness statistics per strategy/variant.

### RunSummary  
*src/metrics/report.py (lines 20–28)*  

- **to_markdown**: Produce publication-ready Markdown tables.  
- **to_csv**: Export metrics to CSV.  
- **build_plots**: Create plots such as latency vs. accuracy.


# Cluster Runtime Build Steps (in-depth)


## 5. Execution Steps

### 1. **Prepare prompts locally** using `PromptFileSet`. Persist canonical IDs to keep rows aligned between 2k and 4k variants.

### 2. **Build the prompt DataFrame** (`PromptDataFrameBuilder`). Store to disk before inference so downstream scripts can reload the same ordering.

### 3. **Run baseline OpenAI inference** with `LocalInferenceRunner`; update `local_completion` + metrics columns.

---

### 4. **Launch distributed strategies sequentially** via `DistributedExperimentPipeline`

For each configured strategy (e.g. different PP/TP setups, batch sizes, etc.):

#### 4.1. Materialize a **strategy config object**

From your `ParallelismStrategy` subclass, construct a **pure data object**:

* `strategy_id` (e.g. `"llama8b_pp2_tp1"`)
* model info:

  * `model_name`, `model_path` (cluster path to LLaMA 8B weights)
* parallelism:

  * `num_nodes`
  * `gpus_per_node`
  * `pp_size`
  * `tp_size`
  * optional `dp_size`
* runtime:

  * `batch_size`
  * `max_new_tokens`
  * `precision` (fp16/bf16)
  * `profiling_enabled` flags (Nsight/perf)
* paths:

  * `prompt_df_path` (the parquet from Step 2, on shared storage)
  * `output_root` (e.g. `/scratch/$USER/experiments/run_<ts>/<strategy_id>/`)
  * `logs_dir`, `profiling_dir` (subdirs under `output_root`)
* bookkeeping:

  * `run_id`
  * `slurm_time_limit`, `partition`, `account` if needed

This is still in Python memory at this point.

#### 4.2. Emit a **strategy experiment config JSON** (for the container entry script)

Write a JSON file, e.g.:

`/scratch/$USER/experiments/run_<ts>/<strategy_id>/exp_config.json`

Containing:

* all the fields above, plus:

  * `appainter_image` (name or path of the container)
  * `deepspeed_config_path` (if you use a separate DeepSpeed JSON)
  * `world_size` (pp × tp × dp)
  * `output_shard_pattern` (e.g. `"shard_{rank}.jsonl"`)
  * `metrics_file` (e.g. `"metrics_rank_{rank}.json"`)

This is what `run_distributed_inference.py` will read inside the container.

#### 4.3. Use `SlurmJobFactory` to generate a **concrete Slurm script**

For this strategy, write a file like:

`/scratch/$USER/experiments/run_<ts>/<strategy_id>/job.slurm`

The factory fills in:

* `#SBATCH` header:

  * `--job-name=llama8b_${strategy_id}`
  * `--nodes=<num_nodes>`
  * `--gres=gpu:<gpus_per_node>`
  * `--ntasks-per-node=<gpus_per_node>` (or 1 if DeepSpeed handles spawning)
  * `--time=<slurm_time_limit>`
  * partition/account/etc.
* environment setup:

  * any `module load` you still need
  * NCCL env (if not handled elsewhere): `NCCL_DEBUG`, etc.
* Appainter + DeepSpeed launch:

```bash
appainter pull ${APPAINTER_IMAGE}   # optional if pulling from registry

appainter run \
    --mount /scratch/$USER/experiments:/exp \
    --mount /scratch/$USER/models:/models \
    ${APPAINTER_IMAGE} \
    deepspeed \
      --num_nodes=${NUM_NODES} \
      --num_gpus=${GPUS_PER_NODE} \
      /app/run_distributed_inference.py \
      --config /exp/run_${RUN_ID}/${STRATEGY_ID}/exp_config.json
```

You now have a **fully self-contained job** for this strategy: config + Slurm script + known output directory.

#### 4.4. Submit and track the job with `SlurmJobManager`

For each strategy:

1. Call `sbatch job.slurm`, capture the returned `job_id`.

2. Persist a small manifest locally and/or on cluster, e.g.:

   `run_<ts>/manifest.json` entry:

   ```json
   {
     "strategy_id": "llama8b_pp2_tp1",
     "job_id": 123456,
     "exp_config_path": "/scratch/.../exp_config.json",
     "output_root": "/scratch/.../outputs",
     "status": "SUBMITTED"
   }
   ```

3. Poll Slurm:

   * `squeue -j <job_id>` while PENDING/RUNNING
   * `sacct -j <job_id> --format=State,Elapsed,MaxRSS,...` after completion

4. Update manifest with:

   * final `status` (COMPLETED/FAILED/TIMEOUT)
   * basic Slurm resource metrics from `sacct` (walltime, memory, etc.)
   * path to `slurm-<job_id>.out`

The **pipeline logic** at this level should:

* run strategies **sequentially** (as you said)
* do not move to `ResultCollector` for a given strategy until `status == COMPLETED`

---

### 4.5. What happens on the cluster inside the container (for each job)

`run_distributed_inference.py` (inside the container) should:

1. Parse `--config` and load `exp_config.json`.

2. Initialize distributed:

   * `deepspeed.init_distributed()` or equivalent
   * interpret `RANK`, `WORLD_SIZE`, etc. from Slurm/DeepSpeed

3. Construct pipeline & tensor parallel groups:

   * assign `pp_rank`, `tp_rank`, `data_rank` based on `rank`

4. Load the appropriate **model partition** for this rank:

   * pointer to `/models/llama8b/...`
   * either:

     * DeepSpeed handles partitioning, or
     * your code loads only the layers assigned to `pp_rank`

5. Load the prompt dataframe from shared storage:

   * `/exp/run_<ts>/prompts.parquet`
   * possibly shard across data ranks, but always preserve `prompt_id`

6. Run inference in batches:

   * for each batch:

     * forward through PP+TP
     * measure per-batch latency, tokens/sec

7. Each rank writes **its own shard output** to:

   `/exp/run_<ts>/<strategy_id>/outputs/shard_<rank>.jsonl`

   With rows like:

   ```jsonl
   {"prompt_id": 123, "strategy_id": "llama8b_pp2_tp1", "completion": "...", "latency_ms": 42.7, "tokens_generated": 128}
   ```

8. Each rank writes a **rank-level metrics file**:

   `/exp/run_<ts>/<strategy_id>/metrics/metrics_rank_<rank>.json`
   (e.g. average latency, throughput, etc.)

9. If profiling enabled:

   * wrap main loop in Nsight/perf
   * write profiler outputs to:

     * `/exp/run_<ts>/<strategy_id>/profiling/`

10. Exit cleanly; let Slurm produce `slurm-<job_id>.out`.

At this point, all artifacts needed by `ResultCollector` exist on the shared filesystem.

---

### 5. **Ingest job outputs** with `ResultCollector`, append to the shared DataFrame, and checkpoint

For each strategy with `status == COMPLETED`:

#### 5.1. Locate and validate shard outputs

* Look under:

  * `output_root = /scratch/.../run_<ts>/<strategy_id>/outputs/`
* Expect exactly `WORLD_SIZE` shard files:

  * `shard_0.jsonl`, `shard_1.jsonl`, …
* If missing / incomplete, mark strategy as failed or partial.

#### 5.2. Load and concatenate shard data

* Read all `shard_*.jsonl` into a temporary DataFrame:

  * columns: `prompt_id`, `completion`, `latency_ms`, `tokens_generated`, etc.
* Add a `strategy_id` column.

#### 5.3. Join with the master prompt DataFrame

* Reload the canonical prompt parquet saved in Step 2 (local or from cluster).

* Left-join on `prompt_id`:

  * existing columns:

    * `prompt_text_2k`, `prompt_text_4k`, `local_completion`, baseline metrics…
  * new strategy-specific columns (either wide or long form):

    * wide: `llama8b_pp2_tp1_completion`, `llama8b_pp2_tp1_latency_ms`, etc.
    * long: extra column `engine` and one `completion` column.

* Update in-memory master DataFrame for this run.

#### 5.4. Attach cluster metrics

* Optionally parse:

  * per-rank metrics JSONs
  * sacct metrics for this `job_id`
* Aggregate them (e.g. mean latency per prompt, total tokens/sec, GPU utilization summary).
* Store into a separate “cluster_metrics” table keyed by `strategy_id`, or add strategy-level columns to a summary DataFrame.

#### 5.5. Checkpoint results to disk

* After each strategy ingestion, write:

  * `results/run_<timestamp>.parquet` (full per-prompt DF with all strategies)
  * optionally:

    * `results/cluster_metrics_run_<timestamp>.parquet`
    * a JSON manifest summarizing which strategies completed and where their logs are.

This gives you **safe intermediate checkpoints** even if later strategies fail.

---

### 6. **Compute correctness metrics** and export a comparative summary

Once all target strategies have run and been ingested:

#### 6.1. Compute per-strategy correctness vs. baseline labels

* Using the combined DataFrame from Step 5:

  * Compare each strategy’s completion to:

    * ground-truth label (if you have one), or
    * baseline OpenAI completion (`local_completion`)
* Compute metrics per strategy:

  * accuracy / exact match
  * distance metrics (e.g. string similarity, BLEU, etc.)
* Store in a **summary DataFrame** with one row per `strategy_id`.

#### 6.2. Compute performance metrics per strategy

From ingested metrics & sacct data:

* Effective throughput (tokens/sec, prompts/sec)
* Median/percentile latencies
* Resource usage (wallclock, GPU-hours, memory from sacct)

Add these columns to the same summary DataFrame.

#### 6.3. Export human-readable summaries

* Write:

  * `results/summary_run_<timestamp>.csv`
  * `results/summary_run_<timestamp>.md` (nicely formatted comparison of strategies: accuracy vs latency vs throughput vs cost)
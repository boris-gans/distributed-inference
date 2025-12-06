#!/usr/bin/env python
"""Two-stage pipeline-parallel inference entrypoint for Slurm launches.

This script is invoked by slurm/run.sh inside the container. It:
  - Initializes torch.distributed via DeepSpeed helpers
  - Downloads model/tokenizer to shared cache on /workspace/pipeline_run
  - Splits the Llama model into two pipeline stages (one GPU per node)
  - Streams hidden states between stages to generate text greedily
  - Emits per-prompt JSONL outputs and basic throughput metrics

Notes:
  - Designed for world_size == 2 (pipeline_parallel_size == 2).
  - Uses BF16 and micro-batch size 1.
  - Assumes prompt JSONL lives on shared storage (EXPERIMENT_ROOT).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


@dataclass
class ExpConfig:
    model_path: Optional[Path]
    model_name: str
    prompt_path: Path
    max_new_tokens: int
    temperature: float = 0.0
    seed: int = 42
    batch_size: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed pipeline-parallel inference entrypoint.")
    parser.add_argument("--exp_config", type=Path, required=True, help="Path to experiment config JSON.")
    parser.add_argument("--ds_config", type=Path, required=True, help="Path to DeepSpeed config JSON.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Shared output directory.")
    parser.add_argument("--prompt_limit", type=int, default=None, help="Optional limit on number of prompts to run.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--pipeline_size", type=int, default=None, help="Optional pipeline/world size override.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with Path(path).open("r") as handle:
        return json.load(handle)


def _env_int(name: str) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_exp_config(path: Path) -> ExpConfig:
    cfg = load_json(path)
    try:
        model_section = cfg["model"]
        model_path = model_section.get("model_path")
        return ExpConfig(
            model_path=Path(model_path) if model_path else None,
            model_name=model_section["model_name"],
            prompt_path=Path(cfg["inputs"]["prompt_path"]),
            max_new_tokens=int(cfg["inference"].get("max_new_tokens", 64)),
            temperature=float(cfg["inference"].get("temperature", 0.0)),
            seed=int(cfg.get("seed", 42)),
            batch_size=int(cfg.get("inference", {}).get("batch_size", 1)),
        )
    except KeyError as exc:
        raise SystemExit(f"Missing required exp_config field: {exc}") from exc


def init_logging(output_dir: Path, rank: int) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"rank_{rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s][rank=%(rank)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    logger = logging.getLogger("pipeline_inference")
    logger = logging.LoggerAdapter(logger, extra={"rank": rank})
    return logger


def init_distributed() -> int:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if not dist.is_initialized():
        raise SystemExit("torch.distributed was not initialized.")
    return dist.get_rank()


def set_hf_cache(root: Path, logger: logging.Logger) -> None:
    cache_dir = root / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    logger.info("Using Hugging Face cache at %s", cache_dir)


def load_tokenizer(model_name: str, cache_dir: Path, logger: logging.Logger):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False,
        local_files_only=True,
    )
    # Ensure pad token exists for attention masks
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("Loaded tokenizer for %s", model_name)
    return tokenizer


def partition_model(
    model_name: str,
    cache_dir: Path,
    split_idx: Optional[int],
    device: torch.device,
    stage: int,
    logger: logging.Logger,
) -> tuple[nn.Module, int]:

    logger.info("Loading model %s for pipeline stage %d...", model_name, stage)

    # STEP 1 — Load config only (safe, tiny memory)
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    config.use_cache = False
    config.pretraining_tp = 1
    config._attn_implementation = "eager"

    split = split_idx or (num_layers // 2)

    # STEP 2 — Build minimal device_map so this rank loads only its layers
    if stage == 0:
        # Layers 0 ... split-1 live on this GPU
        layer_ids = list(range(0, split))
    else:
        # Layers split ... num_layers-1 live on this GPU
        layer_ids = list(range(split, num_layers))

    device_map = {}

    for i in range(num_layers):
        if i in layer_ids:
            device_map[f"model.layers.{i}"] = device.type  # cuda
        else:
            device_map[f"model.layers.{i}"] = "disk"       # throwaway layers; other node loads these but must specify

    if stage == 0:
        device_map["model.embed_tokens"] = device.type

        # Rank 0 does NOT take lm_head or final norm, load anyway to avoid error
        device_map["model.norm"] = "disk"
        device_map["lm_head"] = "disk"
    else:
        # Rank 1 keeps final layers + norm + lm_head
        device_map["model.norm"] = device.type
        device_map["lm_head"] = device.type

        # Load ALL required modules onto CPU, even if layer 2 won't use them (avoid error)
        device_map["model.embed_tokens"] = "disk"

    logger.info("Stage %d device_map=%s", stage, device_map)

    # STEP 3 — Load ONLY the layers for this stage (zero CPU spike)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=device_map,  # <<< prevents full CPU load
        offload_folder=f"/tmp/offload_rank_{stage}",  # must be unique per rank
    )

    logger.info("Loaded model...")

    # HF loads submodules directly to the right GPU → no manual trimming needed
    # But your pipeline code expects a simplified Module, so extract it:

    base = model.model  # type: ignore[attr-defined]

    stage_module = nn.Module()

    if stage == 0:
        stage_module.embed_tokens = base.embed_tokens
        stage_module.layers = nn.ModuleList([base.layers[i] for i in layer_ids])
        stage_module.config = config
    else:
        stage_module.layers = nn.ModuleList([base.layers[i] for i in layer_ids])
        stage_module.norm = base.norm

        # Reconstruct lm_head so it can sit on a different device if needed
        lm_head = model.lm_head
        cloned_head = nn.Linear(
            lm_head.in_features,
            lm_head.out_features,
            bias=False,
            dtype=lm_head.weight.dtype,
            device=lm_head.weight.device,
        )
        cloned_head.weight.data.copy_(lm_head.weight.data)
        stage_module.lm_head = cloned_head
        stage_module.config = config

    logger.info(
        "Stage %d loads %d layers (%s) on device %s",
        stage,
        len(layer_ids),
        "0..split" if stage == 0 else "split..end",
        device,
    )

    stage_module.to(device)
    stage_module.eval()
    torch.cuda.empty_cache()

    return stage_module, hidden_size


def read_prompts(prompt_path: Path) -> List[str]:
    prompts: List[str] = []
    with prompt_path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompts.append(record["prompt"])
    if not prompts:
        raise SystemExit(f"No prompts found in {prompt_path}")
    return prompts


def stage0_forward(
    module: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    use_nvtx: bool,
) -> torch.Tensor:
    if use_nvtx:
        torch.cuda.nvtx.range_push("stage0_forward")
    hidden = module.embed_tokens(input_ids)
    for layer in module.layers:
        hidden = layer(
            hidden_states=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )[0]
    if use_nvtx:
        torch.cuda.nvtx.range_pop()
    return hidden


def stage1_forward(
    module: nn.Module,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    use_nvtx: bool,
) -> torch.Tensor:
    if use_nvtx:
        torch.cuda.nvtx.range_push("stage1_forward")
    for layer in module.layers:
        hidden = layer(
            hidden_states=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )[0]
    hidden = module.norm(hidden)
    logits = module.lm_head(hidden)
    if use_nvtx:
        torch.cuda.nvtx.range_pop()
    return logits


def maybe_log_sacct(logger: logging.Logger, output_dir: Path) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return
    sacct = shutil.which("sacct")  # type: ignore[attr-defined]
    if sacct is None:
        return
    try:
        cmd = [sacct, "-j", job_id, "--format=JobID,JobName%30,State,Elapsed,MaxRSS"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        summary_path = output_dir / f"sacct_{job_id}.txt"
        summary_path.write_text(result.stdout)
        logger.info("Wrote sacct summary to %s", summary_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to log sacct summary: %s", exc)


def generate_pipeline(
    stage_module: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    eos_token_id: int,
    hidden_size: int,
    device: torch.device,
    rank: int,
    use_nvtx: bool,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Pipeline loop: rank 0 sends hidden states, rank 1 completes and writes outputs."""
    output_path = output_dir / f"completions_rank_{rank}.jsonl"
    if rank == 1:
        writer = output_path.open("w")
    start_all = time.time()

    with torch.inference_mode():
        for prompt_idx, prompt in enumerate(prompts):
            if rank == 0:
                # Tokenize on rank 0 and broadcast initial length to rank 1
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
                input_ids = inputs["input_ids"].to(device)
                seq_len = input_ids.size(1)
                attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
                position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

                # Inform stage 1 of prompt length and tokens
                seq_tensor = torch.tensor([seq_len], device=device, dtype=torch.long)
                dist.send(seq_tensor, dst=1)
                dist.send(input_ids, dst=1)

                for step in range(max_new_tokens):
                    hidden = stage0_forward(stage_module, input_ids, attention_mask, position_ids, use_nvtx)
                    # Send sequence length and hidden states to stage 1
                    seq_tensor = torch.tensor([hidden.size(1)], device=device, dtype=torch.long)
                    dist.send(seq_tensor, dst=1)
                    dist.send(hidden, dst=1)

                    next_token = torch.empty((1,), device=device, dtype=torch.long)
                    dist.recv(next_token, src=1)

                    input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
                    attention_mask = torch.ones((1, input_ids.size(1)), device=device, dtype=torch.long)
                    position_ids = torch.arange(input_ids.size(1), device=device, dtype=torch.long).unsqueeze(0)

                    if next_token.item() == eos_token_id:
                        break

            else:
                # Receive initial prompt length
                seq_tensor = torch.empty((1,), device=device, dtype=torch.long)
                dist.recv(seq_tensor, src=0)
                seq_len = int(seq_tensor.item())
                input_ids = torch.empty((1, seq_len), device=device, dtype=torch.long)
                dist.recv(input_ids, src=0)

                prompt_start = time.time()
                generated = 0
                for step in range(max_new_tokens):
                    seq_tensor = torch.empty((1,), device=device, dtype=torch.long)
                    dist.recv(seq_tensor, src=0)
                    seq_len = int(seq_tensor.item())
                    hidden = torch.empty((1, seq_len, hidden_size), device=device, dtype=torch.bfloat16)
                    dist.recv(hidden, src=0)

                    attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
                    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

                    logits = stage1_forward(stage_module, hidden, attention_mask, position_ids, use_nvtx)
                    next_token = logits[:, -1, :].argmax(dim=-1)
                    generated += 1

                    dist.send(next_token, dst=0)
                    input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

                    if int(next_token.item()) == eos_token_id:
                        break

                elapsed = time.time() - prompt_start
                text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                record = {
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "completion": text,
                    "tokens_generated": generated,
                    "latency_s": elapsed,
                    "throughput_tok_s": generated / max(elapsed, 1e-6),
                }
                writer.write(json.dumps(record) + "\n")
                writer.flush()
                logger.info(
                    "Prompt %d done: %d tokens in %.2fs (%.2f tok/s)",
                    prompt_idx,
                    generated,
                    elapsed,
                    record["throughput_tok_s"],
                )

    if rank == 1:
        total_elapsed = time.time() - start_all
        logger.info("Completed %d prompts in %.2fs", len(prompts), total_elapsed)
        writer.close()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = load_exp_config(args.exp_config)
    ds_cfg = load_json(args.ds_config)

    # Optional overrides: CLI > env > config
    prompt_limit = args.prompt_limit or _env_int("PROMPT_LIMIT")
    batch_override = args.batch_size or _env_int("BATCH_SIZE")
    pipeline_override = args.pipeline_size or _env_int("PIPELINE_SIZE") or _env_int("NUM_NODES")
    if batch_override is not None:
        exp_cfg.batch_size = batch_override

    # Enable NVTX markers when PROFILER=nsys
    use_nvtx = os.environ.get("PROFILER", "").lower() == "nsys"

    # Initialize distributed after env is set
    rank = init_distributed()
    logger = init_logging(output_dir, rank)
    world_size = dist.get_world_size()
    logger.info("Initialized torch.distributed (rank=%d, world_size=%d)", rank, world_size)
    # Ensure shared cache lives on the mounted workspace
    exp_root = output_dir.parent
    set_hf_cache(exp_root, logger)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(device)

    torch.manual_seed(exp_cfg.seed)
    torch.cuda.manual_seed_all(exp_cfg.seed)

    world_size = dist.get_world_size()
    pipeline_size = (
        pipeline_override
        or int(ds_cfg.get("distributed_training", {}).get("pipeline_parallel_size", 0))
        or world_size
    )
    if world_size != pipeline_size:
        logger.warning(
            "world_size (%d) != pipeline_parallel_size (%d); proceeding anyway.",
            world_size,
            pipeline_size,
        )

    prompts = read_prompts(exp_cfg.prompt_path)
    if prompt_limit is not None:
        prompts = prompts[:prompt_limit]
        logger.info("Applying prompt limit: %d prompts", len(prompts))
    if exp_cfg.batch_size > 1:
        logger.info(
            "Batch size override set to %d; prompts will be processed sequentially per item.",
            exp_cfg.batch_size,
        )

    cache_dir = Path(os.environ["HF_HOME"])
    # If model_path is provided, use it; otherwise fall back to model_name (HF hub)
    model_ref = str(exp_cfg.model_path) if exp_cfg.model_path else exp_cfg.model_name
    tokenizer = load_tokenizer(model_ref, cache_dir, logger)

    config = AutoConfig.from_pretrained(model_ref, cache_dir=cache_dir, local_files_only=True)
    split_idx = max(1, config.num_hidden_layers // pipeline_size) if pipeline_size > 1 else config.num_hidden_layers

    stage_module, hidden_size = partition_model(
        model_name=model_ref,
        cache_dir=cache_dir,
        split_idx=split_idx,
        device=device,
        stage=rank,
        logger=logger,
    )

    dist.barrier()
    logger.info("Starting pipeline generation for %d prompts", len(prompts))

    generate_pipeline(
        stage_module=stage_module,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=exp_cfg.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=hidden_size,
        device=device,
        rank=rank,
        use_nvtx=use_nvtx,
        output_dir=output_dir,
        logger=logger,
    )

    dist.barrier()
    if rank == 1:
        maybe_log_sacct(logger, output_dir)
    dist.barrier()


if __name__ == "__main__":
    main()

"""Runtime helpers for both local debug and distributed inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence
from dotenv import load_dotenv

import torch

from .. import utils
from src.data.prompts import PromptRepository, PromptFileSet
from src.data.table import PromptDataFrameBuilder
from src.inference.fireworksAI_client import FireworksAICompletionClient
from src.inference.local_runner import LocalInferenceRunner


load_dotenv()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the inference entrypoint."""
    parser = argparse.ArgumentParser(
        description="DeepSeek distributed inference entry point."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="results/local_debug.jsonl",
        help="Optional path to store JSONL completions.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Reuse existing dataframes saved as parquets earlier.",
    )
    parser.add_argument(
        "--no-override",
        dest="override",
        action="store_false",
        help="Reuse existing dataframes saved as parquets earlier.",
    )
    parser.set_defaults(override=True)
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Reuse existing dataframes saved as parquets earlier ('2k'/'4k').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Reuse existing dataframes saved as parquets earlier.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts per inference step.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model identifier used for remote execution.",
    )
    parser.add_argument(
        "--local_debug",
        action="store_true",
        help="Run a CPU-only mock using torch.no_grad() and skip DeepSpeed/NCCL.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def load_prompts(path: Path) -> List[str]:
    """Load newline-delimited prompts, ensuring the file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def chunked(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    """Yield fixed-size chunks from the provided sequence."""
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_debug_model() -> torch.nn.Module:
    """Construct the toy PyTorch module used for CPU-only debugging."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for --local_debug mode. Install torch before running."
        )

    class SimpleDebugModel(torch.nn.Module):  # type: ignore[misc]
        """Tiny module that maps ASCII codes to a scalar score."""

        def __init__(self) -> None:
            """Initialize the projection layer with deterministic weights."""
            super().__init__()
            self.proj = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.constant_(self.proj.weight, 1.0 / 255.0)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Compute mean ASCII value per prompt as a stand-in for logits."""
            return inputs.mean(dim=1, keepdim=True)

    return SimpleDebugModel()


def run_local_debug(
    prompts: Sequence[str],
    *,
    batch_size: int,
    tracker: utils.MetricsTracker,
) -> List[dict]:
    """Execute the CPU-only debug path and emit mock completions."""
    model = _build_debug_model()
    model.eval()

    outputs: List[dict] = []
    start_wall = utils.monotonic_time_s()

    for batch in chunked(prompts, batch_size):
        with utils.timed_phase(tracker, "batch_latency_s"):
            ascii_tensors = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor([ord(ch) for ch in text], dtype=torch.float32)
                    for text in batch
                ],
                batch_first=True,
            )
            with torch.no_grad():
                scores = model(ascii_tensors).squeeze(-1)

        tracker.increment("prompts_processed", len(batch))
        for prompt, score in zip(batch, scores.tolist()):
            outputs.append(
                {
                    "prompt": prompt,
                    "score": score,
                    "debug_completion": f"[local-debug] score={score:.3f}",
                }
            )

    total_wall = utils.monotonic_time_s() - start_wall
    outputs.append(
        {
            "metrics": {
                "total_time_s": total_wall,
                "throughput_qps": tracker.throughput(
                    "prompts_processed", max(total_wall, 1e-6)
                ),
                **tracker.summary(),
            }
        }
    )
    return outputs


def run_distributed_stub(args: argparse.Namespace) -> None:
    """Submit the single pipeline-parallel job via the Slurm helper script."""
    launcher = Path(__file__).resolve().parents[2] / "slurm" / "launch_pipeline.sh"
    if not launcher.exists():
        raise SystemExit(
            f"Slurm launcher script not found at {launcher}. "
            "Run with --local_debug or create the launcher script."
        )

    if shutil.which("sbatch") is None:
        raise SystemExit(
            "Slurm (sbatch) is not available on this machine. "
            "Run with --local_debug locally, or execute launch_pipeline.sh on your cluster login node."
        )

    env = os.environ.copy()
    env.setdefault(
        "PIPELINE_ROOT",
        str(Path(__file__).resolve().parents[2] / "experiments" / "pipeline_run"),
    )

    try:
        subprocess.run(
            ["bash", str(launcher)],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"sbatch submission failed with exit code {exc.returncode}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for both local debug and forthcoming cluster modes."""
    args = parse_args(argv)
    logger = utils.setup_logging(level=args.log_level.upper())

    tracker = utils.MetricsTracker()

    # Clean args
    if args.limit is not None and args.limit > 20:
        logger.warning("Limit can be at most 20, ignoring argument.")
        args.limit = None
    if args.variant is not None and args.variant not in ("2k", "4k"):
        logger.warning("Variant must be either '2k' or '4k', ignoring argumnet.")
        args.variant = None


    logger.info("Loading prompts and creating DataFrame")

    df = None
    df_acc = None
    prompts: List[str] = []

    if not args.override:
        print(Path(os.getenv("SAVE_PATH_ACC")).exists())
        if Path(os.getenv("SAVE_PATH_ACC")).exists():
            logger.info("Found df_acc parquet, loading and skipping initialization.")
            df_builder = PromptDataFrameBuilder()
            df_acc = df_builder.load_df_from_parquet(path=os.getenv("SAVE_PATH_ACC"))

        if df_acc is None and Path(os.getenv("SAVE_PATH_BASE")).exists():
            # If df_acc is a valid df then we don't care about the base df
            logger.info("Found base df parquet, loading and skipping initialization.")
            df_builder = PromptDataFrameBuilder()
            df = df_builder.load_df_from_parquet(path=os.getenv("SAVE_PATH_BASE"))

    if df_acc is None and df is None:
        # Skip building the base df if either DataFrames already exist
        prompt_file_set = PromptFileSet(path_2k=os.getenv("PATH_2K"), path_4k=os.getenv("PATH_4K"), variant=args.variant, limit=args.limit)
        prompt_repository = PromptRepository(file_set=prompt_file_set)
        prompt_records = prompt_repository.load_all()
        prompts = [record.prompt for record in prompt_records]

        df_builder = PromptDataFrameBuilder(prompts=prompt_records)
        df = df_builder.build()

        df_builder.persist(path=os.getenv("SAVE_PATH_BASE"))


    if df_acc is None:
        logger.info(
            "Getting baseline response data for variant '%s' with limit '%s'",
            args.variant if args.variant else "all",
            str(args.limit) if args.limit else "all"
        )
        # return
        # Initialize api client for populating baseline response data
        firework_client = FireworksAICompletionClient(
            model_name=os.getenv("MODEL_NAME"),
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )
        baseline_inference = LocalInferenceRunner(
            client=firework_client,
            dataframe=df,
        )
        df_acc = baseline_inference.run(
            variant=args.variant,
            limit=args.limit,
        )
        baseline_inference.persist(path=os.getenv("SAVE_PATH_ACC"))

    

    # return

    logger.info(
        "Starting inference (%s mode) with batch_size=%d",
        "local-debug" if args.local_debug else "cluster",
        args.batch_size,
    )

    if not prompts and df_acc is not None:
        prompts = df_acc["prompt"].tolist()
    if not prompts and df is not None:
        prompts = df["prompt"].tolist()

    if not prompts:
        logger.error("No prompts are available to run.")
        raise SystemExit(1)

    if args.local_debug:
        try:
            results = run_local_debug(
                prompts,
                batch_size=args.batch_size,
                tracker=tracker,
            )
        except ImportError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
    else:
        run_distributed_stub(args)
        return

    if args.output:
        logger.info("Writing outputs to %s", args.output)
        with args.output.open("w") as handle:
            for record in results:
                handle.write(json.dumps(record) + "\n")
    else:
        for record in results:
            logger.info("Result: %s", json.dumps(record))

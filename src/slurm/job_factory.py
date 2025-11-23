"""Factories that turn strategies into concrete Slurm submission scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..parallelism.strategy import ParallelismStrategy

# this should create a json file for configs, and .slurm scripts for runtime (or just pass the path of the already made scripts)

@dataclass
class SlurmConfig:
    """Serializable description of a Slurm job submission."""

    job_name: str
    nodes: int
    ntasks_per_node: int
    time_limit: str
    partition: Optional[str] = None
    constraint: Optional[str] = None
    output_path: Optional[Path] = None
    env: Optional[Dict[str, str]] = None


class SlurmJobFactory:
    """Generates Slurm configs and scripts for each strategy run."""

    def __init__(self, script_root: Path) -> None:
        """Store the root directory where rendered scripts will live."""
        self.script_root = script_root

    def create_config(self, strategy: ParallelismStrategy) -> SlurmConfig:
        """Build a `SlurmConfig` tailored to the provided strategy."""
        raise NotImplementedError("Slurm config creation is not implemented yet.")

    def render_script(self, config: SlurmConfig) -> Path:
        """Render the shell script that will be submitted via sbatch."""
        raise NotImplementedError("Slurm script rendering is not implemented yet.")

    def script_path(self, config: SlurmConfig) -> Path:
        """Return the filesystem location for the rendered script."""
        raise NotImplementedError("Script path resolution is not implemented yet.")

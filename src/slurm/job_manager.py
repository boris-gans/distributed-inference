"""Submission and monitoring utilities for Slurm-managed jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from .job_factory import SlurmConfig

# creates appainter image, sends it to cluster to run, 
#   collects results/artifcats once job has finished. 

class SlurmJobManager:
    """Handles sbatch submission, monitoring, and artifact callbacks."""

    def __init__(self, workdir: Path) -> None:
        """Store the working directory used for job output files."""
        self.workdir = workdir

    def submit(self, config: SlurmConfig) -> str:
        """Submit the rendered script to Slurm and return the job ID."""
        raise NotImplementedError("Slurm submission is not implemented yet.")

    def monitor(self, job_id: str) -> None:
        """Poll Slurm until the specified job leaves the queue."""
        raise NotImplementedError("Slurm monitoring is not implemented yet.")

    def collect_artifacts(self, job_id: str, callback: Callable[[Path], None]) -> None:
        """Invoke a callback once job artifacts arrive on the orchestration node."""
        raise NotImplementedError("Artifact collection is not implemented yet.")

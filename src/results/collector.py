"""Utilities to gather per-strategy artifacts and merge them into a table."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

class ResultCollector:
    """Collects outputs written by Slurm jobs and normalizes them."""

    def __init__(self) -> None:
        """Initialize empty buffers for run-level and prompt-level data."""
        self._records: List[dict] = []

    def ingest_run(self, artifact_dir: Path) -> None:
        """Load all artifacts from a completed strategy run."""
        raise NotImplementedError("Result ingestion is not implemented yet.")

    def merge(self) -> List[dict]:
        """Return a list of normalized prompt-level records."""
        raise NotImplementedError("Result merging is not implemented yet.")

    def to_dataframe(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Return the merged results as a pandas DataFrame."""
        raise NotImplementedError("Result DataFrame conversion is not implemented yet.")

    def iter_traces(self) -> Iterable[Path]:
        """Yield trace or profiling files produced during a run."""
        raise NotImplementedError("Trace iteration is not implemented yet.")

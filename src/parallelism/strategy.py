"""Strategy abstractions covering each parallelism technique from the README."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from ..data.prompts import PromptRepository


# TODO: load a yaml file describing cluster topology. store this within ParallelsimStratedgy, amking it available to child strats
    # child strats then dynamically update their configs based on this, with a nromalized output to use for job factory

class ParallelismStrategy(ABC):
    """Base class for any TP/PP configuration we want to evaluate."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        """Store the prompt repository used to build job input shards."""
        self.prompt_repository = prompt_repository

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return a short identifier that tags artifacts produced by this strategy."""

    @abstractmethod
    def describe_topology(self) -> str:
        """Explain the TP/PP/DP layout for documentation and logging."""

    @abstractmethod
    def slurm_constraints(self) -> Dict[str, str]:
        """Return Slurm resource hints (nodes, partition, constraint flags)."""

    @abstractmethod
    def deepspeed_args(self) -> Dict[str, str]:
        """Expose the DeepSpeed configuration overrides for this run."""

    @abstractmethod
    def expected_jobs(self) -> int:
        """Return how many sequential jobs make up this strategy."""

    @abstractmethod
    def postprocess(self, dataframe) -> None:
        """Apply any strategy-specific cleanup to the aggregated DataFrame."""


class PipelineParallelism(ParallelismStrategy):
    """Hybrid tensor-parallel within nodes plus pipeline across nodes."""

    @property
    def strategy_name(self) -> str:
        """Return the canonical identifier for the hybrid configuration."""
        return "pipeline"

    def describe_topology(self) -> str:
        """Describe the hybrid TP+PP layout used for this strategy."""
        raise NotImplementedError("Topology description is not implemented yet.")

    def slurm_constraints(self) -> Dict[str, str]:
        """Return Slurm parameters for the hybrid campaign."""
        raise NotImplementedError("Slurm constraints are not implemented yet.")

    def deepspeed_args(self) -> Dict[str, str]:
        """Return DeepSpeed overrides for the hybrid campaign."""
        raise NotImplementedError("DeepSpeed args are not implemented yet.")

    def expected_jobs(self) -> int:
        """Return how many jobs make up the hybrid campaign."""
        raise NotImplementedError("Expected jobs count is not implemented yet.")

    def postprocess(self, dataframe) -> None:
        """Attach metadata to results produced by hybrid TP+PP runs."""
        raise NotImplementedError("Postprocessing is not implemented yet.")

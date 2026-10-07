"""Experiment configuration and reproducibility metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import platform
from pathlib import Path
import subprocess
import sys

import matplotlib
import numpy
import scipy
import torch

from .reverse import GENERATOR_SETS
from ..classical.diffusion import ClassicalDiffusionConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for the single-spin experiment."""

    j: float = 2.0
    n_data: int = 2000
    sigma: float = 0.22
    diffusion_steps: int = 6
    diffusion_rate: float = 1.0
    time_step: float = 0.35
    reverse_layers: int = 3
    epochs: int = 700
    learning_rate: float = 0.04
    seed: int = 7
    objective: str = "frobenius"
    multipole_weighting: str = "rank-balanced"
    loss_grid_theta: int = 16
    loss_grid_phi: int = 32
    q_grid_theta: int = 80
    q_grid_phi: int = 160
    generator_set: str = "full"
    parameter_sharing: str = "independent"
    ancilla_count: int = 1
    starting_state: str = "mixed"
    supervision: str = "path"
    hybrid_lambda: float = 0.0
    output_directory: str = "outputs/experiments/single_spin"
    classical: ClassicalDiffusionConfig = field(default_factory=ClassicalDiffusionConfig)

    def validate(self) -> None:
        self.classical.validate()
        if self.n_data < 1:
            raise ValueError("n_data must be positive")
        if self.j < 0.5 or abs(2 * self.j - round(2 * self.j)) > 1e-12:
            raise ValueError("j must be a positive integer or half-integer")
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be at least one")
        if self.reverse_layers < 1 or self.epochs < 1:
            raise ValueError("reverse_layers and epochs must be positive")
        if self.learning_rate <= 0 or self.time_step <= 0 or self.diffusion_rate < 0:
            raise ValueError("learning_rate/time_step must be positive and D nonnegative")
        if self.ancilla_count < 1:
            raise ValueError("ancilla_count must be at least one")
        if self.generator_set not in GENERATOR_SETS:
            raise ValueError(f"generator_set must be one of {GENERATOR_SETS}")
        if self.parameter_sharing not in {"independent", "shared"}:
            raise ValueError("parameter_sharing must be independent or shared")
        if self.starting_state not in {"mixed", "forward"}:
            raise ValueError("starting_state must be mixed or forward")
        if self.supervision not in {"path", "final"}:
            raise ValueError("supervision must be path or final")
        if not 0 <= self.hybrid_lambda <= 1:
            raise ValueError("hybrid_lambda must lie in [0, 1]")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def reproducibility_metadata(
    config: ExperimentConfig,
    repository: Path,
) -> dict:
    """Record software, source-control, platform, and determinism context."""
    status = _git_value(repository, "status", "--porcelain")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "seed": config.seed,
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "platform": platform.platform(),
        "git": {
            "commit": _git_value(repository, "rev-parse", "HEAD"),
            "branch": _git_value(repository, "branch", "--show-current"),
            "dirty": None if status is None else bool(status),
        },
    }

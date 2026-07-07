"""Checkpoint file conventions shared by training (writes) and tournaments (read).

Deliberately torch-free: this module deals only in paths and iteration numbers.
Saving and loading tensors stays with the callers, so anything that consumes
checkpoint files (e.g. a tournament script) can discover them without the
training stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Zero-padded so lexical and numeric order agree in directory listings.
_CHECKPOINT_FORMAT = "checkpoint-{iteration:05d}.pt"
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)\.pt$")


@dataclass(frozen=True)
class Checkpoint:
    """A discovered checkpoint file and the training iteration it was saved at."""

    iteration: int
    path: Path


def checkpoint_path(directory: Path, iteration: int) -> Path:
    """The canonical path for a checkpoint saved at the given iteration."""
    if iteration < 0:
        raise ValueError(f"iteration must be >= 0, got {iteration}")
    return directory / _CHECKPOINT_FORMAT.format(iteration=iteration)


def discover_checkpoints(directory: Path) -> list[Checkpoint]:
    """Checkpoints in the directory, in iteration order.

    Ignores files that don't match the checkpoint naming convention (e.g. the
    final model.pt). Iterations are parsed numerically, so files written with
    a different zero-padding width still order correctly. A missing directory
    yields an empty list — same as a directory with no checkpoints.
    """
    if not directory.is_dir():
        return []
    checkpoints = [
        Checkpoint(iteration=int(match.group(1)), path=path)
        for path in directory.iterdir()
        if (match := _CHECKPOINT_PATTERN.match(path.name))
    ]
    checkpoints.sort(key=lambda checkpoint: checkpoint.iteration)
    return checkpoints

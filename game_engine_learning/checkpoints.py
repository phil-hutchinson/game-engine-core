"""Checkpoint file conventions shared by training (writes) and tournaments (read).

Each training run gets its own timestamped run directory, and checkpoints are
named by iteration within it — so consumers can resolve "the latest run" and
"the checkpoints of a run" without the runs ever mixing.

Deliberately torch-free: this module deals only in paths and iteration numbers.
Saving and loading tensors stays with the callers, so anything that consumes
checkpoint files (e.g. a tournament script) can discover them without the
training stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

# Zero-padded so lexical and numeric order agree in directory listings.
_CHECKPOINT_FORMAT = "checkpoint-{iteration:05d}.pt"
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)\.pt$")

# Top-down date format so lexical order is chronological order.
_RUN_DIR_FORMAT = "%Y%m%d-%H%M%S"
_RUN_DIR_PATTERN = re.compile(r"^\d{8}-\d{6}$")


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

    Two files parsing to the same iteration (possible only with hand-placed
    files, since a run directory is written by one process with one padding
    width) raise here, where the ambiguity originates, rather than surfacing
    downstream as a confusing duplicate-player error.
    """
    if not directory.is_dir():
        return []
    checkpoints = [
        Checkpoint(iteration=int(match.group(1)), path=path)
        for path in directory.iterdir()
        if (match := _CHECKPOINT_PATTERN.match(path.name))
    ]
    checkpoints.sort(key=lambda checkpoint: checkpoint.iteration)
    for previous, current in pairwise(checkpoints):
        if previous.iteration == current.iteration:
            raise ValueError(
                f"Duplicate checkpoint iteration {current.iteration} in {directory}: "
                f"{previous.path.name} and {current.path.name}"
            )
    return checkpoints


def new_run_directory(base_dir: Path) -> Path:
    """Create and return a timestamped run directory under base_dir.

    One run directory per training run keeps each run's checkpoints isolated:
    a re-train never mixes its checkpoints with a previous run's.
    """
    run_dir = base_dir / datetime.now().strftime(_RUN_DIR_FORMAT)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_directory(base_dir: Path) -> Path | None:
    """The most recent run directory under base_dir, or None if there is none.

    Run names are top-down dates, so the latest is simply the lexical maximum.
    Entries that don't look like run directories are ignored.
    """
    if not base_dir.is_dir():
        return None
    runs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and _RUN_DIR_PATTERN.match(path.name)
    ]
    return max(runs, key=lambda path: path.name, default=None)

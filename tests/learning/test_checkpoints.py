"""Checkpoint naming/discovery tests: pure path logic over a tmp directory."""

from pathlib import Path

import pytest

from game_engine_learning.checkpoints import (
    Checkpoint,
    checkpoint_path,
    discover_checkpoints,
)


def test_paths_are_zero_padded(tmp_path: Path) -> None:
    assert checkpoint_path(tmp_path, 5).name == "checkpoint-00005.pt"
    assert checkpoint_path(tmp_path, 12345).name == "checkpoint-12345.pt"


def test_rejects_negative_iterations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iteration"):
        checkpoint_path(tmp_path, -1)


def test_discovery_round_trips_named_paths(tmp_path: Path) -> None:
    for iteration in [10, 2, 6]:
        checkpoint_path(tmp_path, iteration).touch()
    assert discover_checkpoints(tmp_path) == [
        Checkpoint(iteration=2, path=checkpoint_path(tmp_path, 2)),
        Checkpoint(iteration=6, path=checkpoint_path(tmp_path, 6)),
        Checkpoint(iteration=10, path=checkpoint_path(tmp_path, 10)),
    ]


def test_discovery_orders_mixed_padding_widths_numerically(tmp_path: Path) -> None:
    # checkpoint-5.pt sorts after checkpoint-00010.pt lexically; discovery
    # must parse the iteration instead of trusting filename order.
    (tmp_path / "checkpoint-5.pt").touch()
    (tmp_path / "checkpoint-00010.pt").touch()
    assert [c.iteration for c in discover_checkpoints(tmp_path)] == [5, 10]


def test_discovery_ignores_non_checkpoint_files(tmp_path: Path) -> None:
    checkpoint_path(tmp_path, 3).touch()
    for name in ["model.pt", "checkpoint-abc.pt", "checkpoint-3.txt", "notes.md"]:
        (tmp_path / name).touch()
    assert [c.iteration for c in discover_checkpoints(tmp_path)] == [3]


def test_discovery_of_missing_directory_is_empty(tmp_path: Path) -> None:
    assert discover_checkpoints(tmp_path / "does-not-exist") == []

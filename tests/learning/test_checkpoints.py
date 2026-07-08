"""Checkpoint naming/discovery tests: pure path logic over a tmp directory."""

import re
from pathlib import Path

import pytest

from game_engine_learning.checkpoints import (
    Checkpoint,
    checkpoint_path,
    discover_checkpoints,
    latest_run_directory,
    new_run_directory,
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


def test_discovery_rejects_duplicate_iterations(tmp_path: Path) -> None:
    # Only reachable with hand-placed files: one run writes one padding width.
    (tmp_path / "checkpoint-5.pt").touch()
    (tmp_path / "checkpoint-00005.pt").touch()
    with pytest.raises(ValueError, match="Duplicate checkpoint iteration 5"):
        discover_checkpoints(tmp_path)


def test_discovery_ignores_non_checkpoint_files(tmp_path: Path) -> None:
    checkpoint_path(tmp_path, 3).touch()
    for name in ["model.pt", "checkpoint-abc.pt", "checkpoint-3.txt", "notes.md"]:
        (tmp_path / name).touch()
    assert [c.iteration for c in discover_checkpoints(tmp_path)] == [3]


def test_discovery_of_missing_directory_is_empty(tmp_path: Path) -> None:
    assert discover_checkpoints(tmp_path / "does-not-exist") == []


def test_new_run_directory_is_created_and_timestamped(tmp_path: Path) -> None:
    run_dir = new_run_directory(tmp_path / "runs")
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path / "runs"
    assert re.fullmatch(r"\d{8}-\d{6}", run_dir.name)


def test_latest_run_directory_round_trips_new_run(tmp_path: Path) -> None:
    run_dir = new_run_directory(tmp_path / "runs")
    assert latest_run_directory(tmp_path / "runs") == run_dir


def test_latest_run_directory_picks_lexical_maximum(tmp_path: Path) -> None:
    for name in ["20260102-000000", "20260107-235959", "20260107-120000"]:
        (tmp_path / name).mkdir()
    latest = latest_run_directory(tmp_path)
    assert latest is not None
    assert latest.name == "20260107-235959"


def test_latest_run_directory_ignores_non_run_entries(tmp_path: Path) -> None:
    (tmp_path / "20260102-000000").mkdir()
    # Lexically "zzz" and "notes.txt" sort above the timestamp; neither is a run.
    (tmp_path / "zzz-stray-directory").mkdir()
    (tmp_path / "notes.txt").touch()
    latest = latest_run_directory(tmp_path)
    assert latest is not None
    assert latest.name == "20260102-000000"


def test_latest_run_directory_of_missing_or_empty_base_is_none(tmp_path: Path) -> None:
    assert latest_run_directory(tmp_path / "does-not-exist") is None
    assert latest_run_directory(tmp_path) is None

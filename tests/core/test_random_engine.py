"""RandomEngine tests: legality of selections and terminal-position rejection."""

import pytest

from game_engine_core.engines.random_engine import RandomEngine

from .nim_fixture import NimPly, NimPosition


def test_selects_only_legal_plies() -> None:
    engine: RandomEngine[NimPly, NimPosition] = RandomEngine()
    position = NimPosition(pile=2)
    permitted = {1, 2}
    for _ in range(50):
        assert engine.select_ply(position).take in permitted


def test_single_legal_ply_is_selected() -> None:
    engine: RandomEngine[NimPly, NimPosition] = RandomEngine()
    assert engine.select_ply(NimPosition(pile=1)).take == 1


def test_terminal_position_raises() -> None:
    engine: RandomEngine[NimPly, NimPosition] = RandomEngine()
    with pytest.raises(ValueError):
        engine.select_ply(NimPosition(pile=0))

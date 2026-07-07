"""Pins down the fixture game's own contract before other tests build on it."""

import pytest

from .nim_fixture import NimPly, NimPosition


def test_nonempty_pile_is_ongoing() -> None:
    position = NimPosition(pile=5)
    assert position.outcome is None
    assert position.active_player_id == 1


def test_legal_plies_respect_pile_and_takes() -> None:
    assert [ply.take for ply in NimPosition(pile=5).legal_plies] == [1, 2]
    # A take larger than the remaining pile is not offered.
    assert [ply.take for ply in NimPosition(pile=1).legal_plies] == [1]


def test_apply_ply_decrements_pile_and_alternates_player() -> None:
    position = NimPosition(pile=5)
    after = position.apply_ply(NimPly(2))
    assert after.pile == 3
    assert after.active_player_id == -1
    assert after.apply_ply(NimPly(1)).active_player_id == 1


def test_apply_ply_leaves_source_position_unchanged() -> None:
    position = NimPosition(pile=5)
    position.apply_ply(NimPly(2))
    assert position.pile == 5
    assert position.active_player_id == 1


def test_taking_last_token_wins() -> None:
    # Pile empty: the player to move lost, so the outcome is -1
    # (current-player-relative, per the GamePosition protocol).
    terminal = NimPosition(pile=2).apply_ply(NimPly(2))
    assert terminal.pile == 0
    assert terminal.outcome == -1
    assert terminal.legal_plies == []


def test_illegal_take_raises() -> None:
    with pytest.raises(ValueError):
        NimPosition(pile=1).apply_ply(NimPly(2))  # larger than the pile
    with pytest.raises(ValueError):
        NimPosition(pile=5).apply_ply(NimPly(3))  # not a permitted take size


def test_forced_line_mode_has_single_legal_ply() -> None:
    position = NimPosition(pile=3, takes=(1,))
    while position.outcome is None:
        plies = position.legal_plies
        assert [ply.take for ply in plies] == [1]
        position = position.apply_ply(plies[0])
    assert position.pile == 0

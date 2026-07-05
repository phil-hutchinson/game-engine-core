"""TicTacToePly tests: input validation and the str(ply) identity key."""

import pytest

from examples.tictactoe.tictactoe_ply import TicTacToePly


def test_all_board_squares_are_valid() -> None:
    for square in range(1, 10):
        ply = TicTacToePly(square)
        assert ply.square == square
        assert str(ply) == str(square)


def test_out_of_range_squares_are_rejected() -> None:
    for square in (0, 10, -1):
        with pytest.raises(ValueError):
            TicTacToePly(square)


def test_str_is_unique_across_a_position_s_legal_plies() -> None:
    # str(ply) is the framework's identity key (see the GamePly protocol).
    representations = [str(TicTacToePly(square)) for square in range(1, 10)]
    assert len(set(representations)) == 9

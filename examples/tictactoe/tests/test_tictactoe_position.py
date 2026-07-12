"""TicTacToePosition tests: legality, outcome detection, and the relative-sign convention.

The GamePosition protocol states outcomes current-player-relative: after a winning
ply the active player flips to the loser, so a just-won game reads as -1.
"""

from typing import cast

import pytest

from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import Board, TicTacToePosition


def _play(squares: list[int]) -> TicTacToePosition:
    """Apply plies alternately (player 1 first) from a fresh board."""
    position = TicTacToePosition.new_game()
    for square in squares:
        position = position.apply_ply(TicTacToePly(square))
    return position


def test_new_game_is_empty_and_player_one_moves() -> None:
    position = TicTacToePosition.new_game()
    assert position.active_player_id == 1
    assert position.outcome is None
    assert [str(ply) for ply in position.legal_plies] == [str(n) for n in range(1, 10)]


def test_apply_ply_marks_square_and_alternates_player() -> None:
    position = _play([5])
    assert position.board[4] == 1
    assert position.active_player_id == -1
    assert [ply.square for ply in position.legal_plies] == [1, 2, 3, 4, 6, 7, 8, 9]


def test_apply_ply_leaves_source_position_unchanged() -> None:
    position = TicTacToePosition.new_game()
    position.apply_ply(TicTacToePly(5))
    assert position.board == (0,) * 9
    assert position.active_player_id == 1


def test_occupied_square_is_rejected() -> None:
    position = _play([5])
    with pytest.raises(ValueError):
        position.apply_ply(TicTacToePly(5))


def test_row_win_reads_minus_one_for_the_player_facing_it() -> None:
    # P1 takes the top row (1, 2, 3); P2 is then the active player and has lost.
    position = _play([1, 4, 2, 5, 3])
    assert position.active_player_id == -1
    assert position.outcome == -1


def test_column_win_by_player_two() -> None:
    # P2 takes the left column (1, 4, 7); P1 is then active and has lost.
    position = _play([2, 1, 3, 4, 5, 7])
    assert position.active_player_id == 1
    assert position.outcome == -1


def test_diagonal_win() -> None:
    # P1 takes the main diagonal (1, 5, 9).
    position = _play([1, 2, 5, 3, 9])
    assert position.outcome == -1


def test_win_reads_plus_one_when_the_active_player_holds_the_line() -> None:
    # Constructed directly: the protocol also covers positions where the active
    # player already has a winning line (outcome +1 from their perspective).
    board = cast("Board", (1, 1, 1, -1, -1, 0, 0, 0, 0))
    position = TicTacToePosition(board=board, active_player_id=1)
    assert position.outcome == 1


def test_ongoing_game_has_no_outcome() -> None:
    position = _play([1, 4, 2, 5])
    assert position.outcome is None
    assert position.outcome_reason is None


def test_win_reason_is_three_in_a_row() -> None:
    position = _play([1, 4, 2, 5, 3])
    assert position.outcome_reason == "Three in a row"


def test_draw_reason_is_board_full() -> None:
    position = _play([1, 3, 2, 4, 6, 5, 7, 8, 9])
    assert position.outcome == 0
    assert position.outcome_reason == "Board full"


def test_full_board_without_a_line_is_a_draw() -> None:
    # P1 ends with squares {1, 2, 6, 7, 9}, P2 with {3, 4, 5, 8}: no line for
    # either player at any point, and a draw (0) once the board fills.
    squares = [1, 3, 2, 4, 6, 5, 7, 8, 9]
    position = TicTacToePosition.new_game()
    for square in squares:
        assert position.outcome is None
        position = position.apply_ply(TicTacToePly(square))
    assert position.outcome == 0
    assert position.legal_plies == []

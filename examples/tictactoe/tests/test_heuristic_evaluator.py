"""TicTacToeHeuristicEvaluator sanity tests: value bounds and policy shape/ordering."""

import pytest

from examples.tictactoe.tictactoe_heuristic_evaluator import TicTacToeHeuristicEvaluator
from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import TicTacToePosition


def _position_with_win_and_block() -> TicTacToePosition:
    """P1 to move with 1, 2 on the top row; P2 holds 4, 5 on the middle row.

    Square 3 wins immediately; square 6 blocks P2's reply threat.
    """
    position = TicTacToePosition.new_game()
    for square in (1, 4, 2, 5):
        position = position.apply_ply(TicTacToePly(square))
    return position


def test_policy_is_a_distribution_over_the_legal_plies() -> None:
    evaluation = TicTacToeHeuristicEvaluator().evaluate_position(_position_with_win_and_block())

    assert evaluation.policy is not None
    assert set(evaluation.policy) == {"3", "6", "7", "8", "9"}
    assert all(probability > 0 for probability in evaluation.policy.values())
    assert sum(evaluation.policy.values()) == pytest.approx(1.0)


def test_winning_move_gets_the_most_policy_mass() -> None:
    evaluation = TicTacToeHeuristicEvaluator().evaluate_position(_position_with_win_and_block())

    policy = evaluation.policy
    assert policy is not None
    assert max(policy, key=lambda key: policy[key]) == "3"
    # The block of the opponent's threat ranks second.
    remaining = {key: p for key, p in policy.items() if key != "3"}
    assert max(remaining, key=lambda key: remaining[key]) == "6"


def test_value_stays_in_bounds() -> None:
    evaluation = TicTacToeHeuristicEvaluator().evaluate_position(_position_with_win_and_block())
    assert -1.0 <= evaluation.value <= 1.0

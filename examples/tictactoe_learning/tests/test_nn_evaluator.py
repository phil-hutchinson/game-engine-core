"""TicTacToeNNEvaluator tests: perspective encoding and legal-ply policy masking.

The evaluator carries the example's densest sign convention: encode_position
multiplies every cell by the active player's id so the model always sees "my
pieces as +1", and decode_policy masks illegal squares out of the distribution.
A regression in either silently corrupts training data, so both are pinned here.
"""

from typing import cast

import pytest
import torch

from examples.tictactoe.tictactoe_position import Board, TicTacToePosition
from examples.tictactoe_learning.tictactoe_mlp import TicTacToeMLP
from examples.tictactoe_learning.tictactoe_nn_evaluator import TicTacToeNNEvaluator


def _evaluator() -> TicTacToeNNEvaluator:
    # encode_position/decode_policy never touch the model, so any model instance
    # serves; these tests deliberately exercise only the sign-convention logic.
    return TicTacToeNNEvaluator(model=TicTacToeMLP())


def _mid_game_board() -> Board:
    # 1s, -1s and empties so every branch of the encoding is exercised.
    #  1 | -1 |  .
    #  . |  1 |  .
    #  . |  . | -1
    return cast('Board', (1, -1, 0, 0, 1, 0, 0, 0, -1))


def test_encoding_negates_between_the_two_perspectives() -> None:
    # Same board, opposite player to move: each player must see the identical
    # position with the signs flipped ("my pieces as +1").
    evaluator = _evaluator()
    board = _mid_game_board()

    as_player_one = evaluator.encode_position(TicTacToePosition(board, active_player_id=1))
    as_player_two = evaluator.encode_position(TicTacToePosition(board, active_player_id=-1))

    assert torch.equal(as_player_two, -as_player_one)


def test_encoding_maps_occupied_to_plus_minus_one_and_empty_to_zero() -> None:
    # From player 1's perspective the raw board values pass through unchanged:
    # own pieces +1, opponent -1, empty 0.
    evaluator = _evaluator()
    encoded = evaluator.encode_position(TicTacToePosition(_mid_game_board(), active_player_id=1))

    assert encoded.shape == (9,)
    assert encoded.tolist() == [1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]


def test_decode_policy_is_a_distribution_over_exactly_the_legal_plies() -> None:
    # The mask must confine all probability mass to legal squares: the returned
    # dict is keyed by legal plies only and sums to 1, so no mass leaks onto the
    # occupied squares (which would leave the legal total below 1).
    evaluator = _evaluator()
    position = TicTacToePosition(_mid_game_board(), active_player_id=1)
    legal_plies = position.legal_plies

    policy = evaluator.decode_policy(torch.zeros(9), legal_plies)

    assert set(policy) == {str(ply) for ply in legal_plies}
    assert all(probability > 0 for probability in policy.values())
    assert sum(policy.values()) == pytest.approx(1.0)


def test_decode_policy_reflects_the_logit_ordering() -> None:
    # The highest logit among legal squares must receive the most mass.
    evaluator = _evaluator()
    position = TicTacToePosition(_mid_game_board(), active_player_id=1)

    logits = torch.zeros(9)
    logits[2] = 5.0  # square 3, a legal empty square

    policy = evaluator.decode_policy(logits, position.legal_plies)

    assert max(policy, key=lambda key: policy[key]) == "3"


def test_decode_policy_ignores_logits_on_illegal_squares() -> None:
    # A large logit on an occupied square must not appear in or distort the
    # distribution over the legal plies.
    evaluator = _evaluator()
    position = TicTacToePosition(_mid_game_board(), active_player_id=1)
    legal_plies = position.legal_plies

    baseline = evaluator.decode_policy(torch.zeros(9), legal_plies)
    spiked = torch.zeros(9)
    spiked[0] = 100.0  # square 1 is occupied and must be masked out
    with_illegal_spike = evaluator.decode_policy(spiked, legal_plies)

    assert "1" not in with_illegal_spike
    for ply in legal_plies:
        assert with_illegal_spike[str(ply)] == pytest.approx(baseline[str(ply)])

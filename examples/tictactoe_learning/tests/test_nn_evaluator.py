"""TicTacToeNNEvaluator tests: perspective encoding and legal-ply policy masking.

The evaluator carries the example's densest sign convention: encode_positions
multiplies every row by its position's active player id so the model always
sees "my pieces as +1", and decode_policies masks illegal squares out of each
row's distribution. A regression in either silently corrupts training data, so
both are pinned here.
"""

from collections.abc import Sequence
from typing import cast

import pytest
import torch
from torch import Tensor

from examples.tictactoe.tictactoe_position import Board, TicTacToePosition
from examples.tictactoe_learning.tictactoe_mlp import TicTacToeMLP
from examples.tictactoe_learning.tictactoe_nn_evaluator import TicTacToeNNEvaluator


def _evaluator() -> TicTacToeNNEvaluator:
    # encode_positions/decode_policies never touch the model, so any model instance
    # serves; these tests deliberately exercise only the sign-convention logic.
    return TicTacToeNNEvaluator(model=TicTacToeMLP())


class _RecordingEvaluator(TicTacToeNNEvaluator):
    """Records the positions evaluate_positions hands to decode_policies.

    Delegates to the real decode_policies so the recorded object's behaviour
    (not just its identity) still gets exercised.
    """

    def __init__(self, model: TicTacToeMLP):
        super().__init__(model=model)
        self.received_positions: Sequence[TicTacToePosition] | None = None

    def decode_policies(
        self, policy_logits: Tensor, positions: Sequence[TicTacToePosition]
    ) -> Sequence[dict[str, float]]:
        self.received_positions = positions
        return super().decode_policies(policy_logits, positions)


def _mid_game_board() -> Board:
    # 1s, -1s and empties so every branch of the encoding is exercised.
    #  1 | -1 |  .
    #  . |  1 |  .
    #  . |  . | -1
    return cast('Board', (1, -1, 0, 0, 1, 0, 0, 0, -1))


def _one_square_left_board() -> Board:
    # Only square 9 (index 8) is empty, and no line is accidentally complete —
    # a different legal set than _mid_game_board's, for the alignment test.
    return cast('Board', (1, -1, 1, -1, -1, 1, -1, 1, 0))


def test_encoding_negates_between_the_two_perspectives() -> None:
    # Same board, opposite player to move: each player must see the identical
    # position with the signs flipped ("my pieces as +1").
    evaluator = _evaluator()
    board = _mid_game_board()

    encoded = evaluator.encode_positions([
        TicTacToePosition(board, active_player_id=1),
        TicTacToePosition(board, active_player_id=-1),
    ])

    assert torch.equal(encoded[1], -encoded[0])


def test_encoding_maps_occupied_to_plus_minus_one_and_empty_to_zero() -> None:
    # From player 1's perspective the raw board values pass through unchanged:
    # own pieces +1, opponent -1, empty 0.
    evaluator = _evaluator()
    encoded = evaluator.encode_positions([TicTacToePosition(_mid_game_board(), active_player_id=1)])

    assert encoded.shape == (1, 9)
    assert encoded[0].tolist() == [1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]


def test_decode_policies_is_a_distribution_over_exactly_the_legal_plies() -> None:
    # The mask must confine all probability mass to legal squares: the returned
    # dict is keyed by legal plies only and sums to 1, so no mass leaks onto the
    # occupied squares (which would leave the legal total below 1).
    evaluator = _evaluator()
    position = TicTacToePosition(_mid_game_board(), active_player_id=1)
    legal_plies = position.legal_plies

    policy = evaluator.decode_policies(torch.zeros(1, 9), [position])[0]

    assert set(policy) == {str(ply) for ply in legal_plies}
    assert all(probability > 0 for probability in policy.values())
    assert sum(policy.values()) == pytest.approx(1.0)


def test_decode_policies_reflects_the_logit_ordering() -> None:
    # The highest logit among legal squares must receive the most mass.
    evaluator = _evaluator()
    position = TicTacToePosition(_mid_game_board(), active_player_id=1)

    logits = torch.zeros(1, 9)
    logits[0, 2] = 5.0  # square 3, a legal empty square

    policy = evaluator.decode_policies(logits, [position])[0]

    assert max(policy, key=lambda key: policy[key]) == "3"


def test_decode_policies_ignores_logits_on_illegal_squares() -> None:
    # A large logit on an occupied square must not appear in or distort the
    # distribution over the legal plies.
    evaluator = _evaluator()
    position = TicTacToePosition(_mid_game_board(), active_player_id=1)
    legal_plies = position.legal_plies

    baseline = evaluator.decode_policies(torch.zeros(1, 9), [position])[0]
    spiked = torch.zeros(1, 9)
    spiked[0, 0] = 100.0  # square 1 is occupied and must be masked out
    with_illegal_spike = evaluator.decode_policies(spiked, [position])[0]

    assert "1" not in with_illegal_spike
    for ply in legal_plies:
        assert with_illegal_spike[str(ply)] == pytest.approx(baseline[str(ply)])


def test_decode_policies_pairs_each_row_with_its_own_position() -> None:
    # Two positions with different legal sets: a row/position mispairing would
    # surface as probability mass landing on a square illegal for whichever
    # position it actually got paired with.
    evaluator = _evaluator()
    position_a = TicTacToePosition(_mid_game_board(), active_player_id=1)
    position_b = TicTacToePosition(_one_square_left_board(), active_player_id=1)

    policies = evaluator.decode_policies(torch.zeros(2, 9), [position_a, position_b])

    assert set(policies[0]) == {str(ply) for ply in position_a.legal_plies}
    assert set(policies[1]) == {str(ply) for ply in position_b.legal_plies}
    assert policies[1] == {"9": pytest.approx(1.0)}


def test_evaluate_positions_passes_the_positions_themselves_to_decode_policies() -> None:
    # decode_policies must receive the positions, not merely their legal plies:
    # read active_player_id, a property that only exists on the position, to
    # confirm the real objects reach decode_policies through evaluate_positions's
    # internal plumbing (not a stand-in that happens to also support iteration).
    evaluator = _RecordingEvaluator(model=TicTacToeMLP())
    position = TicTacToePosition(_mid_game_board(), active_player_id=-1)

    evaluator.evaluate_positions([position])

    assert evaluator.received_positions is not None
    assert evaluator.received_positions[0].active_player_id == -1

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import TicTacToePosition
from game_engine_learning.neural_network_evaluator import NeuralNetworkEvaluator


class TicTacToeNNEvaluator(NeuralNetworkEvaluator[TicTacToePly, TicTacToePosition]):

    def encode_position(self, position: TicTacToePosition) -> Tensor:
        # Encode the board as a 1-D float tensor of shape (9,), one value per square.
        # Each cell is multiplied by the active player's id (1 or -1) so that the
        # model always sees its own pieces as +1 and the opponent's as -1,
        # regardless of which player is moving.
        active = position.active_player_id
        return torch.tensor(
            [active * cell for cell in position.board],
            dtype=torch.float32,
        )

    def decode_policy(self, policy_logits: Tensor, legal_plies: Sequence[TicTacToePly]) -> dict[str, float]:
        # Step 1: Mask illegal plies
        # Build a mask tensor: 0.0 for legal squares, -inf for illegal ones.
        # Mask tensor must match the shape of policy_logits so they can be added element-wise.
        mask = torch.full((9,), float('-inf'))
        for ply in legal_plies:
            mask[ply.square - 1] = 0.0

        # Step 2: Use softmax to create a normalized tensor (such that the probabilities add to 1)
        # Adding the mask before softmax ensures illegal squares receive exactly
        # zero probability (-inf + any logit → -inf → exp(-inf) = 0).
        probs = F.softmax(policy_logits + mask, dim=-1)

        # Step 3: Map the probabilities back to valid plies
        return {str(ply): probs[ply.square - 1].item() for ply in legal_plies}

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from examples.tictactoe.tictactoe_position import TicTacToePosition
from game_engine_learning.neural_network_evaluator import NeuralNetworkEvaluator


class TicTacToeNNEvaluator(NeuralNetworkEvaluator[TicTacToePosition]):

    def encode_positions(self, positions: Sequence[TicTacToePosition]) -> Tensor:
        # Encode every board as a row of a (N, 9) float tensor, one value per
        # square. Each row is multiplied by its position's active player id
        # (1 or -1) so the model always sees its own pieces as +1 and the
        # opponent's as -1, regardless of which player is moving — a genuine
        # batch computation, not a per-position loop.
        boards = torch.tensor([position.board for position in positions], dtype=torch.float32)
        actives = torch.tensor(
            [position.active_player_id for position in positions], dtype=torch.float32
        ).unsqueeze(1)
        return boards * actives

    def decode_policies(
        self, policy_logits: Tensor, positions: Sequence[TicTacToePosition]
    ) -> Sequence[dict[str, float]]:
        policies: list[dict[str, float]] = []
        for row_logits, position in zip(policy_logits, positions, strict=True):
            legal_plies = position.legal_plies

            # Step 1: Mask illegal plies
            # Build a mask tensor: 0.0 for legal squares, -inf for illegal ones.
            # Mask tensor must match the shape of policy_logits so they can be added element-wise.
            mask = torch.full((9,), float('-inf'))
            for ply in legal_plies:
                mask[ply.square - 1] = 0.0

            # Step 2: Use softmax to create a normalized tensor (such that the probabilities add to 1)
            # Adding the mask before softmax ensures illegal squares receive exactly
            # zero probability (-inf + any logit → -inf → exp(-inf) = 0).
            probs = F.softmax(row_logits + mask, dim=-1)

            # Step 3: Map the probabilities back to valid plies
            policies.append({str(ply): probs[ply.square - 1].item() for ply in legal_plies})
        return policies

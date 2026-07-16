"""Minimal torch model and NeuralNetworkEvaluator over the Nim fixture game.

Just enough network to exercise the learning package's plumbing: a shared trunk
with a value head and a 2-logit policy head (one column per take size, str(ply)
"1" -> column 0, "2" -> column 1 — the same convention as the TicTacToe example).
Prediction quality is irrelevant to the tests; only shapes and contracts matter.
"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from game_engine_learning.neural_network_evaluator import NeuralNetworkEvaluator
from tests.core.nim_fixture import NimPosition


class NimMLP(nn.Module):
    """Input: 1 float (pile size). Output: (value: Tensor[1], policy_logits: Tensor[2]).

    The optional dropout layer exists so tests can make train-mode inference
    observably nondeterministic (see the eval-mode enforcement test).
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self._backbone = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._value_head = nn.Linear(8, 1)
        self._policy_head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._backbone(x)
        value = torch.tanh(self._value_head(features))
        policy_logits = self._policy_head(features)
        return value, policy_logits


class NimNNEvaluator(NeuralNetworkEvaluator[NimPosition]):

    def encode_position(self, position: NimPosition) -> Tensor:
        return torch.tensor([float(position.pile)], dtype=torch.float32)

    def decode_policy(
        self, policy_logits: Tensor, position: NimPosition
    ) -> dict[str, float]:
        legal_plies = position.legal_plies

        # Mask to the legal columns, then softmax so probabilities sum to 1.
        legal_logits = policy_logits[[ply.take - 1 for ply in legal_plies]]
        probs = torch.softmax(legal_logits, dim=-1)
        return {
            str(ply): float(prob)
            for ply, prob in zip(legal_plies, probs, strict=True)
        }


def nim_policy_loss(
    policy_logits: Tensor, target_policies: Sequence[Mapping[str, float]]
) -> Tensor:
    """Mean-reduced cross-entropy against the visit distributions (cf. the
    TicTacToe example's tictactoe_policy_loss; mean reduction is the TrainingLoop
    contract for epoch-loss reporting)."""
    targets = torch.zeros((len(target_policies), 2), dtype=torch.float32)
    for row, policy in enumerate(target_policies):
        for ply_str, prob in policy.items():
            targets[row, int(ply_str) - 1] = prob
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()

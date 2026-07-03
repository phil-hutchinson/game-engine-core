from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TrainingSample:
    encoded_position: Tensor
    """Encoded board state — output of NeuralNetworkEvaluator.encode_position."""

    target_value: float
    """Game outcome from the active player's perspective: 1.0 win, 0.0 draw, -1.0 loss.
    Used to train the value head to predict position quality."""

    target_policy: dict[str, float]
    """MCTS visit distribution over all legal plies at this position: str(ply) -> probability.
    Used to train the policy head to predict which plies are most promising."""

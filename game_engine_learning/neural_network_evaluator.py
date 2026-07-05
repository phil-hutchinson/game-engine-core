from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from game_engine_core.models.position_evaluation import PositionEvaluation
from game_engine_core.protocols.game_ply import GamePly
from game_engine_core.protocols.game_position import GamePosition


class NeuralNetworkEvaluator[TPly: GamePly, TPosition: GamePosition[Any]](ABC):
    """Abstract base class wrapping a PyTorch model as a PositionEvaluator.

    Subclasses implement encode_position and decode_policy; this class handles
    the forward pass and assembles the PositionEvaluation.

    The wrapped model's forward() must accept a batched input tensor of shape
    (batch, *sample_shape) — where sample_shape is whatever encode_position
    produces for a single position — and return (value_tensor, policy_logits),
    each with a leading batch dimension. This matches how TrainingLoop feeds the
    model; evaluate_position adds and removes the batch dimension itself, so the
    model never needs to handle unbatched input.
    """

    def __init__(self, model: nn.Module):
        self._model = model

    @abstractmethod
    def encode_position(self, position: TPosition) -> Tensor:
        """Convert a game position into a tensor suitable for model input.

        The tensor represents a single position without a batch dimension —
        1-D for an MLP, multi-dimensional for a CNN, etc. Callers add the batch
        dimension themselves (evaluate_position wraps the sample in a batch of
        size 1; TrainingLoop stacks samples into a batch). Values should be
        encoded from the
        active player's perspective so the model always reasons about "my pieces"
        vs "opponent pieces" regardless of which player is moving.

        Args:
            position: The current game position to encode.

        Returns:
            A tensor representation of the position for the model's forward pass.
        """
        ...

    @abstractmethod
    def decode_policy(self, policy_logits: Tensor, legal_plies: Sequence[TPly]) -> dict[str, float]:
        """Convert raw policy logits into a probability distribution over legal moves.

        Implementations should mask illegal moves (typically by adding -inf to their
        logits) before applying softmax, so that only legal moves receive non-zero
        probability. The resulting probabilities must sum to 1.

        Args:
            policy_logits: Raw unbounded output from the model's policy head —
                one value per possible move in the game's full action space.
            legal_plies: The moves that are actually legal in the current position.

        Returns:
            A dict mapping str(ply) to probability for each legal ply.
        """
        ...

    def evaluate_position(self, position: TPosition) -> PositionEvaluation:
        # Encode the board state into a tensor the model can process.
        encoded = self.encode_position(position)

        # Always run inference in eval mode, regardless of what state the caller
        # (e.g. TrainingLoop, which switches the shared model to train() and never
        # restores it) left the model in. Matters for BatchNorm/Dropout layers.
        self._model.eval()

        # Run the forward pass without building a gradient graph — we're doing
        # inference only, not training, so autograd tracking is unnecessary.
        # unsqueeze(0) wraps the single sample in a batch of size 1, so the model
        # sees the same (batch, …) shape here as it does in TrainingLoop.
        with torch.no_grad():
            value_tensor, policy_logits = self._model(encoded.unsqueeze(0))

        # squeeze() removes the size-1 batch and value-head dimensions (shape
        # (1, 1) → scalar tensor), then float() converts it to a plain Python float.
        value = float(value_tensor.squeeze())

        # Convert raw logits to a masked probability distribution over legal moves.
        # squeeze(0) drops the batch dimension so decode_policy receives logits for
        # a single position, matching the unbatched shape encode_position produces.
        policy = self.decode_policy(policy_logits.squeeze(0), list(position.legal_plies))

        return PositionEvaluation(value=value, policy=policy)

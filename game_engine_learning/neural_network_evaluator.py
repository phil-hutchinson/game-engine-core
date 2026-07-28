from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from game_engine_core.models.position_evaluation import PositionEvaluation
from game_engine_core.protocols.game_position import GamePosition


class NeuralNetworkEvaluator[TPosition: GamePosition[Any]](ABC):
    """Abstract base class wrapping a PyTorch model as a PositionEvaluator.

    Subclasses implement encode_positions and decode_policies; this class runs
    a single stacked forward pass and assembles the PositionEvaluations.

    The wrapped model's forward() must accept a batched input tensor of shape
    (N, *sample_shape) — where sample_shape is whatever encode_positions
    produces per position — and return (value_tensor, policy_logits), each
    with a leading batch dimension of size N. This matches how TrainingLoop
    feeds the model: evaluate_positions and TrainingLoop both hand the model
    one real batch, never a Python-level loop of batch-of-one calls.

    value_tensor must be exactly (N, 1) — one row per position, each holding
    the single scalar a value head produces. That is what an nn.Linear(*, 1)
    head emits unsqueezed, and it is the shape TrainingLoop requires: its
    targets are built as (N, 1), and a mean-squared-error loss compares
    predictions and targets elementwise, so a model returning (N,) here would
    broadcast against those targets to (N, N) and train on a silently wrong
    loss. evaluate_positions rejects any other shape rather than let that
    divergence start at inference time.
    """

    def __init__(self, model: nn.Module):
        self._model = model

    @abstractmethod
    def encode_positions(self, positions: Sequence[TPosition]) -> Tensor:
        """Convert N positions into one stacked tensor suitable for model input.

        Returns a tensor of shape (N, *sample_shape) — one row per position,
        aligned by index — where sample_shape is 1-D for an MLP,
        multi-dimensional for a CNN, etc. Values should be encoded from each
        position's active player's perspective so the model always reasons
        about "my pieces" vs "opponent pieces" regardless of which player is
        moving.

        Args:
            positions: The positions to encode.

        Returns:
            A stacked tensor representation for the model's forward pass.
        """
        ...

    @abstractmethod
    def decode_policies(
        self, policy_logits: Tensor, positions: Sequence[TPosition]
    ) -> Sequence[dict[str, float]]:
        """Convert stacked policy logits into per-position probability distributions.

        Implementations should mask illegal moves (typically by adding -inf to
        their logits) before applying softmax, so that only legal moves
        receive non-zero probability. Each returned distribution must sum to 1.

        Args:
            policy_logits: Raw unbounded output from the model's policy head,
                shape (N, *policy_shape) — row i is the logits for
                positions[i].
            positions: The positions the logits were computed for, aligned by
                index with the rows of policy_logits. Implementations that
                only need the legal moves read position.legal_plies; the full
                position is available for anything else a decoding scheme
                needs (e.g. position.active_player_id, to interpret logits
                laid out from the active player's perspective).

        Returns:
            A sequence of dicts, aligned by index with positions, each mapping
            str(ply) to probability for that position's legal plies.
        """
        ...

    def evaluate_positions(self, positions: Sequence[TPosition]) -> Sequence[PositionEvaluation]:
        # An empty batch never reaches the model: torch cannot stack an empty
        # list of encodings, and there is nothing to evaluate anyway. The fleet
        # wave hits this whenever every leaf it selected that wave is terminal,
        # so it is a routine call rather than a misuse.
        if not positions:
            return []

        # Encode all positions into one (N, *sample_shape) tensor.
        encoded = self.encode_positions(positions)

        # Always run inference in eval mode, regardless of what state the caller
        # (e.g. TrainingLoop, which switches the shared model to train() and never
        # restores it) left the model in. Matters for BatchNorm/Dropout layers.
        self._model.eval()

        # Run the forward pass without building a gradient graph — we're doing
        # inference only, not training, so autograd tracking is unnecessary.
        with torch.no_grad():
            value_tensor, policy_logits = self._model(encoded)

        # Check the value shape before unpacking it. Every wrong shape is
        # otherwise caught somewhere less legible — (N,) raises "iteration over
        # a 0-d tensor" at N == 1 but passes silently above it, a trailing
        # spatial dimension survives as a one-element tensor per row, and a
        # transposed (1, N) only trips the zip below. One check, one message.
        if value_tensor.shape != (len(positions), 1):
            raise ValueError(
                f"Model value output must have shape ({len(positions)}, 1), "
                f"got {tuple(value_tensor.shape)}"
            )

        # squeeze(-1) removes the size-1 value-head column (shape (N, 1) ->
        # (N,)), leaving the batch dimension intact even when N == 1 — unlike a
        # bare squeeze(), which would also collapse a batch of one.
        values = [float(value) for value in value_tensor.squeeze(-1)]

        # Convert raw logits to masked probability distributions, one per position.
        policies = self.decode_policies(policy_logits, positions)

        return [
            PositionEvaluation(value=value, policy=policy)
            for value, policy in zip(values, policies, strict=True)
        ]

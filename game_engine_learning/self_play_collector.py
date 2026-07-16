from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.protocols.game_ply import GamePly
from game_engine_core.protocols.game_position import GamePosition

from .neural_network_evaluator import NeuralNetworkEvaluator
from .training_sample import TrainingSample


class SelfPlayCollector[TPly: GamePly, TPosition: GamePosition[Any]]:
    """Runs self-play games and accumulates TrainingSamples for network training.

    Game-agnostic: all game-specific knowledge is encapsulated in the evaluator
    (position encoding) and the position factory (initial state).

    Args:
        evaluator: Used to encode each position into a tensor for the training sample.
        engine_factory: Called once per game to produce the MCTS engine. Allows the
            caller to control iterations, temperature, and evaluator per game.
        position_factory: Called once per game to produce the starting position.
    """

    def __init__(
        self,
        evaluator: NeuralNetworkEvaluator[TPosition],
        engine_factory: Callable[[], MCTSEngine[TPly, TPosition, Any]],
        position_factory: Callable[[], TPosition],
    ):
        self._evaluator = evaluator
        self._engine_factory = engine_factory
        self._position_factory = position_factory

    def collect(self, n_games: int) -> list[TrainingSample]:
        """Play n_games complete games and return all resulting TrainingSamples."""
        samples: list[TrainingSample] = []
        for _ in range(n_games):
            samples.extend(self._play_game())
        return samples

    def _play_game(self) -> list[TrainingSample]:
        engine = self._engine_factory()
        position = self._position_factory()

        # At each step, record the encoded position and the MCTS visit distribution.
        # Target values are not yet known — they depend on the final game outcome.
        step_records: list[tuple[Tensor, dict[str, float]]] = []

        while position.outcome is None:
            encoded = self._evaluator.encode_position(position)
            ply, policy = engine.select_ply_with_policy(position)
            step_records.append((encoded, policy))
            position = position.apply_ply(ply)

        # Assign target values by propagating the outcome backwards through the game.
        # The terminal position's outcome is from the perspective of the player who
        # would move next — i.e. the player who did NOT make the last ply. So the
        # last recorded step (taken by the other player) has value = -final_outcome,
        # and the sign alternates for each earlier step.
        final_outcome = float(position.outcome)
        samples: list[TrainingSample] = []
        value = -final_outcome
        for encoded, policy in reversed(step_records):
            samples.append(TrainingSample(
                encoded_position=encoded,
                target_value=value,
                target_policy=policy,
            ))
            value = -value

        return samples

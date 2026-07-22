from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.protocols.game_ply import GamePly
from game_engine_core.protocols.game_position import GamePosition

from .neural_network_evaluator import NeuralNetworkEvaluator
from .training_sample import TrainingSample

type PolicyTransform[TPosition: GamePosition[Any]] = Callable[
    [TPosition, dict[str, float]], dict[str, float]
]
"""(position, target_policy) -> re-keyed target_policy.

Applied at capture time to re-express an MCTS visit distribution before it is
stored on a TrainingSample. The position is passed because interpreting the
distribution can be player-dependent: in a perspective-relative action space the
policy head is laid out from the mover's frame, so mapping a global-frame
str(ply) to its logit column needs position.active_player_id. Capture is the
last point where that context is still in scope — see SelfPlayCollector.
"""


class SelfPlayCollector[TPly: GamePly, TPosition: GamePosition[Any]]:
    """Runs self-play games and accumulates TrainingSamples for network training.

    Game-agnostic: all game-specific knowledge is encapsulated in the evaluator
    (position encoding), the position factory (initial state), and the optional
    policy transform (frame-correcting the visit distribution).

    Game-specific reading of a recorded step (frame-correcting the policy,
    extracting provenance, tagging the outcome) must happen here, at capture, via
    a hook — the collector is the last place a step still has its position, and
    therefore its active_player_id, legal_plies, placement, and outcome type. A
    TrainingSample is a lossy projection that has already dropped the position, so
    any such interpretation deferred to a downstream consumer is unrecoverable.

    Args:
        evaluator: Used to encode each position into a tensor for the training sample.
        engine_factory: Called once per game to produce the MCTS engine. Allows the
            caller to control iterations, temperature, and evaluator per game.
        position_factory: Called once per game to produce the starting position.
        policy_transform: Optional hook to re-express each step's MCTS visit
            distribution while the position is still in scope. Defaults to None
            (identity): the raw str(ply) -> probability distribution is stored as
            the target_policy, unchanged. Supply a transform for a
            perspective-relative action space, where aligning a target with its
            policy-logit column depends on the mover (see PolicyTransform).
    """

    def __init__(
        self,
        evaluator: NeuralNetworkEvaluator[TPosition],
        engine_factory: Callable[[], MCTSEngine[TPly, TPosition, Any]],
        position_factory: Callable[[], TPosition],
        policy_transform: PolicyTransform[TPosition] | None = None,
    ):
        self._evaluator = evaluator
        self._engine_factory = engine_factory
        self._position_factory = position_factory
        self._policy_transform = policy_transform

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
            # Frame-correct the visit distribution while the position — and thus its
            # active_player_id — is still in scope. Without a transform the raw
            # str(ply) distribution is stored verbatim (identity).
            if self._policy_transform is not None:
                policy = self._policy_transform(position, policy)
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

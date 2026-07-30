from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.game.batch_position_processor import BatchPositionProcessor
from game_engine_core.protocols.game_ply import GamePly
from game_engine_core.protocols.game_position import GamePosition

from .neural_network_evaluator import NeuralNetworkEvaluator
from .training_sample import TrainingSample

type PolicyTransform[TPosition: GamePosition[Any]] = Callable[
    [Sequence[TPosition], Sequence[dict[str, float]]], Sequence[dict[str, float]]
]
"""(positions, target_policies) -> re-keyed target_policies, aligned by index.

Applied at capture time to re-express MCTS visit distributions before they are
stored on TrainingSamples. positions[i] pairs with target_policies[i], and the
returned sequence must keep the same alignment. The positions are passed
because interpreting a distribution can be player-dependent: in a
perspective-relative action space the policy head is laid out from the
mover's frame, so mapping a global-frame str(ply) to its logit column needs
position.active_player_id. Capture is the last point where that context is
still in scope — see SelfPlayCollector.
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

    The game loop stays sequential: one game is played to completion, one ply at
    a time, before the next starts. batch_ops is called batch-of-one throughout
    this class, matching the shape the fleet wave (multiple games at once) will
    widen without changing this class's call sites.

    Args:
        evaluator: Used to encode each position into a tensor for the training sample.
        engine_factory: Called once per collect() to produce the MCTS engine that
            searches every game. Allows the caller to control iterations, temperature
            and evaluator, but not per game: all games share the one engine, and
            therefore its iteration budget — which is what keeps them in lockstep.
            The engine must retain no per-game state on the training path. MCTSEngine
            satisfies this by construction (its fleet roots are call-scoped, and the
            retained _root_node belongs to select_ply, which is never called here);
            a subclass relying on per-game construction to clear state would not.
        position_factory: Called once per game to produce the starting position.
        policy_transform: Optional hook to re-express each step's MCTS visit
            distribution while the position is still in scope. Defaults to None
            (identity): the raw str(ply) -> probability distribution is stored as
            the target_policy, unchanged. Supply a transform for a
            perspective-relative action space, where aligning a target with its
            policy-logit column depends on the mover (see PolicyTransform).
        batch_ops: Used for the terminal test and ply application. Defaults to a
            base BatchPositionProcessor instance.
    """

    def __init__(
        self,
        evaluator: NeuralNetworkEvaluator[TPosition],
        engine_factory: Callable[[], MCTSEngine[TPly, TPosition, Any]],
        position_factory: Callable[[], TPosition],
        policy_transform: PolicyTransform[TPosition] | None = None,
        batch_ops: BatchPositionProcessor[TPly, TPosition] | None = None,
    ):
        self._evaluator = evaluator
        self._engine_factory = engine_factory
        self._position_factory = position_factory
        self._policy_transform = policy_transform
        self._batch_ops = batch_ops if batch_ops is not None else BatchPositionProcessor()

    def collect(self, n_games: int) -> list[TrainingSample]:
        """Play n_games complete games and return all resulting TrainingSamples."""
        engine = self._engine_factory()
        samples: list[TrainingSample] = []
        for _ in range(n_games):
            samples.extend(self._play_game(engine))
        return samples

    def _play_game(self, engine: MCTSEngine[TPly, TPosition, Any]) -> list[TrainingSample]:
        position = self._position_factory()

        # At each step, record the encoded position and the MCTS visit distribution.
        # Target values are not yet known — they depend on the final game outcome.
        step_records: list[tuple[Tensor, dict[str, float]]] = []

        while self._batch_ops.outcomes([position])[0] is None:
            encoded = self._evaluator.encode_positions([position])[0]
            # A fleet of one: this loop still plays games one at a time, so it drives
            # the engine's training path at width one. Turning it into a real fleet
            # driver is issue #24.
            ply, policy = engine.select_plies_for_training([position])[0]
            # Frame-correct the visit distribution while the position — and thus its
            # active_player_id — is still in scope. Without a transform the raw
            # str(ply) distribution is stored verbatim (identity).
            if self._policy_transform is not None:
                policy = self._policy_transform([position], [policy])[0]
            step_records.append((encoded, policy))
            position = self._batch_ops.apply_plies([position], [ply])[0]

        # Assign target values by propagating the outcome backwards through the game.
        # The terminal position's outcome is from the perspective of the player who
        # would move next — i.e. the player who did NOT make the last ply. So the
        # last recorded step (taken by the other player) has value = -final_outcome,
        # and the sign alternates for each earlier step.
        terminal_outcome = self._batch_ops.outcomes([position])[0]
        assert terminal_outcome is not None
        final_outcome = float(terminal_outcome)
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

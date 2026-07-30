from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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


@dataclass
class _FleetGame[TPosition: GamePosition[Any]]:
    """One game's slot in the fleet, holding everything that game alone owns.

    Identity is this object, not an index: the live set is a filter over the slots,
    so a game's position within a batch shrinks as earlier games retire while the
    game itself never moves. Its samples are filled in when it retires and read back
    in slot order once the whole fleet has drained.
    """

    position: TPosition
    step_records: list[tuple[Tensor, dict[str, float]]] = field(default_factory=lambda: [])
    samples: list[TrainingSample] = field(default_factory=lambda: [])


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
            # A fleet of one: collect still enters the fleet loop once per game, so
            # every batch below is width one. Bootstrapping all n_games into a single
            # fleet is the rest of issue #24.
            samples.extend(self._play_fleet(engine, fleet_size=1))
        return samples

    def _play_fleet(
        self, engine: MCTSEngine[TPly, TPosition, Any], fleet_size: int
    ) -> list[TrainingSample]:
        """Play a fleet of games in lockstep and return their samples in slot order.

        Each turn retires the games that have finished, then advances every game that
        remains by exactly one ply. The live set shrinks as games finish and never
        grows again, so the final turns run at low batch occupancy — the long tail the
        skateboard accepts (epic backlog P3, P4).
        """
        games = [_FleetGame[TPosition](position=self._position_factory()) for _ in range(fleet_size)]
        live = games

        while live:
            # One batched terminal test per turn serves both jobs: it decides which
            # games leave the fleet, and the outcome that retires a game is exactly
            # the one its back-fill needs. It also enforces the engine's precondition
            # below, where a terminal slot would raise and forfeit every game's search.
            outcomes = self._batch_ops.outcomes([game.position for game in live])
            still_live: list[_FleetGame[TPosition]] = []
            for game, outcome in zip(live, outcomes, strict=True):
                if outcome is None:
                    still_live.append(game)
                else:
                    game.samples = self._back_fill(game.step_records, float(outcome))
            # An order-preserving filter, so the live set stays in slot order — which
            # is what lets a batch index be read back as the game it came from.
            live = still_live
            if live:
                self._play_turn(engine, live)

        return [sample for game in games for sample in game.samples]

    def _play_turn(
        self, engine: MCTSEngine[TPly, TPosition, Any], live: Sequence[_FleetGame[TPosition]]
    ) -> None:
        """Advance every live game by one ply, one batched call per seam.

        Batch index i belongs to live[i] throughout: positions go into the engine in
        slot order and results come back aligned, so a result never needs to carry its
        game's identity.
        """
        positions = [game.position for game in live]
        results = engine.select_plies_for_training(positions)
        plies = [ply for ply, _ in results]
        policies = [policy for _, policy in results]

        # Frame-correct the visit distributions while the positions — and thus their
        # active_player_ids — are still in scope. Without a transform the raw str(ply)
        # distributions are stored verbatim (identity). Note the batch spans several
        # different games, so a transform may not assume one game per call.
        if self._policy_transform is not None:
            policies = self._policy_transform(positions, policies)

        # Record the encoded position and visit distribution for each game's step.
        # Target values are not yet known — they depend on the final game outcome.
        encoded = self._evaluator.encode_positions(positions)
        for game, encoded_position, policy in zip(live, encoded, policies, strict=True):
            game.step_records.append((encoded_position, policy))

        for game, new_position in zip(live, self._batch_ops.apply_plies(positions, plies), strict=True):
            game.position = new_position

    def _back_fill(
        self, step_records: Sequence[tuple[Tensor, dict[str, float]]], final_outcome: float
    ) -> list[TrainingSample]:
        """Turn one finished game's steps into samples, newest step first.

        Assigns target values by propagating the outcome backwards through the game.
        The terminal position's outcome is from the perspective of the player who
        would move next — i.e. the player who did NOT make the last ply. So the last
        recorded step (taken by the other player) has value = -final_outcome, and the
        sign alternates for each earlier step.
        """
        samples: list[TrainingSample] = []
        value = -final_outcome
        for encoded_position, policy in reversed(step_records):
            samples.append(TrainingSample(
                encoded_position=encoded_position,
                target_value=value,
                target_policy=policy,
            ))
            value = -value

        return samples

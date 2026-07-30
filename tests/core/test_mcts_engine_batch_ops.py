"""Pins MCTSEngine's routing through BatchPositionProcessor (batch-of-one, Step 2 of #22).

No search behaviour changes in this step — only that every position operation
the engine performs goes through the processor rather than straight at the
position. Two angles: a recording processor shows every call the engine makes
is one of the three batch methods, and a position whose scalar members raise
still searches successfully when paired with a processor that never consults
them, which is what pins the easy-to-miss fallback paths (the empty-children
and zero-total-visits branches).
"""

from collections.abc import Sequence
from typing import Literal

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.game.batch_position_processor import BatchPositionProcessor
from game_engine_core.models.position_evaluation import PositionEvaluation

from .nim_fixture import NimPly, NimPosition


class _RecordingBatchProcessor(BatchPositionProcessor[NimPly, NimPosition]):
    """Counts calls per method while still delegating to the base loop."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def outcomes(self, positions: Sequence[NimPosition]) -> Sequence[Literal[1, 0, -1] | None]:
        self.calls.append("outcomes")
        return super().outcomes(positions)

    def legal_plies(self, positions: Sequence[NimPosition]) -> Sequence[Sequence[NimPly]]:
        self.calls.append("legal_plies")
        return super().legal_plies(positions)

    def apply_plies(
        self, positions: Sequence[NimPosition], plies: Sequence[NimPly]
    ) -> Sequence[NimPosition]:
        self.calls.append("apply_plies")
        return super().apply_plies(positions, plies)


class _FixedPolicyEvaluator:
    """Evaluator that never touches the position, only str(ply) keys supplied by the engine."""

    def __init__(self, policy: dict[str, float]):
        self._policy = policy

    def evaluate_positions(self, positions: Sequence[NimPosition]) -> Sequence[PositionEvaluation]:
        return [PositionEvaluation(value=0.0, policy=dict(self._policy)) for _ in positions]


def _engine_with_recorder(iterations: int) -> tuple[MCTSEngine[NimPly, NimPosition, _FixedPolicyEvaluator], _RecordingBatchProcessor]:
    recorder = _RecordingBatchProcessor()
    engine = MCTSEngine(
        evaluator=_FixedPolicyEvaluator({"1": 0.5, "2": 0.5}),
        iterations=iterations,
        batch_ops=recorder,
    )
    return engine, recorder


def test_engine_reaches_positions_only_through_the_batch_processor() -> None:
    engine, recorder = _engine_with_recorder(iterations=20)

    engine.select_ply(NimPosition(pile=5))

    assert recorder.calls, "engine made no calls through batch_ops"
    assert set(recorder.calls) <= {"outcomes", "legal_plies", "apply_plies"}
    # All three seams get exercised across a search deep enough to expand,
    # terminate, and apply plies.
    assert set(recorder.calls) == {"outcomes", "legal_plies", "apply_plies"}


class _RaisingScalarsNimPosition(NimPosition):
    """A position whose GamePosition scalar members are poisoned.

    Used to prove the engine never reads them directly: only a processor that
    also avoids them can search this position successfully.
    """

    @property
    def outcome(self) -> Literal[1, 0, -1] | None:
        raise AssertionError("engine must not read position.outcome directly")

    @property
    def legal_plies(self) -> list[NimPly]:
        raise AssertionError("engine must not read position.legal_plies directly")

    def apply_ply(self, ply: NimPly) -> "_RaisingScalarsNimPosition":
        raise AssertionError("engine must not call position.apply_ply directly")


class _ScalarAvoidingBatchProcessor(BatchPositionProcessor[NimPly, _RaisingScalarsNimPosition]):
    """Reimplements all three seams from Nim internals, never touching the poisoned scalars."""

    def outcomes(
        self, positions: Sequence[_RaisingScalarsNimPosition]
    ) -> Sequence[Literal[1, 0, -1] | None]:
        return [(-1 if position.pile == 0 else None) for position in positions]

    def legal_plies(
        self, positions: Sequence[_RaisingScalarsNimPosition]
    ) -> Sequence[Sequence[NimPly]]:
        return [
            [NimPly(take) for take in (1, 2) if take <= position.pile]
            for position in positions
        ]

    def apply_plies(
        self, positions: Sequence[_RaisingScalarsNimPosition], plies: Sequence[NimPly]
    ) -> Sequence[_RaisingScalarsNimPosition]:
        return [
            _RaisingScalarsNimPosition(
                pile=position.pile - ply.take,
                active_player_id=-position.active_player_id,
            )
            for position, ply in zip(positions, plies, strict=True)
        ]


def test_search_succeeds_when_position_scalars_raise_but_processor_avoids_them() -> None:
    # Pins the fallback paths (_visit_distribution's zero-total branch and the
    # empty-children branch of _select_best_ply / _select_best_ply_with_temperature),
    # which are easy to leave reading root.position directly.
    engine = MCTSEngine(
        evaluator=_FixedPolicyEvaluator({"1": 0.5, "2": 0.5}),
        iterations=10,
        batch_ops=_ScalarAvoidingBatchProcessor(),
    )

    ply = engine.select_ply(_RaisingScalarsNimPosition(pile=5))
    assert ply.take in {1, 2}

    ply, policy = engine.select_plies_for_training([_RaisingScalarsNimPosition(pile=5)])[0]
    assert ply.take in {1, 2}
    assert set(policy) == {"1", "2"}


def test_zero_iteration_search_uses_only_the_processor() -> None:
    # iterations=0 never grows the tree, so select_ply goes straight to the
    # empty-children fallback in _select_best_ply.
    engine = MCTSEngine(
        evaluator=_FixedPolicyEvaluator({"1": 0.5, "2": 0.5}),
        iterations=0,
        batch_ops=_ScalarAvoidingBatchProcessor(),
    )

    ply = engine.select_ply(_RaisingScalarsNimPosition(pile=5))
    assert ply.take in {1, 2}

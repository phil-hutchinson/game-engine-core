"""BatchPositionProcessor tests, over the Nim fixture.

Covers the two things the base class promises: every method loops the
corresponding GamePosition member with results aligned to the input by index,
and a subclass can override exactly one method while the rest keep looping.
"""

from collections.abc import Sequence

import pytest

from game_engine_core.game.batch_position_processor import BatchPositionProcessor

from .nim_fixture import NimPly, NimPosition

NimBatchProcessor = BatchPositionProcessor[NimPly, NimPosition]


def test_outcomes_aligned_by_index() -> None:
    positions = [NimPosition(pile=0), NimPosition(pile=5)]

    assert NimBatchProcessor().outcomes(positions) == [-1, None]


def test_legal_plies_aligned_by_index_and_nested_one_level() -> None:
    positions = [NimPosition(pile=1), NimPosition(pile=5)]

    result = NimBatchProcessor().legal_plies(positions)

    assert [[ply.take for ply in plies] for plies in result] == [[1], [1, 2]]


def test_apply_plies_pairs_positions_and_plies_by_index() -> None:
    # A cross product would yield four results from two positions and two
    # plies; index-pairing yields exactly two.
    positions = [NimPosition(pile=5), NimPosition(pile=3)]
    plies = [NimPly(1), NimPly(2)]

    result = NimBatchProcessor().apply_plies(positions, plies)

    assert [position.pile for position in result] == [4, 1]


def test_apply_plies_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        NimBatchProcessor().apply_plies([NimPosition(pile=5)], [NimPly(1), NimPly(2)])


def test_empty_batch_behaviour() -> None:
    processor = NimBatchProcessor()

    assert processor.outcomes([]) == []
    assert processor.legal_plies([]) == []
    assert processor.apply_plies([], []) == []


class _OverridesLegalPliesOnly(BatchPositionProcessor[NimPly, NimPosition]):
    """Subclass overriding exactly one of the three batch methods."""

    def __init__(self) -> None:
        self.legal_plies_calls = 0

    def legal_plies(self, positions: Sequence[NimPosition]) -> Sequence[Sequence[NimPly]]:
        self.legal_plies_calls += 1
        return [list(position.legal_plies) for position in positions]


def test_selective_override_is_used() -> None:
    processor = _OverridesLegalPliesOnly()
    positions = [NimPosition(pile=5), NimPosition(pile=1)]

    result = processor.legal_plies(positions)

    assert processor.legal_plies_calls == 1
    assert [[ply.take for ply in plies] for plies in result] == [[1, 2], [1]]


def test_unoverridden_methods_still_loop_on_a_selective_subclass() -> None:
    processor = _OverridesLegalPliesOnly()
    positions = [NimPosition(pile=5), NimPosition(pile=0)]

    assert processor.outcomes(positions) == [None, -1]
    applied = processor.apply_plies([NimPosition(pile=5)], [NimPly(1)])
    assert [position.pile for position in applied] == [4]


def test_overridden_methods_reports_only_the_overridden_method() -> None:
    assert _OverridesLegalPliesOnly().overridden_methods() == frozenset({"legal_plies"})


def test_overridden_methods_is_empty_for_the_base_processor() -> None:
    assert NimBatchProcessor().overridden_methods() == frozenset()

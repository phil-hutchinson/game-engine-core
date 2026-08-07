"""Pins the fleet behaviour of MCTSEngine.select_plies_for_training (#23).

The single-game properties live in test_mcts_engine.py; these tests are about the
things only a fleet can get wrong. Two themes:

- **Batching.** All N trees advance one iteration together, so each iteration must
  make exactly one evaluator call, of width N — that is the whole point of the
  plural form. Terminal leaves are scored from their outcome and drop out of the
  batch, narrowing it without costing their game its iteration.
- **Lanes.** Game identity is the fleet position. A result that came back in the wrong
  order would be a silent, plausible-looking wrong answer, so alignment is tested
  against positions whose correct answers cannot be confused.

The evaluator here records the *width* of every call rather than just a count, which
is what distinguishes one batched call from N scalar ones — a call count alone passes
against a scalar implementation.
"""

from collections.abc import Sequence
from typing import Literal

import pytest

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator
from game_engine_core.game.batch_position_processor import BatchPositionProcessor
from game_engine_core.models.position_evaluation import PositionEvaluation

from .nim_fixture import NimPly, NimPosition

# Deep enough that no leaf reached within the iteration budgets used here is
# terminal, so every iteration takes the evaluator and batch widths stay full.
DEEP_PILE = 20


class _WidthRecordingEvaluator:
    """Uniform-policy evaluator that records the batch width of every call."""

    def __init__(self) -> None:
        self.batch_widths: list[int] = []

    @property
    def calls(self) -> int:
        return len(self.batch_widths)

    def evaluate_positions(self, positions: Sequence[NimPosition]) -> Sequence[PositionEvaluation]:
        self.batch_widths.append(len(positions))
        # Built per position rather than shared, so a policy covers exactly the plies
        # legal in the position it was produced for.
        return [
            PositionEvaluation(
                value=0.0,
                policy={str(ply): 1.0 / len(position.legal_plies) for ply in position.legal_plies},
            )
            for position in positions
        ]


class _WidthRecordingBatchProcessor(BatchPositionProcessor[NimPly, NimPosition]):
    """Records the batch width of every seam call while delegating to the base loop."""

    def __init__(self) -> None:
        self.legal_plies_widths: list[int] = []
        self.apply_plies_widths: list[int] = []
        self.outcomes_widths: list[int] = []

    def legal_plies(self, positions: Sequence[NimPosition]) -> Sequence[Sequence[NimPly]]:
        self.legal_plies_widths.append(len(positions))
        return super().legal_plies(positions)

    def apply_plies(
        self, positions: Sequence[NimPosition], plies: Sequence[NimPly]
    ) -> Sequence[NimPosition]:
        self.apply_plies_widths.append(len(positions))
        return super().apply_plies(positions, plies)

    def outcomes(self, positions: Sequence[NimPosition]) -> Sequence[Literal[1, 0, -1] | None]:
        self.outcomes_widths.append(len(positions))
        return super().outcomes(positions)


def _fleet_engine(
    iterations: int,
) -> tuple[MCTSEngine[NimPly, NimPosition, _WidthRecordingEvaluator], _WidthRecordingEvaluator]:
    evaluator = _WidthRecordingEvaluator()
    return MCTSEngine(evaluator=evaluator, iterations=iterations), evaluator


def test_each_iteration_makes_one_evaluator_call_spanning_the_whole_fleet() -> None:
    # The story's central claim: N games' evaluations are collected into one forward
    # pass per iteration instead of N calls of width one. Three deep piles keep every
    # selected leaf non-terminal, so no game ever drops out of the batch.
    engine, evaluator = _fleet_engine(iterations=5)

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)] * 3)

    assert evaluator.batch_widths == [3, 3, 3, 3, 3]


def test_a_single_game_is_the_fleet_at_width_one() -> None:
    engine, evaluator = _fleet_engine(iterations=5)

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)])

    assert evaluator.batch_widths == [1, 1, 1, 1, 1]


def test_results_are_aligned_with_the_positions_they_came_from() -> None:
    # Each game's policy must cover its own position's legal plies. Pile 1 has a
    # single legal ply and pile 5 has two, so a swapped result is not merely wrong
    # but impossible — the pile-1 game cannot legitimately offer a take of 2.
    # The forced game goes first deliberately: a fleet whose key-set signature reads
    # the same backwards would pass under a reversed result order.
    engine, _ = _fleet_engine(iterations=20)

    results = engine.select_plies_for_training(
        [NimPosition(pile=1), NimPosition(pile=5), NimPosition(pile=DEEP_PILE)]
    )

    assert [set(policy) for _, policy in results] == [{"1"}, {"1", "2"}, {"1", "2"}]
    # The forced game's ply is the one ply it has, whatever the search did elsewhere.
    assert results[0][0].take == 1


def test_fleet_order_follows_the_input_order_not_the_position_contents() -> None:
    # The same two positions in both orders: the results must swap with them. Guards
    # against an implementation that happens to be right for one ordering — e.g. one
    # that sorted or grouped the batch and never mapped it back.
    engine, _ = _fleet_engine(iterations=20)

    forwards = engine.select_plies_for_training([NimPosition(pile=1), NimPosition(pile=5)])
    backwards = engine.select_plies_for_training([NimPosition(pile=5), NimPosition(pile=1)])

    assert [set(policy) for _, policy in forwards] == [{"1"}, {"1", "2"}]
    assert [set(policy) for _, policy in backwards] == [{"1", "2"}, {"1"}]


def test_terminal_leaves_leave_the_batch_without_costing_their_game_an_iteration() -> None:
    # Pile 1: iteration 1 evaluates and expands the root, whose only child is the
    # empty pile — terminal. Every later iteration selects that child, reads its
    # outcome and skips the evaluator, so the batch narrows from 2 to 1 while the
    # deep game keeps evaluating.
    engine, evaluator = _fleet_engine(iterations=3)
    roots = engine._create_roots(  # pyright: ignore[reportPrivateUsage]
        [NimPosition(pile=1), NimPosition(pile=DEEP_PILE)]
    )

    engine._grow_trees(roots)  # pyright: ignore[reportPrivateUsage]

    assert evaluator.batch_widths == [2, 1, 1]
    # Lockstep is the point: the narrowed batch must not cost the terminal game its
    # iterations. Both trees ran all three.
    assert [root.visits for root in roots] == [3, 3]


def test_an_all_terminal_iteration_never_reaches_the_evaluator() -> None:
    # Both games are pile 1, so after the first iteration expands both roots every
    # subsequent iteration selects a terminal leaf in every tree and the non-terminal
    # partition is empty. Routine late in a game, and the evaluator must not be
    # handed an empty batch.
    engine, evaluator = _fleet_engine(iterations=4)
    roots = engine._create_roots(  # pyright: ignore[reportPrivateUsage]
        [NimPosition(pile=1), NimPosition(pile=1)]
    )

    engine._grow_trees(roots)  # pyright: ignore[reportPrivateUsage]

    assert evaluator.batch_widths == [2]
    assert [root.visits for root in roots] == [4, 4]


def test_each_fleet_position_gets_its_own_tree_and_its_own_iteration_budget() -> None:
    # Equal positions in two fleet positions are the case where a leaked or shared tree would
    # look most plausible: the answers would still be identical. Visit counts are
    # what give it away — each root must absorb the full budget, not a share of it.
    engine, _ = _fleet_engine(iterations=8)
    roots = engine._create_roots(  # pyright: ignore[reportPrivateUsage]
        [NimPosition(pile=5), NimPosition(pile=5)]
    )

    engine._grow_trees(roots)  # pyright: ignore[reportPrivateUsage]

    assert [root.visits for root in roots] == [8, 8]
    assert roots[0] is not roots[1]
    # The statistics arrays are where a shared tree would show up now that a
    # child's visits and values live on its parent rather than on itself.
    assert roots[0].child_visits is not roots[1].child_visits
    assert roots[0].children[0] is not roots[1].children[0]


def test_an_empty_fleet_does_no_work_at_all() -> None:
    # #24 shrinks the fleet as games finish, so a width-zero call is reachable. It must
    # return empty without touching the evaluator — and without running the iteration
    # loop either. Every phase of an empty iteration is a width-zero seam call, so a
    # loop that ran anyway would make 10 of them here and thousands at a realistic
    # budget, which a vectorised processor need not treat as free.
    recorder = _WidthRecordingBatchProcessor()
    evaluator = _WidthRecordingEvaluator()
    engine = MCTSEngine(evaluator=evaluator, iterations=10, batch_ops=recorder)

    assert engine.select_plies_for_training([]) == []
    assert evaluator.batch_widths == []
    assert recorder.outcomes_widths == []
    assert recorder.legal_plies_widths == []
    assert recorder.apply_plies_widths == []


def test_width_one_training_search_agrees_with_the_play_path() -> None:
    # The two surfaces share one iteration, so at N = 1 they must reach the same
    # conclusion from the same position and budget. Deterministic: PUCT descent and
    # visit-count selection both break ties on order, and neither path samples.
    position = NimPosition(pile=5)
    play_engine: MCTSEngine[NimPly, NimPosition, NullEvaluator[NimPly, NimPosition]] = MCTSEngine(
        evaluator=NullEvaluator(), iterations=50
    )
    training_engine: MCTSEngine[NimPly, NimPosition, NullEvaluator[NimPly, NimPosition]] = MCTSEngine(
        evaluator=NullEvaluator(), iterations=50
    )

    played = play_engine.select_ply(position)
    trained, _ = training_engine.select_plies_for_training([position])[0]

    assert played.take == trained.take


def test_expansion_never_calls_apply_plies() -> None:
    # #26 Step 4: expansion gives every leaf priors, nothing more — successor
    # positions are no longer built for every legal ply up front. One iteration
    # over three deep piles does nothing but expand the three roots (each is its
    # own first leaf), so apply_plies must not be called at all.
    recorder = _WidthRecordingBatchProcessor()
    engine = MCTSEngine(
        evaluator=_WidthRecordingEvaluator(), iterations=1, batch_ops=recorder
    )

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)] * 3)

    assert recorder.apply_plies_widths == []
    # Expansion asks for legality once for the whole fleet too, which #22 already
    # batched. Only the first entry belongs to expansion; later ones come from
    # choosing a ply once the search is over.
    assert recorder.legal_plies_widths[0] == 3


def test_lazy_materialisation_builds_one_successor_per_tree_in_one_batched_call() -> None:
    # #26 Step 5: a slot's successor is still built the first time descent picks
    # it, but every tree's pending materialisation for the wave is resolved by
    # one batched apply_plies call rather than one call per tree. The first
    # iteration only expands the three roots (each is its own first leaf, so
    # nothing is pending yet); the second iteration has all three trees descend
    # into their best slot and go pending together, producing exactly one call
    # of width 3 — not three calls of width 1.
    recorder = _WidthRecordingBatchProcessor()
    engine = MCTSEngine(
        evaluator=_WidthRecordingEvaluator(), iterations=2, batch_ops=recorder
    )

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)] * 3)

    assert recorder.apply_plies_widths == [3]


def test_one_wave_issues_at_most_one_apply_plies_call_bounded_by_fleet_size() -> None:
    # #26 Step 5: descent defers materialisation until the whole fleet has
    # descended, so a wave's pending slots are resolved by a single apply_plies
    # call rather than one per tree — and that call's width is bounded by how
    # many trees went pending, never by branching factor. Four deep piles branch
    # at 2: the first iteration only expands the roots (nothing pending yet, so
    # apply_plies is not called at all that wave), and the second has all four
    # trees pick their best, unmaterialised slot and go pending together.
    fleet_size = 4
    recorder = _WidthRecordingBatchProcessor()
    engine = MCTSEngine(
        evaluator=_WidthRecordingEvaluator(), iterations=2, batch_ops=recorder
    )

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)] * fleet_size)

    # Two waves ran; at most one of them had anything pending, so at most one
    # apply_plies call exists in total, and its width cannot exceed the fleet.
    assert len(recorder.apply_plies_widths) <= 1
    assert all(width <= fleet_size for width in recorder.apply_plies_widths)
    # This particular wave did have every tree go pending together.
    assert recorder.apply_plies_widths == [fleet_size]


def test_a_wave_that_only_revisits_materialised_slots_issues_no_apply_plies_call() -> None:
    # #26 Step 5: once a slot's successor is materialised, a later wave that
    # descends straight into it again has nothing pending — the descent stops
    # at an already-materialised leaf and never reaches the "unmaterialised
    # slot" branch. Two piles of 1 under the single-take fixture (takes=(1,))
    # both terminate after one ply: iteration 1 expands the bare roots
    # (nothing pending), iteration 2 materialises both single children — which
    # land on pile 0, terminal — in one batched call, and iteration 3
    # re-descends straight into those already-materialised terminal leaves,
    # contributing no apply_plies call at all.
    recorder = _WidthRecordingBatchProcessor()
    engine = MCTSEngine(
        evaluator=_WidthRecordingEvaluator(), iterations=3, batch_ops=recorder
    )
    roots = engine._create_roots(  # pyright: ignore[reportPrivateUsage]
        [NimPosition(pile=1, takes=(1,)), NimPosition(pile=1, takes=(1,))]
    )

    engine._grow_trees(roots)  # pyright: ignore[reportPrivateUsage]

    # Only iteration 2 had anything pending (both trees' single slot); the list
    # has exactly one entry even though three waves ran, so iteration 3 is the
    # zero-apply_plies wave this test is pinning.
    assert recorder.apply_plies_widths == [2]


def test_the_zero_visit_fallback_still_asks_for_legality_one_game_at_a_time() -> None:
    """Pins the width-one residue documented on MCTSEngine._visit_distribution."""
    # Documents a known width-one residue rather than endorsing it. A budget of one
    # expands each root but never descends past it, so every child sits at 0 visits
    # and the visit distribution falls back to a uniform over legal plies — asked for
    # per game, since that fallback was never widened. It fires only when the budget
    # cannot descend past a root, which is why it is tolerable; if it is ever widened,
    # this expectation becomes a single width-3 call.
    recorder = _WidthRecordingBatchProcessor()
    engine = MCTSEngine(
        evaluator=_WidthRecordingEvaluator(), iterations=1, batch_ops=recorder
    )

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)] * 3)

    assert recorder.legal_plies_widths[1:] == [1, 1, 1]


class _PolicyMissingTakeTwoEvaluator:
    """Returns a take-1-only policy: complete for pile 1, incomplete for anything larger."""

    def evaluate_positions(self, positions: Sequence[NimPosition]) -> Sequence[PositionEvaluation]:
        return [PositionEvaluation(value=0.0, policy={"1": 1.0}) for _ in positions]


def test_an_incomplete_policy_leaves_every_leaf_in_the_batch_unexpanded() -> None:
    # Expansion is all-or-nothing across the fleet, not just within one leaf. Game 0
    # is pile 1, whose single legal ply the policy covers, so it would expand cleanly
    # on its own; game 1 is pile 5, where the missing take-2 entry raises. Because
    # every prior resolves before any successor is built, the valid game must be left
    # unexpanded too — otherwise it would read as expanded and never be evaluated again.
    engine: MCTSEngine[NimPly, NimPosition, _PolicyMissingTakeTwoEvaluator] = MCTSEngine(
        evaluator=_PolicyMissingTakeTwoEvaluator(), iterations=5
    )
    roots = engine._create_roots(  # pyright: ignore[reportPrivateUsage]
        [NimPosition(pile=1), NimPosition(pile=5)]
    )

    with pytest.raises(ValueError, match="'2'"):
        engine._grow_trees(roots)  # pyright: ignore[reportPrivateUsage]

    assert [root.child_count for root in roots] == [0, 0]


def test_the_training_path_retains_nothing_between_calls() -> None:
    # Bare roots every call, unlike select_ply, which keeps its tree across plies.
    # Two identical calls must therefore agree exactly; a retained tree would give
    # the second call twice the search and could shift the distribution.
    engine, evaluator = _fleet_engine(iterations=6)
    position = NimPosition(pile=5)

    first = engine.select_plies_for_training([position])
    second = engine.select_plies_for_training([position])

    assert [policy for _, policy in first] == [policy for _, policy in second]
    # Equal work each time: nothing was carried over to shorten the second search.
    assert evaluator.batch_widths == [1] * 12
    # And the play path's retained root is untouched by training.
    assert engine._root_node is None  # pyright: ignore[reportPrivateUsage]

"""Pins the fleet behaviour of MCTSEngine.select_plies_for_training (#23).

The single-game properties live in test_mcts_engine.py; these tests are about the
things only a fleet can get wrong. Two themes:

- **Batching.** All N trees advance one iteration together, so each iteration must
  make exactly one evaluator call, of width N — that is the whole point of the
  plural form. Terminal leaves are scored from their outcome and drop out of the
  batch, narrowing it without costing their game its iteration.
- **Lanes.** Game identity is the slot index. A result that came back in the wrong
  order would be a silent, plausible-looking wrong answer, so alignment is tested
  against positions whose correct answers cannot be confused.

The evaluator here records the *width* of every call rather than just a count, which
is what distinguishes one batched call from N scalar ones — a call count alone passes
against a scalar implementation.
"""

from collections.abc import Sequence

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator
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


def _fleet_engine(
    iterations: int,
) -> tuple[MCTSEngine[NimPly, NimPosition, _WidthRecordingEvaluator], _WidthRecordingEvaluator]:
    evaluator = _WidthRecordingEvaluator()
    return MCTSEngine(evaluator=evaluator, iterations=iterations), evaluator


def test_each_iteration_makes_one_evaluator_call_spanning_the_whole_fleet() -> None:
    # The story's central claim: N games' evaluations are collected into one forward
    # pass per iteration instead of N calls of width one. Three deep piles keep every
    # selected leaf non-terminal, so no slot ever drops out of the batch.
    engine, evaluator = _fleet_engine(iterations=5)

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)] * 3)

    assert evaluator.batch_widths == [3, 3, 3, 3, 3]


def test_a_single_game_is_the_fleet_at_width_one() -> None:
    engine, evaluator = _fleet_engine(iterations=5)

    engine.select_plies_for_training([NimPosition(pile=DEEP_PILE)])

    assert evaluator.batch_widths == [1, 1, 1, 1, 1]


def test_results_are_aligned_with_the_positions_they_came_from() -> None:
    # Each slot's policy must cover its own position's legal plies. Pile 1 has a
    # single legal ply and pile 5 has two, so a swapped result is not merely wrong
    # but impossible — the pile-1 slot cannot legitimately offer a take of 2.
    # The forced slot goes first deliberately: a fleet whose key-set signature reads
    # the same backwards would pass under a reversed result order.
    engine, _ = _fleet_engine(iterations=20)

    results = engine.select_plies_for_training(
        [NimPosition(pile=1), NimPosition(pile=5), NimPosition(pile=DEEP_PILE)]
    )

    assert [set(policy) for _, policy in results] == [{"1"}, {"1", "2"}, {"1", "2"}]
    # The forced slot's ply is the one ply it has, whatever the search did elsewhere.
    assert results[0][0].take == 1


def test_slot_order_follows_the_input_order_not_the_position_contents() -> None:
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
    # deep slot keeps evaluating.
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
    # Both slots are pile 1, so after the first iteration expands both roots every
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


def test_each_slot_gets_its_own_tree_and_its_own_iteration_budget() -> None:
    # Equal positions in two slots are the case where a leaked or shared tree would
    # look most plausible: the answers would still be identical. Visit counts are
    # what give it away — each root must absorb the full budget, not a share of it.
    engine, _ = _fleet_engine(iterations=8)
    roots = engine._create_roots(  # pyright: ignore[reportPrivateUsage]
        [NimPosition(pile=5), NimPosition(pile=5)]
    )

    engine._grow_trees(roots)  # pyright: ignore[reportPrivateUsage]

    assert [root.visits for root in roots] == [8, 8]
    assert roots[0] is not roots[1]
    assert roots[0].children[0] is not roots[1].children[0]


def test_an_empty_fleet_returns_no_results_and_evaluates_nothing() -> None:
    # #24 shrinks the fleet as games finish, so a width-zero call is reachable.
    engine, evaluator = _fleet_engine(iterations=10)

    assert engine.select_plies_for_training([]) == []
    assert evaluator.batch_widths == []


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

"""MCTSEngine tests, centred on the sign conventions the search depends on.

Uses the Nim fixture with takes 1-2: from pile 2 taking 2 wins immediately, giving
the search a known-correct answer. The null evaluator (value 0, no policy) keeps
all signal coming from terminal outcomes, which is exactly what these tests pin.
"""

import math
import random
from collections.abc import Sequence

import numpy as np
import pytest

from game_engine_core.engines.mcts_engine import MCTSEngine, MCTSNode
from game_engine_core.evaluators.null_evaluator import NullEvaluator
from game_engine_core.models.position_evaluation import PositionEvaluation

from .nim_fixture import NimPly, NimPosition

NimMCTSEngine = MCTSEngine[NimPly, NimPosition, NullEvaluator[NimPly, NimPosition]]


def _engine(iterations: int, temperature: float = 0.0) -> NimMCTSEngine:
    return MCTSEngine(
        evaluator=NullEvaluator(), iterations=iterations, temperature=temperature
    )


def _slot_of(root: MCTSNode[NimPosition, NimPly], ply: str) -> int:
    """The slot a ply occupies in a node's arrays.

    Children are slots rather than objects, so a test that wants a particular
    ply's statistics looks its slot up here and indexes the parent's arrays.
    """
    return next(slot for slot, slot_ply in enumerate(root.child_plies) if str(slot_ply) == ply)


def test_forced_win_in_one_is_found() -> None:
    assert _engine(iterations=200).select_ply(NimPosition(pile=2)).take == 2


def test_search_values_carry_correct_signs() -> None:
    # Inspects the tree via the private _create_root/_grow_trees: the per-node value signs are
    # the convention under test and are not observable through the public API.
    engine = _engine(iterations=100)
    root = engine._create_root(NimPosition(pile=2))  # pyright: ignore[reportPrivateUsage]
    engine._grow_trees([root])  # pyright: ignore[reportPrivateUsage]

    assert {str(ply) for ply in root.child_plies} == {"1", "2"}
    take_one, take_two = _slot_of(root, "1"), _slot_of(root, "2")

    # The take-2 child is terminal: every visit evaluates to its exact outcome,
    # -1 from the perspective of the player who just lost.
    assert root.child_average_value(take_two) == -1.0
    # From the root mover's perspective the position is winning.
    assert root.average_value > 0
    # The winning ply attracts the visits.
    assert root.child_visits[take_two] > root.child_visits[take_one]
    # Visit accounting: every iteration's backpropagation path passes through the
    # root, but the first iteration stops at the root itself (evaluating and
    # expanding it), so only the remaining 99 reach a child.
    assert root.visits == 100
    assert root.child_visits.sum() == 99


def test_select_ply_on_position_without_plies_raises() -> None:
    with pytest.raises(RuntimeError):
        _engine(iterations=10).select_ply(NimPosition(pile=0))


def test_single_legal_ply_gets_full_distribution() -> None:
    ply, policy = _engine(iterations=10).select_plies_for_training([NimPosition(pile=1)])[0]
    assert ply.take == 1
    assert policy == {"1": 1.0}


def test_visit_distribution_covers_all_legal_plies_and_sums_to_one() -> None:
    _, policy = _engine(iterations=100).select_plies_for_training([NimPosition(pile=5)])[0]
    assert set(policy) == {"1", "2"}
    assert all(p >= 0.0 for p in policy.values())
    assert sum(policy.values()) == pytest.approx(1.0)


def test_visit_distribution_includes_zero_visit_plies() -> None:
    # Iteration 1 evaluates and expands the root, attaching both children at 0
    # visits; iteration 2 descends into exactly one of them. The sibling is a
    # child by then rather than an unexplored ply, but must still appear in the
    # distribution with probability 0.
    _, policy = _engine(iterations=2).select_plies_for_training([NimPosition(pile=5)])[0]
    assert set(policy) == {"1", "2"}
    assert sorted(policy.values()) == [0.0, 1.0]


def test_visit_distribution_is_uniform_while_every_child_is_unvisited() -> None:
    # After the single iteration that expands the root, every child exists at 0
    # visits, so there are no counts to normalise and the uniform fallback holds.
    _, policy = _engine(iterations=1).select_plies_for_training([NimPosition(pile=5)])[0]
    assert policy == {"1": 0.5, "2": 0.5}


def test_visit_distribution_uniform_fallback_without_visits() -> None:
    _, policy = _engine(iterations=0).select_plies_for_training([NimPosition(pile=5)])[0]
    assert policy == {"1": 0.5, "2": 0.5}


def test_backpropagation_alternates_value_sign_per_level() -> None:
    # Drives the private _backpropagate directly: the per-level sign flip is the
    # convention under test and is not observable through the public API.
    # Built through expand/attach_children rather than by hand, so each node ends
    # up addressing its statistics through its parent's arrays the way the search
    # would leave it — which is the storage the sign flip has to land in.
    engine = _engine(iterations=0)
    root: MCTSNode[NimPosition, NimPly] = MCTSNode(
        position=NimPosition(pile=3), parent=None, ply_from_parent=None
    )
    root.expand([NimPly(1)], [1.0])
    root.attach_children([NimPosition(pile=2)])
    mid = root.children[0]
    mid.expand([NimPly(1)], [1.0])
    mid.attach_children([NimPosition(pile=1)])
    leaf = mid.children[0]

    engine._backpropagate(leaf, 1.0)  # pyright: ignore[reportPrivateUsage]

    assert (leaf.total_value, mid.total_value, root.total_value) == (1.0, -1.0, 1.0)
    assert (leaf.visits, mid.visits, root.visits) == (1, 1, 1)

    engine._backpropagate(leaf, -1.0)  # pyright: ignore[reportPrivateUsage]

    assert (leaf.total_value, mid.total_value, root.total_value) == (0.0, 0.0, 0.0)
    assert (leaf.visits, mid.visits, root.visits) == (2, 2, 2)


class _FixedPolicyEvaluator:
    """Evaluator returning a known, deliberately skewed policy over takes 1-2."""

    def __init__(self, policy: dict[str, float]):
        self._policy = policy
        self.calls = 0

    def evaluate_positions(self, positions: Sequence[NimPosition]) -> Sequence[PositionEvaluation]:
        self.calls += 1
        return [PositionEvaluation(value=0.0, policy=dict(self._policy)) for _ in positions]


def _policy_engine(
    policy: dict[str, float], iterations: int
) -> tuple[MCTSEngine[NimPly, NimPosition, _FixedPolicyEvaluator], _FixedPolicyEvaluator]:
    evaluator = _FixedPolicyEvaluator(policy)
    return MCTSEngine(evaluator=evaluator, iterations=iterations), evaluator


def test_first_iteration_evaluates_the_root_and_attaches_every_child() -> None:
    # The root is simply the first leaf, so one iteration is enough to expand it
    # fully — and its children carry real priors from the evaluator rather than
    # the uniform default the old expand-a-child flow left them with.
    engine, evaluator = _policy_engine({"1": 0.25, "2": 0.75}, iterations=1)
    root = engine._create_root(NimPosition(pile=5))  # pyright: ignore[reportPrivateUsage]
    engine._grow_trees([root])  # pyright: ignore[reportPrivateUsage]

    assert evaluator.calls == 1
    assert root.visits == 1
    # The priors array *is* the policy, positional rather than string-keyed.
    priors = {
        str(ply): float(prior)
        for ply, prior in zip(root.child_plies, root.child_priors, strict=True)
    }
    assert priors == {"1": 0.25, "2": 0.75}
    # Expansion alone confers no visits: the value backpropagated is the root's.
    assert [int(visits) for visits in root.child_visits] == [0, 0]


def test_zero_visit_root_children_are_ranked_by_prior() -> None:
    # A budget too small to descend past the root leaves every child at 0 visits,
    # where a plain visit-count max returns whichever ply is first in legal order
    # ("1" here) and ignores the priors the iteration just computed. The tie
    # breaks on prior instead, so the chosen ply follows the policy rather than
    # the order the plies happened to be enumerated in.
    engine, _ = _policy_engine({"1": 0.25, "2": 0.75}, iterations=1)

    assert str(engine.select_ply(NimPosition(pile=5))) == "2"


def test_a_dominant_prior_is_reselected_while_its_sibling_stays_unvisited() -> None:
    # The point of true PUCT. An unvisited child scores on its exploration term
    # alone, which is proportional to its prior, so a 0.99-prior ply is taken
    # every time while its 0.01-prior sibling never gets a visit. The old
    # one-child-per-iteration expansion could not produce this: it visited every
    # sibling once before PUCT governed anything.
    # Pile 20 keeps every leaf reached within 5 iterations non-terminal, so the
    # values stay flat at 0 and the priors are the only thing driving selection.
    engine, _ = _policy_engine({"1": 0.99, "2": 0.01}, iterations=5)
    root = engine._create_root(NimPosition(pile=20))  # pyright: ignore[reportPrivateUsage]
    engine._grow_trees([root])  # pyright: ignore[reportPrivateUsage]

    visits = {
        str(ply): int(count) for ply, count in zip(root.child_plies, root.child_visits, strict=True)
    }
    assert visits == {"1": 4, "2": 0}


def _node_with_child_stats(
    parent_visits: int, priors: list[float], visits: list[int], total_values: list[float]
) -> MCTSNode[NimPosition, NimPly]:
    """A root carrying arbitrary child statistics, for exercising selection directly.

    child_puct_values only reads the parent's arrays and its own visit count, so
    the children never need real successor positions or objects of their own.
    """
    node: MCTSNode[NimPosition, NimPly] = MCTSNode(
        position=NimPosition(pile=99), parent=None, ply_from_parent=None
    )
    node.root_visits = parent_visits
    node.child_plies = [NimPly(1) for _ in priors]
    node.child_priors = np.array(priors, dtype=np.float64)
    node.child_visits = np.array(visits, dtype=np.int64)
    node.child_total_values = np.array(total_values, dtype=np.float64)
    return node


def _reference_puct_slot(node: MCTSNode[NimPosition, NimPly], exploration_constant: float = 1.41) -> int:
    """Scalar reference for the winning slot, mirroring child_puct_values slot by slot."""
    best_slot = 0
    best_score = float("-inf")
    for slot in range(node.child_count):
        visits = int(node.child_visits[slot])
        average_value = 0.0 if visits == 0 else float(node.child_total_values[slot]) / visits
        exploitation = -average_value
        exploration = (
            exploration_constant * float(node.child_priors[slot]) * math.sqrt(node.visits) / (1 + visits)
        )
        score = exploitation + exploration
        if score > best_score:
            best_score = score
            best_slot = slot
    return best_slot


def test_vectorised_puct_selection_agrees_with_a_scalar_reference() -> None:
    # Drives both the vectorised child_puct_values and an independently written
    # scalar loop over the same statistics, and requires they pick the same slot.
    # Covers the cases the plan calls out by name plus a randomised sweep, since
    # the vectorised zero-visit guard (np.divide's out/where) is exactly the kind
    # of thing that can quietly diverge from the scalar version it replaces.
    cases = [
        # every slot unvisited: exploitation is 0 everywhere, priors alone decide.
        _node_with_child_stats(parent_visits=5, priors=[0.5, 0.2, 0.3], visits=[0, 0, 0], total_values=[0.0, 0.0, 0.0]),
        # exactly one visited slot.
        _node_with_child_stats(
            parent_visits=8, priors=[0.4, 0.4, 0.2], visits=[0, 4, 0], total_values=[0.0, 1.5, 0.0]
        ),
        # an exact tie across every slot: first index must win, not a random one.
        _node_with_child_stats(
            parent_visits=3, priors=[0.25] * 4, visits=[0] * 4, total_values=[0.0] * 4
        ),
    ]

    rng = random.Random(20260803)
    for _ in range(200):
        slot_count = rng.randint(1, 6)
        visits = [rng.randint(0, 20) for _ in range(slot_count)]
        cases.append(
            _node_with_child_stats(
                parent_visits=rng.randint(0, 50),
                priors=[rng.random() for _ in range(slot_count)],
                visits=visits,
                total_values=[rng.uniform(-v, v) if v else 0.0 for v in visits],
            )
        )

    for case in cases:
        assert int(np.argmax(case.child_puct_values())) == _reference_puct_slot(case)


def test_evaluator_is_called_exactly_once_per_iteration() -> None:
    # The property the fleet wave depends on. Pile 20 is deep enough that no leaf
    # reached in 5 iterations is terminal, so every iteration takes the evaluator.
    engine, evaluator = _policy_engine({"1": 0.5, "2": 0.5}, iterations=5)
    engine.select_ply(NimPosition(pile=20))

    assert evaluator.calls == 5


def test_terminal_leaves_are_scored_from_their_outcome_without_evaluating() -> None:
    # From pile 1 the only ply empties the pile, so after the root is expanded
    # every later iteration lands on that terminal child and re-reads its
    # outcome. The evaluator is never called again.
    engine, evaluator = _policy_engine({"1": 1.0}, iterations=5)
    root = engine._create_root(NimPosition(pile=1))  # pyright: ignore[reportPrivateUsage]
    engine._grow_trees([root])  # pyright: ignore[reportPrivateUsage]

    assert evaluator.calls == 1
    assert root.child_visits[0] == 4
    # -1 from the perspective of the player left facing the empty pile.
    assert root.child_average_value(0) == -1.0


class _IncompletePolicyEvaluator:
    """Evaluator whose policy omits a legal ply — a contract violation."""

    def evaluate_positions(self, positions: Sequence[NimPosition]) -> Sequence[PositionEvaluation]:
        return [PositionEvaluation(value=0.0, policy={"1": 1.0}) for _ in positions]


def test_policy_missing_a_legal_ply_raises() -> None:
    engine: MCTSEngine[NimPly, NimPosition, _IncompletePolicyEvaluator] = MCTSEngine(
        evaluator=_IncompletePolicyEvaluator(), iterations=50
    )
    # The message must name the offending ply: it is what tells an evaluator
    # author which entry their policy head dropped.
    with pytest.raises(ValueError, match="'2'"):
        engine.select_ply(NimPosition(pile=5))


def test_incomplete_policy_leaves_the_node_unexpanded() -> None:
    # Expansion is all-or-nothing. A node left holding the slots created before
    # the bad ply would read as expanded and never be evaluated again.
    engine: MCTSEngine[NimPly, NimPosition, _IncompletePolicyEvaluator] = MCTSEngine(
        evaluator=_IncompletePolicyEvaluator(), iterations=50
    )
    root = engine._create_root(NimPosition(pile=5))  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError):
        engine._grow_trees([root])  # pyright: ignore[reportPrivateUsage]

    assert root.child_count == 0


def test_temperature_zero_picks_most_visited_ply() -> None:
    # Builds a root with known visit counts and drives the private _choose_plies:
    # a real search cannot guarantee a specific visit split.
    engine = _engine(iterations=0)
    root: MCTSNode[NimPosition, NimPly] = MCTSNode(
        position=NimPosition(pile=5), parent=None, ply_from_parent=None
    )
    root.expand([NimPly(1), NimPly(2)], [0.5, 0.5])
    root.attach_children(
        [NimPosition(pile=4, active_player_id=-1), NimPosition(pile=3, active_player_id=-1)]
    )
    root.child_visits[:] = [5, 10]

    assert engine._choose_plies([root])[0].take == 2  # pyright: ignore[reportPrivateUsage]


def test_temperature_sampling_returns_a_legal_ply() -> None:
    engine = _engine(iterations=50, temperature=1.0)
    assert engine.select_ply(NimPosition(pile=5)).take in {1, 2}


def test_observe_ply_rerroots_onto_matching_child_preserving_subtree() -> None:
    engine = _engine(iterations=50)
    position = NimPosition(pile=5)
    selected = engine.select_ply(position)
    root = engine._root_node  # pyright: ignore[reportPrivateUsage]
    assert root is not None
    matching_child = root.children[_slot_of(root, str(selected))]
    # Read out before re-rooting: until then these live in the old root's arrays,
    # which is exactly what detaching has to carry across.
    expected_visits = matching_child.visits
    expected_child_plies = matching_child.child_plies
    assert expected_visits > 0  # otherwise this test can't tell retention from a reset

    new_position = position.apply_ply(selected)
    engine.observe_ply(position, selected, new_position)

    new_root = engine._root_node  # pyright: ignore[reportPrivateUsage]
    assert new_root is matching_child
    assert new_root is not None
    assert new_root.parent is None
    assert new_root.slot is None
    assert new_root.ply_from_parent is None
    # The statistics survived the move out of the discarded parent's arrays and
    # into the node's own root scalars.
    assert new_root.visits == expected_visits
    assert new_root.child_plies is expected_child_plies


def test_observe_ply_miss_clears_root_and_rebuilds() -> None:
    # Full expansion makes a miss on a *legal* ply impossible: once the root has
    # been evaluated every legal ply has a slot, so the branch can only be reached
    # by a ply the tree has never seen. Constructed here with a take of 3, outside
    # the position's permitted takes, since no search can produce one.
    engine = _engine(iterations=10)
    position = NimPosition(pile=5)
    engine.select_ply(position)
    root = engine._root_node  # pyright: ignore[reportPrivateUsage]
    assert root is not None
    assert {ply.take for ply in root.child_plies} == {1, 2}
    unseen_ply = NimPly(3)
    new_position = NimPosition(pile=2, active_player_id=-1)

    engine.observe_ply(position, unseen_ply, new_position)

    assert engine._root_node is None  # pyright: ignore[reportPrivateUsage]
    selected = engine.select_ply(new_position)
    assert selected.take in {ply.take for ply in new_position.legal_plies}


def test_observe_ply_before_any_search_leaves_root_none_and_rebuilds() -> None:
    engine = _engine(iterations=10)
    position = NimPosition(pile=5)
    ply = NimPly(1)
    new_position = position.apply_ply(ply)

    engine.observe_ply(position, ply, new_position)

    assert engine._root_node is None  # pyright: ignore[reportPrivateUsage]
    selected = engine.select_ply(new_position)
    assert selected.take in {p.take for p in new_position.legal_plies}


def test_reset_clears_the_root_for_a_cold_start() -> None:
    engine = _engine(iterations=20)
    engine.select_ply(NimPosition(pile=5))
    assert engine._root_node is not None  # pyright: ignore[reportPrivateUsage]

    engine.reset()

    assert engine._root_node is None  # pyright: ignore[reportPrivateUsage]
    engine.select_ply(NimPosition(pile=3))
    root = engine._root_node  # pyright: ignore[reportPrivateUsage]
    assert root is not None
    assert root.position.pile == 3

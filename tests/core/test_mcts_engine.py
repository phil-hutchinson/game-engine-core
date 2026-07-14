"""MCTSEngine tests, centred on the sign conventions the search depends on.

Uses the Nim fixture with takes 1-2: from pile 2 taking 2 wins immediately, giving
the search a known-correct answer. The null evaluator (value 0, no policy) keeps
all signal coming from terminal outcomes, which is exactly what these tests pin.
"""

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


def test_forced_win_in_one_is_found() -> None:
    assert _engine(iterations=200).select_ply(NimPosition(pile=2)).take == 2


def test_search_values_carry_correct_signs() -> None:
    # Inspects the tree via the private _build_tree: the per-node value signs are
    # the convention under test and are not observable through the public API.
    engine = _engine(iterations=100)
    root = engine._create_root(NimPosition(pile=2))  # pyright: ignore[reportPrivateUsage]

    children = {str(child.ply_from_parent): child for child in root.children}
    assert set(children) == {"1", "2"}

    # The take-2 child is terminal: every visit evaluates to its exact outcome,
    # -1 from the perspective of the player who just lost.
    assert children["2"].average_value == -1.0
    # From the root mover's perspective the position is winning.
    assert root.average_value > 0
    # The winning ply attracts the visits.
    assert children["2"].visits > children["1"].visits
    # Visit accounting: every iteration's backpropagation path passes through
    # the root and exactly one of its children.
    assert root.visits == 100
    assert children["1"].visits + children["2"].visits == 100


def test_select_ply_on_position_without_plies_raises() -> None:
    with pytest.raises(RuntimeError):
        _engine(iterations=10).select_ply(NimPosition(pile=0))


def test_single_legal_ply_gets_full_distribution() -> None:
    ply, policy = _engine(iterations=10).select_ply_with_policy(NimPosition(pile=1))
    assert ply.take == 1
    assert policy == {"1": 1.0}


def test_visit_distribution_covers_all_legal_plies_and_sums_to_one() -> None:
    _, policy = _engine(iterations=100).select_ply_with_policy(NimPosition(pile=5))
    assert set(policy) == {"1", "2"}
    assert all(p >= 0.0 for p in policy.values())
    assert sum(policy.values()) == pytest.approx(1.0)


def test_visit_distribution_includes_zero_visit_plies() -> None:
    # One iteration expands exactly one root child; the other legal ply must
    # still appear in the distribution, with probability 0.
    _, policy = _engine(iterations=1).select_ply_with_policy(NimPosition(pile=5))
    assert set(policy) == {"1", "2"}
    assert sorted(policy.values()) == [0.0, 1.0]


def test_visit_distribution_uniform_fallback_without_visits() -> None:
    _, policy = _engine(iterations=0).select_ply_with_policy(NimPosition(pile=5))
    assert policy == {"1": 0.5, "2": 0.5}


def test_backpropagation_alternates_value_sign_per_level() -> None:
    # Drives the private _backpropagate directly: the per-level sign flip is the
    # convention under test and is not observable through the public API.
    engine = _engine(iterations=0)
    root: MCTSNode[NimPosition, NimPly] = MCTSNode(
        position=NimPosition(pile=3), parent=None, ply_from_parent=None
    )
    mid = MCTSNode(position=NimPosition(pile=2), parent=root, ply_from_parent=NimPly(1))
    leaf = MCTSNode(position=NimPosition(pile=1), parent=mid, ply_from_parent=NimPly(1))

    engine._backpropagate(leaf, 1.0)  # pyright: ignore[reportPrivateUsage]

    assert (leaf.total_value, mid.total_value, root.total_value) == (1.0, -1.0, 1.0)
    assert (leaf.visits, mid.visits, root.visits) == (1, 1, 1)

    engine._backpropagate(leaf, -1.0)  # pyright: ignore[reportPrivateUsage]

    assert (leaf.total_value, mid.total_value, root.total_value) == (0.0, 0.0, 0.0)
    assert (leaf.visits, mid.visits, root.visits) == (2, 2, 2)


def test_policy_priors_are_distributed_to_children() -> None:
    # Drives the private _expand_node with a policy already stored on the node
    # (the engine sets node.policy at evaluation time; see finding #2 in the
    # general-cleanup review for why the root itself never carries one).
    engine = _engine(iterations=0)
    node: MCTSNode[NimPosition, NimPly] = MCTSNode(
        position=NimPosition(pile=5), parent=None, ply_from_parent=None
    )
    node.policy = {"1": 0.25, "2": 0.75}

    engine._expand_node(node)  # pyright: ignore[reportPrivateUsage]
    engine._expand_node(node)  # pyright: ignore[reportPrivateUsage]

    priors = {str(child.ply_from_parent): child.prior for child in node.children}
    assert priors == {"1": 0.25, "2": 0.75}


class _IncompletePolicyEvaluator:
    """Evaluator whose policy omits a legal ply — a contract violation."""

    def evaluate_position(self, position: NimPosition) -> PositionEvaluation:
        return PositionEvaluation(value=0.0, policy={"1": 1.0})


def test_policy_missing_a_legal_ply_raises() -> None:
    engine: MCTSEngine[NimPly, NimPosition, _IncompletePolicyEvaluator] = MCTSEngine(
        evaluator=_IncompletePolicyEvaluator(), iterations=50
    )
    with pytest.raises(ValueError):
        engine.select_ply(NimPosition(pile=5))


def test_temperature_zero_picks_most_visited_ply() -> None:
    # Builds a root with known visit counts and drives the private _choose_ply:
    # a real search cannot guarantee a specific visit split.
    engine = _engine(iterations=0)
    root: MCTSNode[NimPosition, NimPly] = MCTSNode(
        position=NimPosition(pile=5), parent=None, ply_from_parent=None
    )
    root.children = [
        MCTSNode(position=NimPosition(pile=4, active_player_id=-1), parent=root,
                 ply_from_parent=NimPly(1), visits=5),
        MCTSNode(position=NimPosition(pile=3, active_player_id=-1), parent=root,
                 ply_from_parent=NimPly(2), visits=10),
    ]

    assert engine._choose_ply(root).take == 2  # pyright: ignore[reportPrivateUsage]


def test_temperature_sampling_returns_a_legal_ply() -> None:
    engine = _engine(iterations=50, temperature=1.0)
    assert engine.select_ply(NimPosition(pile=5)).take in {1, 2}

"""NullEvaluator tests.

The evaluator carries no knowledge, but it is no longer free of a policy: MCTS
requires one covering every legal ply, and building it here rather than
defaulting it in the engine is what keeps the search's expansion path single.
"""

import pytest

from game_engine_core.evaluators.null_evaluator import NullEvaluator

from .nim_fixture import NimPly, NimPosition

NimNullEvaluator = NullEvaluator[NimPly, NimPosition]


def test_value_is_zero() -> None:
    assert NimNullEvaluator().evaluate_position(NimPosition(pile=5)).value == 0.0


def test_policy_is_uniform_over_the_legal_plies() -> None:
    evaluation = NimNullEvaluator().evaluate_position(NimPosition(pile=5))

    assert evaluation.policy == {"1": 0.5, "2": 0.5}
    assert sum(evaluation.policy.values()) == pytest.approx(1.0)


def test_policy_covers_a_position_with_one_legal_ply() -> None:
    # Pile 1 permits only a take of 1: the mass all lands on the single ply.
    evaluation = NimNullEvaluator().evaluate_position(NimPosition(pile=1))

    assert evaluation.policy == {"1": 1.0}


def test_terminal_position_yields_an_empty_policy() -> None:
    # MCTS never evaluates a terminal leaf (it reads the outcome instead), but
    # the evaluator must not divide by a zero ply count if called directly.
    evaluation = NimNullEvaluator().evaluate_position(NimPosition(pile=0))

    assert evaluation.policy == {}

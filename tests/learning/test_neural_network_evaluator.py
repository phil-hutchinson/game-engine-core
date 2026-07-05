"""NeuralNetworkEvaluator base-class tests: the evaluate_position contract."""

import pytest

from tests.core.nim_fixture import NimPosition

from .nim_nn import NimMLP, NimNNEvaluator


def test_evaluation_has_bounded_value_and_normalised_policy() -> None:
    evaluator = NimNNEvaluator(model=NimMLP())
    evaluation = evaluator.evaluate_position(NimPosition(pile=5))

    assert -1.0 <= evaluation.value <= 1.0
    assert evaluation.policy is not None
    assert set(evaluation.policy) == {"1", "2"}
    assert sum(evaluation.policy.values()) == pytest.approx(1.0)


def test_policy_covers_only_legal_plies() -> None:
    evaluator = NimNNEvaluator(model=NimMLP())
    # Pile 1: take 2 is illegal and must receive no probability mass.
    evaluation = evaluator.evaluate_position(NimPosition(pile=1))

    assert evaluation.policy == {"1": pytest.approx(1.0)}


def test_inference_runs_in_eval_mode_even_after_training_left_train_mode() -> None:
    # With dropout active, train-mode inference is nondeterministic. The
    # evaluator must force eval mode itself (TrainingLoop switches the shared
    # model to train mode and never restores it), so repeated evaluations of
    # the same position are identical.
    model = NimMLP(dropout=0.5)
    evaluator = NimNNEvaluator(model=model)
    model.train()

    first = evaluator.evaluate_position(NimPosition(pile=5))
    second = evaluator.evaluate_position(NimPosition(pile=5))

    assert first.value == second.value
    assert first.policy == second.policy

"""NeuralNetworkEvaluator base-class tests: the evaluate_positions contract."""

import pytest
import torch
from torch import Tensor

from tests.core.nim_fixture import NimPosition

from .nim_nn import NimMLP, NimNNEvaluator


def test_evaluation_has_bounded_value_and_normalised_policy() -> None:
    evaluator = NimNNEvaluator(model=NimMLP())
    evaluation = evaluator.evaluate_positions([NimPosition(pile=5)])[0]

    assert -1.0 <= evaluation.value <= 1.0
    assert evaluation.policy is not None
    assert set(evaluation.policy) == {"1", "2"}
    assert sum(evaluation.policy.values()) == pytest.approx(1.0)


def test_policy_covers_only_legal_plies() -> None:
    evaluator = NimNNEvaluator(model=NimMLP())
    # Pile 1: take 2 is illegal and must receive no probability mass.
    evaluation = evaluator.evaluate_positions([NimPosition(pile=1)])[0]

    assert evaluation.policy == {"1": pytest.approx(1.0)}


def test_inference_runs_in_eval_mode_even_after_training_left_train_mode() -> None:
    # With dropout active, train-mode inference is nondeterministic. The
    # evaluator must force eval mode itself (TrainingLoop switches the shared
    # model to train mode and never restores it), so repeated evaluations of
    # the same position are identical.
    model = NimMLP(dropout=0.5)
    evaluator = NimNNEvaluator(model=model)
    model.train()

    first = evaluator.evaluate_positions([NimPosition(pile=5)])[0]
    second = evaluator.evaluate_positions([NimPosition(pile=5)])[0]

    assert first.value == second.value
    assert first.policy == second.policy


def test_inference_runs_in_eval_mode_on_a_genuine_batch() -> None:
    # Same check as above, but with both positions passed to a single
    # evaluate_positions call rather than two batch-of-one calls — pins eval
    # mode on the batched forward pass itself, not just the interim loop it
    # replaced. With dropout active and train mode leaking through, the two
    # rows of one batched forward would pick up independent dropout masks and
    # diverge even though the input rows are identical.
    model = NimMLP(dropout=0.5)
    evaluator = NimNNEvaluator(model=model)
    model.train()

    position = NimPosition(pile=5)
    first, second = evaluator.evaluate_positions([position, position])

    assert first.value == second.value
    assert first.policy == second.policy


def test_batched_evaluation_matches_elementwise_single_position_evaluation() -> None:
    # The point of the batching: N positions through one evaluate_positions
    # call must equal, elementwise, N separate batch-of-one calls.
    evaluator = NimNNEvaluator(model=NimMLP())
    positions = [NimPosition(pile=1), NimPosition(pile=5), NimPosition(pile=2)]

    batched = evaluator.evaluate_positions(positions)
    individually = [evaluator.evaluate_positions([position])[0] for position in positions]

    for from_batch, from_single in zip(batched, individually, strict=True):
        assert from_batch.value == pytest.approx(from_single.value)
        assert from_batch.policy.keys() == from_single.policy.keys()
        for key in from_batch.policy:
            assert from_batch.policy[key] == pytest.approx(from_single.policy[key])


def test_decode_policies_pairs_each_row_with_its_own_position() -> None:
    # Two positions with different legal sets: a row/position mispairing would
    # surface as a probability landing on a ply illegal for whichever position
    # it actually got paired with (pile 1 permits only "1"; pile 5 permits
    # "1" and "2").
    evaluator = NimNNEvaluator(model=NimMLP())
    positions = [NimPosition(pile=1), NimPosition(pile=5)]
    logits = torch.tensor([[10.0, -10.0], [0.0, 0.0]])

    policies = evaluator.decode_policies(logits, positions)

    assert set(policies[0]) == {"1"}
    assert set(policies[1]) == {"1", "2"}


def test_empty_batch_returns_no_evaluations_without_reaching_the_model() -> None:
    # The fleet wave calls evaluate_positions with the non-terminal subset of a
    # wave's leaves, which is empty whenever every selected leaf is terminal.
    # Reaching the model with it would raise (torch cannot stack an empty list),
    # so the base class must short-circuit — matching BatchPositionProcessor,
    # whose three methods are all defined on an empty batch.
    class _ExplodingModel(NimMLP):
        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            raise AssertionError("model must not be called for an empty batch")

    assert NimNNEvaluator(model=_ExplodingModel()).evaluate_positions([]) == []


class _FlatValueHeadMLP(NimMLP):
    """Value head returning (N,) rather than the required (N, 1)."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        value, policy_logits = super().forward(x)
        return value.squeeze(-1), policy_logits


def test_value_output_of_the_wrong_shape_raises_naming_both_shapes() -> None:
    # A (N,) value head is not merely an alternative layout: TrainingLoop's
    # targets are (N, 1), so it would broadcast into a silently wrong loss.
    # Caught here at inference, with a message naming what was expected and
    # what arrived — rather than as "iteration over a 0-d tensor" at N == 1
    # and no error at all above it.
    evaluator = NimNNEvaluator(model=_FlatValueHeadMLP())

    with pytest.raises(ValueError, match=r"shape \(2, 1\), got \(2,\)"):
        evaluator.evaluate_positions([NimPosition(pile=5), NimPosition(pile=3)])


def test_value_shape_check_applies_at_batch_of_one_too() -> None:
    # The N == 1 case is where the old squeeze(-1) handling failed with an
    # opaque TypeError, and it is the width every current call site uses.
    evaluator = NimNNEvaluator(model=_FlatValueHeadMLP())

    with pytest.raises(ValueError, match=r"shape \(1, 1\), got \(1,\)"):
        evaluator.evaluate_positions([NimPosition(pile=5)])

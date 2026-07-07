"""TrainingLoop tests, centred on epoch-loss reporting.

The reported epoch loss must be the sample-weighted mean of the per-batch mean
losses — exact even when the final batch is ragged (general-cleanup finding #7,
verified there with a scratch script; this is the durable version). The trick: a
zero learning rate freezes the model, so each batch's loss can be recomputed
independently and the expected epoch value derived by hand.
"""

from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F

from game_engine_learning.training_loop import TrainingLoop
from game_engine_learning.training_sample import TrainingSample

from .nim_nn import NimMLP, nim_policy_loss


def _samples() -> list[TrainingSample]:
    # Five samples with batch_size 2 gives batches of 2, 2, and a ragged 1.
    return [
        TrainingSample(
            encoded_position=torch.tensor([float(pile)], dtype=torch.float32),
            target_value=1.0 if pile % 2 else -1.0,
            target_policy={"1": 0.25, "2": 0.75} if pile >= 2 else {"1": 1.0},
        )
        for pile in (1, 2, 3, 4, 5)
    ]


def _expected_batch_losses(
    model: NimMLP, batch: Sequence[TrainingSample]
) -> tuple[float, float]:
    with torch.no_grad():
        predicted_values, policy_logits = model(
            torch.stack([sample.encoded_position for sample in batch])
        )
        value_loss = F.mse_loss(
            predicted_values,
            torch.tensor([[sample.target_value] for sample in batch], dtype=torch.float32),
        )
        policy_loss = nim_policy_loss(
            policy_logits, [sample.target_policy for sample in batch]
        )
    return value_loss.item(), policy_loss.item()


def test_epoch_loss_is_sample_weighted_mean_over_batches() -> None:
    model = NimMLP()
    loop = TrainingLoop(
        model=model,
        # lr=0: optimizer steps change nothing, so the model stays frozen and
        # the expected per-batch losses can be computed outside the loop.
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        policy_loss_fn=nim_policy_loss,
    )
    samples = _samples()

    history = loop.train(samples, epochs=1, batch_size=2, shuffle=False)

    batches = [samples[0:2], samples[2:4], samples[4:5]]
    expected = [_expected_batch_losses(model, batch) for batch in batches]
    expected_value = sum(v * len(b) for (v, _), b in zip(expected, batches, strict=True)) / len(samples)
    expected_policy = sum(p * len(b) for (_, p), b in zip(expected, batches, strict=True)) / len(samples)

    assert len(history) == 1
    assert history[0].value == pytest.approx(expected_value)
    assert history[0].policy == pytest.approx(expected_policy)


def test_total_combines_value_and_weighted_policy_loss() -> None:
    model = NimMLP()
    loop = TrainingLoop(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        policy_loss_fn=nim_policy_loss,
        policy_loss_weight=0.5,
    )

    history = loop.train(_samples(), epochs=2, batch_size=2, shuffle=False)

    assert len(history) == 2
    for epoch in history:
        assert epoch.policy_loss_weight == 0.5
        assert epoch.total == pytest.approx(epoch.value + 0.5 * epoch.policy)


def test_training_reduces_loss_on_fixed_targets() -> None:
    # With a real learning rate the loop must actually learn: repeated epochs
    # over a fixed sample set should reduce the combined loss. Seed the weight
    # init so the trajectory is fixed — reproducibility of an environment input,
    # not the "seed and assert an exact sequence" pattern the conventions warn against.
    torch.manual_seed(0)
    model = NimMLP()
    loop = TrainingLoop(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        policy_loss_fn=nim_policy_loss,
    )

    history = loop.train(_samples(), epochs=20, batch_size=2, shuffle=False)

    assert history[-1].total < history[0].total


def test_empty_sample_list_is_rejected() -> None:
    model = NimMLP()
    loop = TrainingLoop(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        policy_loss_fn=nim_policy_loss,
    )
    with pytest.raises(ValueError):
        loop.train([])

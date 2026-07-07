"""tictactoe_policy_loss tests: the str(ply)-to-column mapping and mean reduction."""

import pytest
import torch

from examples.tictactoe_learning.train import tictactoe_policy_loss


def test_mass_on_the_target_column_scores_better_than_elsewhere() -> None:
    # Target: all visits on square 5, which maps to logit column 4.
    target = [{"5": 1.0}]

    aligned = torch.full((1, 9), -10.0)
    aligned[0, 4] = 10.0
    misaligned = torch.full((1, 9), -10.0)
    misaligned[0, 0] = 10.0

    assert tictactoe_policy_loss(aligned, target).item() < 0.01
    assert tictactoe_policy_loss(misaligned, target).item() > 1.0


def test_loss_matches_hand_computed_cross_entropy() -> None:
    # Uniform logits: every column has softmax probability 1/9, so the
    # cross-entropy against any one-hot target is exactly log(9).
    logits = torch.zeros((1, 9))
    loss = tictactoe_policy_loss(logits, [{"1": 1.0}])
    assert loss.item() == pytest.approx(torch.log(torch.tensor(9.0)).item())


def test_loss_is_mean_reduced_over_the_batch() -> None:
    # TrainingLoop's epoch reporting requires mean reduction: duplicating a
    # sample must leave the loss unchanged.
    logits_one = torch.randn((1, 9))
    logits_two = logits_one.repeat(2, 1)
    target = {"3": 0.5, "7": 0.5}

    single = tictactoe_policy_loss(logits_one, [target])
    doubled = tictactoe_policy_loss(logits_two, [target, target])
    assert doubled.item() == pytest.approx(single.item())

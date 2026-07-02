from __future__ import annotations
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from .training_sample import TrainingSample


@dataclass(frozen=True)
class EpochLoss:
    """Mean value- and policy-head loss over one training epoch."""

    value: float # value loss i.e. PositionEvaluator error when evaluating board position
    policy: float # policy loss i.e. PositionEvaluator error when evaluating possible plies

    @property
    def total(self) -> float:
        """The combined loss that gradient descent actually minimises."""
        return self.value + self.policy


ValueLossFn = Callable[[Tensor, Tensor], Tensor]
"""(predicted_values, target_values) -> scalar loss. Both tensors have shape (batch, 1)."""

PolicyLossFn = Callable[[Tensor, Sequence[Mapping[str, float]]], Tensor]
"""(policy_logits, target_policies) -> scalar loss.

policy_logits has shape (batch, action_space); target_policies is the batch of MCTS
visit distributions keyed by str(ply). Aligning each str(ply) with its column in the
logits is the one piece of game-specific knowledge the loop cannot supply, so it lives
in this caller-provided loss rather than inside TrainingLoop.
"""


class TrainingLoop:
    """Joint value/policy gradient descent over a collection of TrainingSamples.

    Game-agnostic: the model defines the architecture, the optimizer, the update rule,
    and the two loss functions how predictions are scored against their targets. The
    loop only batches the samples, runs the forward/backward pass, and steps the
    optimizer.

    Args:
        model: The network being trained. Its forward() must accept a batched encoded
            position tensor and return (value, policy_logits) with shapes
            (batch, 1) and (batch, action_space).
        optimizer: A torch optimizer already bound to the model's parameters.
        policy_loss_fn: Scores the policy head against the MCTS visit distributions.
            This is where the game-specific str(ply) -> column mapping lives.
        value_loss_fn: Scores the value head against the self-play outcomes. Defaults to
            mean squared error, the standard choice for a tanh-bounded value head.
        policy_loss_weight: Scaling factor applied to the policy loss before summing with
            the value loss. Increase to emphasise ply selection; decrease if the policy
            term is swamping the value gradient. Defaults to 1.0 (equal weighting).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        policy_loss_fn: PolicyLossFn,
        value_loss_fn: ValueLossFn = F.mse_loss,
        policy_loss_weight: float = 1.0,
    ):
        self._model = model
        self._optimizer = optimizer
        self._policy_loss_fn = policy_loss_fn
        self._value_loss_fn = value_loss_fn
        self._policy_loss_weight = policy_loss_weight

    def train(
        self,
        samples: Sequence[TrainingSample],
        *,
        epochs: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> list[EpochLoss]:
        """Run `epochs` passes over `samples` and return the mean loss per epoch.

        Args:
            samples: Collection of play examples consisting of a position, an expected game result (value) and ply scores (policy).
                Samples will have been collected previously through self-play.
            epochs: Number of times each sample should be used for training.
            batch_size: Maximum number of samples to process before adjusting weights in the neural network.
            shuffle: Flag indicating samples should be processed in a different order each epoch. (Generally desirable.)

        Returns:
            List of EpochLoss reporting the average loss (error) per epoch, used for reporting purposes.
            A decreasing sequence indicates the network is fitting the self-play targets.
        """
        if not samples:
            raise ValueError("No training samples provided.")

        # switch the model to training mode  -- slower than inference (regular use of the model), but changes internal behaviours, allowing the model to be trained with samples.
        self._model.train()
        # reporting variable for output.
        history: list[EpochLoss] = []
        # list of indices indicating the order that the samples will be processed, used to implement shuffle.
        order = list(range(len(samples)))
        for _ in range(epochs):
            if shuffle:
                random.shuffle(order)
            value_total = 0.0
            policy_total = 0.0
            n_batches = 0
            for start in range(0, len(order), batch_size):
                # slice samples into a batch and train on it.
                batch = [samples[i] for i in order[start:start + batch_size]]
                value_loss, policy_loss = self._train_batch(batch)
                # update totals for reporting purposes.
                value_total += value_loss
                policy_total += policy_loss
                n_batches += 1
            # average the batch losses into a single per-epoch summary.
            history.append(EpochLoss(value=value_total / n_batches, policy=policy_total / n_batches))
        return history

    def _train_batch(self, batch: Sequence[TrainingSample]) -> tuple[float, float]:
        # Stack the per-sample encodings into a single (batch_size, *) tensor so the whole
        # minibatch flows through the network in one forward pass.
        encoded = torch.stack([sample.encoded_position for sample in batch])
        target_values = torch.tensor(
            [[sample.target_value] for sample in batch], dtype=torch.float32
        )

        # get updated value/policy based on current neural network. (The NN updates with each call to this method.)
        predicted_values, policy_logits = self._model(encoded)

        value_loss = self._value_loss_fn(predicted_values, target_values)
        policy_loss = self._policy_loss_fn(
            policy_logits, [sample.target_policy for sample in batch]
        )
        # Value and policy heads share the trunk, so their gradients are summed and the
        # backward pass updates the whole network jointly.
        loss = value_loss + self._policy_loss_weight * policy_loss

        # reset optimizer to clear any values from previous round of optimization
        self._optimizer.zero_grad()

        # compute gradients - i.e. how much each weight in the network contributed to the loss (error)
        # (Note: torch's Tensor.backward stub leaves its parameters partially untyped, which
        # Pylance flags on this otherwise fully-typed call; the call itself is correct.)
        loss.backward()  # type: ignore
        
        # Apply weights from previous step to model
        self._optimizer.step()

        # extract 
        # .item() detaches from the graph; float(...) on a grad-tracked tensor would warn.
        return value_loss.item(), policy_loss.item()

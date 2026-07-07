"""Training script: generate self-play data, train the network, save weights.

Each iteration plays a batch of self-play games with the current network, then runs a
few epochs of gradient descent over the resulting samples. Because self-play and
training share one model instance, every iteration generates data with a stronger
network than the last (the AlphaZero loop, scaled down for TicTacToe).

Usage:
    python -m examples.tictactoe_learning.train
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import TicTacToePosition
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_learning.checkpoints import checkpoint_path
from game_engine_learning.self_play_collector import SelfPlayCollector
from game_engine_learning.training_loop import TrainingLoop

from .tictactoe_mlp import TicTacToeMLP
from .tictactoe_nn_evaluator import TicTacToeNNEvaluator

# Derived artifact, not source — this directory is gitignored.
WEIGHTS_PATH = Path(__file__).parent / "weights" / "model.pt"


def tictactoe_policy_loss(
    policy_logits: Tensor, target_policies: Sequence[Mapping[str, float]]
) -> Tensor:
    """Cross-entropy between the MCTS visit distributions and the policy head.

    Builds a dense (batch, 9) target tensor from the per-sample policy dicts: each key is
    str(TicTacToePly), i.e. the square number 1-9, which maps to column square - 1. The
    target is then scored against the log-softmax of the policy logits. Squares absent
    from a sample's dict stay at probability 0, so the loss only rewards mass placed on
    plies the MCTS search actually visited.
    """
    targets = torch.zeros((len(target_policies), 9), dtype=torch.float32)
    for row, policy in enumerate(target_policies):
        for ply_str, prob in policy.items():
            targets[row, int(ply_str) - 1] = prob
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def main(
    iterations: int = 15,
    games_per_iteration: int = 25,
    epochs_per_iteration: int = 5,
    mcts_iterations: int = 200,
    checkpoint_every: int | None = None,
) -> None:
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    model = TicTacToeMLP()
    evaluator = TicTacToeNNEvaluator(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        policy_loss_fn=tictactoe_policy_loss,
    )

    # Temperature 1.0 keeps self-play exploratory so the data covers varied lines of play
    # instead of collapsing onto a single deterministic game every time.
    def engine_factory() -> MCTSEngine[TicTacToePly, TicTacToePosition, TicTacToeNNEvaluator]:
        return MCTSEngine(evaluator=evaluator, iterations=mcts_iterations, temperature=1.0)

    collector = SelfPlayCollector(
        evaluator=evaluator,
        engine_factory=engine_factory,
        position_factory=TicTacToePosition.new_game,
    )

    print(f"Training for {iterations} iterations ({games_per_iteration} games each)...")
    for iteration in range(1, iterations + 1):
        samples = collector.collect(games_per_iteration)
        losses = training_loop.train(
            samples, epochs=epochs_per_iteration, batch_size=64
        )
        first, last = losses[0], losses[-1]
        print(
            f"Iteration {iteration:>3}/{iterations}: "
            f"{len(samples):>4} samples | "
            f"total {first.total:.4f} -> {last.total:.4f} | "
            f"value {first.value:.4f} -> {last.value:.4f} | "
            f"policy {first.policy:.4f} -> {last.policy:.4f}"
        )
        if checkpoint_every is not None and iteration % checkpoint_every == 0:
            path = checkpoint_path(WEIGHTS_PATH.parent, iteration)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            print(f"Saved checkpoint to {path}")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nSaved trained weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TicTacToe neural network via self-play.")
    parser.add_argument("--iterations", type=int, default=15, help="Self-play/train iterations.")
    parser.add_argument("--games", type=int, default=25, help="Self-play games per iteration.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per iteration.")
    parser.add_argument("--mcts-iterations", type=int, default=200, help="MCTS iterations per ply during self-play.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a weights checkpoint every N iterations (off by default). "
        "Checkpoints can then be entered as players in tournament.py.",
    )
    args = parser.parse_args()
    main(
        iterations=args.iterations,
        games_per_iteration=args.games,
        epochs_per_iteration=args.epochs,
        mcts_iterations=args.mcts_iterations,
        checkpoint_every=args.checkpoint_every,
    )

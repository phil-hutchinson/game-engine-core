"""Self-play diagnostic script.

Runs a number of self-play games and prints summary statistics to confirm
that data collection is working correctly before any training takes place.

Usage:
    python -m examples.tictactoe_learning.selfplay
"""
import math

from game_engine_core.engines.mcts_engine import MCTSEngine
from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import TicTacToePosition
from game_engine_learning.self_play_collector import SelfPlayCollector
from game_engine_learning.training_sample import TrainingSample
from .tictactoe_mlp import TicTacToeMLP
from .tictactoe_nn_evaluator import TicTacToeNNEvaluator


def _policy_entropy(policy: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in policy.values() if p > 0)


def main(n_games: int = 20) -> None:
    model = TicTacToeMLP()
    evaluator = TicTacToeNNEvaluator(model=model)

    # temperature should be set to 1.0 for self-play training
    def engine_factory() -> MCTSEngine[TicTacToePly, TicTacToePosition, TicTacToeNNEvaluator]:
        return MCTSEngine(evaluator=evaluator, iterations=200, temperature=1.0)

    collector = SelfPlayCollector(
        evaluator=evaluator,
        engine_factory=engine_factory,
        position_factory=TicTacToePosition.new_game,
    )

    print(f"Running {n_games} self-play games...")
    samples: list[TrainingSample] = collector.collect(n_games)

    print(f"\nTotal samples collected: {len(samples)}")

    wins  = sum(1 for s in samples if s.target_value > 0)
    draws = sum(1 for s in samples if s.target_value == 0.0)
    losses = sum(1 for s in samples if s.target_value < 0)
    print(f"Value distribution:      {wins} wins / {draws} draws / {losses} losses")

    # higher entropies suggest each ply is weighted similarly (i.e the policy is not achieving much), whereas lower entropies indicate policies are homing in on fewer plies.
    entropies = [_policy_entropy(s.target_policy) for s in samples]
    mean_entropy = sum(entropies) / len(entropies)
    # uniform_entropy is the baseline: the mean entropy if the policy were perfectly uniform
    # over the legal plies at each position. mean_entropy should approach this with random weights
    # and fall below it as training causes the policy to favour specific plies.
    uniform_entropy = sum(math.log2(len(s.target_policy)) for s in samples) / len(samples)
    print(f"Mean policy entropy:     {mean_entropy:.3f} bits  (uniform over actual legal plies = {uniform_entropy:.3f} bits)")


if __name__ == "__main__":
    main()

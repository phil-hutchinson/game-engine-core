import argparse

import torch

from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import TicTacToePosition
from examples.tictactoe.tictactoe_ui import TicTacToeUI
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.engines.random_engine import RandomEngine
from game_engine_core.game.standard_game import StandardGame
from game_engine_core.models.game_result import GameResult
from game_engine_core.players.ai_player import AIPlayer
from game_engine_core.players.human_player import HumanPlayer

from .tictactoe_mlp import TicTacToeMLP
from .tictactoe_nn_evaluator import TicTacToeNNEvaluator
from .train import WEIGHTS_PATH


def _make_neural_engine(temperature: float = 0.0) -> MCTSEngine[TicTacToePly, TicTacToePosition, TicTacToeNNEvaluator]:
    if not WEIGHTS_PATH.exists():
        raise SystemExit(
            f"No trained weights found at {WEIGHTS_PATH}.\n"
            "Run `python -m examples.tictactoe_learning.train` first."
        )
    model = TicTacToeMLP()
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
    evaluator = TicTacToeNNEvaluator(model=model)
    return MCTSEngine(evaluator=evaluator, iterations=10, temperature=temperature)


def make_player(choice: str, symbol: str, ui: TicTacToeUI, render_before_ply: bool, temperature: float):
    match choice:
        case "human":
            return HumanPlayer(game_ui=ui, name=f"Player {symbol}")
        case "random":
            engine: RandomEngine[TicTacToePly, TicTacToePosition] = RandomEngine()
            return AIPlayer(engine=engine, name=f"Random ({symbol})", render_before_ply=render_before_ply)
        case "neural":
            return AIPlayer(engine=_make_neural_engine(temperature), name=f"Neural ({symbol})", render_before_ply=render_before_ply)
        case _:
            raise ValueError(f"Unknown player type: {choice}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p1",
        default="human",
        choices=["human", "random", "neural"],
        help="Player 1 (X): human, random, or neural",
    )
    parser.add_argument(
        "--p2",
        default="human",
        choices=["human", "random", "neural"],
        help="Player 2 (O): human, random, or neural",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Softmax temperature for neural ply selection. 0.0 (default) always picks the "
             "top move, so games are deterministic; higher values add variation at the cost "
             "of strength. ~0.3 keeps near-best play with slight variety.",
    )
    args = parser.parse_args()

    ui = TicTacToeUI()
    both_ai = args.p1 != "human" and args.p2 != "human"

    p1 = make_player(args.p1, "X", ui, render_before_ply=both_ai, temperature=args.temperature)
    p2 = make_player(args.p2, "O", ui, render_before_ply=both_ai, temperature=args.temperature)

    game = StandardGame(
        initial_position=TicTacToePosition.new_game(),
        players={1: p1, -1: p2},
        game_logging=ui,
        game_ui=ui,
    )

    result = game.run()
    _print_result(result)


def _print_result(result: GameResult):
    if result.outcome == 1:
        print("Player X wins!")
    elif result.outcome == -1:
        print("Player O wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()

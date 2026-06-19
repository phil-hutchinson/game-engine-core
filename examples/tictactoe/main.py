import argparse
from game_engine_core.game.standard_game import StandardGame
from game_engine_core.players.human_player import HumanPlayer
from game_engine_core.players.ai_player import AIPlayer
from game_engine_core.engines.random_engine import RandomEngine
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.models.game_result import GameResult
from .tictactoe_ply import TicTacToePly
from .tictactoe_position import TicTacToePosition
from .tictactoe_ui import TicTacToeUI
from game_engine_core.evaluators.null_evaluator import NullEvaluator
from .tictactoe_heuristic_evaluator import TicTacToeHeuristicEvaluator


def make_player(choice: str, symbol: str, ui: TicTacToeUI, render_before_ply: bool):
    match choice:
        case "human":
            return HumanPlayer(game_ui=ui, name=f"Player {symbol}")
        case "random":
            engine: RandomEngine[TicTacToePly, TicTacToePosition] = RandomEngine()
            return AIPlayer(engine=engine, name=f"Random ({symbol})", render_before_ply=render_before_ply)
        case "bruteforce":
            engine2: MCTSEngine[TicTacToePly, TicTacToePosition, NullEvaluator[TicTacToePly, TicTacToePosition]] = MCTSEngine(evaluator=NullEvaluator(), iterations=200_000)
            return AIPlayer(engine=engine2, name=f"Brute Force ({symbol})", render_before_ply=render_before_ply)
        case "heuristic":
            engine3: MCTSEngine[TicTacToePly, TicTacToePosition, TicTacToeHeuristicEvaluator] = MCTSEngine(evaluator=TicTacToeHeuristicEvaluator(), iterations=300)
            return AIPlayer(engine=engine3, name=f"Heuristic ({symbol})", render_before_ply=render_before_ply)
        case _:
            raise ValueError(f"Unknown player type: {choice}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p1",
        default="human",
        choices=["human", "random", "bruteforce", "heuristic"],
        help="Player 1 (X): human, random, bruteforce, or heuristic",
    )
    parser.add_argument(
        "--p2",
        default="human",
        choices=["human", "random", "bruteforce", "heuristic"],
        help="Player 2 (O): human, random, bruteforce, or heuristic",
    )
    args = parser.parse_args()

    ui = TicTacToeUI()
    both_ai = args.p1 != "human" and args.p2 != "human"

    p1 = make_player(args.p1, "X", ui, render_before_ply=both_ai)
    p2 = make_player(args.p2, "O", ui, render_before_ply=both_ai)

    game = StandardGame(
        initial_position=TicTacToePosition.new_game(),
        players={1: p1, -1: p2},
        game_ui=ui,
    )

    result = game.run()
    print_result(result)


def print_result(result: GameResult):
    if result.outcome == 1:
        print("Player X wins!")
    elif result.outcome == -1:
        print("Player O wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()

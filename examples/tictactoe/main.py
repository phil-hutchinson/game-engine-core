import argparse
from game_engine_core.game.standard_game import StandardGame
from game_engine_core.players.human_player import HumanPlayer
from game_engine_core.players.ai_player import AIPlayer
from game_engine_core.engines.random_engine import RandomEngine
from game_engine_core.models.game_result import GameResult
from .tictactoe_ply import TicTacToePly
from .tictactoe_position import TicTacToePosition
from .tictactoe_ui import TicTacToeUI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="hvh",
        choices=["hvh", "hvai", "aivh", "aivai"],
    )
    args = parser.parse_args()

    ui = TicTacToeUI()
    engine: RandomEngine[TicTacToePly, TicTacToePosition] = RandomEngine()

    match args.mode:
        case "hvh":
            p1 = HumanPlayer(game_ui=ui, name="Player X")
            p2 = HumanPlayer(game_ui=ui, name="Player O")
        case "hvai":
            p1 = HumanPlayer(game_ui=ui, name="Player X")
            p2 = AIPlayer(engine=engine, name="AI (O)")
        case "aivh":
            p1 = AIPlayer(engine=engine, name="AI (X)")
            p2 = HumanPlayer(game_ui=ui, name="Player O")
        case "aivai":
            p1 = AIPlayer(engine=engine, name="AI (X)", render_before_ply=True)
            p2 = AIPlayer(engine=engine, name="AI (O)", render_before_ply=True)
        case _:
            raise ValueError(f"Unexpected mode: {args.mode}")

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

from game_engine_core.game.standard_game import StandardGame
from game_engine_core.players.human_player import HumanPlayer
from game_engine_core.models.game_result import GameResult
from .tictactoe_position import TicTacToePosition
from .tictactoe_ui import TicTacToeUI

def main():
    ui = TicTacToeUI()
    
    player1 = HumanPlayer(game_ui=ui, name="Player X")
    player2 = HumanPlayer(game_ui=ui, name="Player O")
    
    game = StandardGame(
        initial_position=TicTacToePosition.new_game(),
        players={1: player1, -1: player2},
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
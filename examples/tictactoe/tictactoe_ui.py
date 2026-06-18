from game_engine_core.protocols.game_ui import GameUI
from .tictactoe_ply import TicTacToePly
from .tictactoe_position import TicTacToePosition

class TicTacToeUI(GameUI[TicTacToePly, TicTacToePosition]):

    def render_board(self, position: TicTacToePosition) -> None:
        print(self.text_board(position))

    def text_board(self, position: TicTacToePosition) -> str:
        def symbol(val: int) -> str:
            if val == 1: return 'X'
            if val == -1: return 'O'
            return ' '
        
        b = [symbol(val) for val in position.board]
        return (
            f" {b[0]} | {b[1]} | {b[2]} \n"
            f"---+---+---\n"
            f" {b[3]} | {b[4]} | {b[5]} \n"
            f"---+---+---\n"
            f" {b[6]} | {b[7]} | {b[8]} \n"
        )

    def get_next_ply(self, position: TicTacToePosition) -> TicTacToePly:
        player_symbol = 'X' if position.active_player_id == 1 else 'O'
        legal_squares = [p.square for p in position.legal_plies]
        
        while True:
            raw = ''
            try:
                raw = input(f"Player {player_symbol}, enter square (1-9): ")
                square = int(raw)
                if square not in legal_squares:
                    print(f"{square} is not a legal move. Legal squares: {legal_squares}")
                    self.render_board(position)
                else:
                    return TicTacToePly(square)
            except ValueError:
                print(f"Invalid input '{raw}' - please enter a number between 1 and 9")
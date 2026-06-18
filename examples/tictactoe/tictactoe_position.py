from __future__ import annotations
from typing import Literal, cast
from game_engine_core.protocols.game_position import GamePosition
from .tictactoe_ply import TicTacToePly

Board = tuple[Literal[1, -1, 0], ...]

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]

class TicTacToePosition(GamePosition[TicTacToePly]):

    def __init__(self, board: Board, active_player_id: Literal[1, -1]):
        self._board = board
        self._active_player_id: Literal[1, -1] = active_player_id

    @classmethod
    def new_game(cls) -> TicTacToePosition:
        return cls(
            board=cast('Board', (0,) * 9),
            active_player_id=1
        )

    @property
    def active_player_id(self) -> Literal[1, -1]:
        return self._active_player_id

    @property
    def outcome(self) -> Literal[1, 0, -1] | None:
        for line in WINNING_LINES:
            values = [self._board[i] for i in line]
            if values == [1, 1, 1]:
                return 1
            if values == [-1, -1, -1]:
                return -1
        if 0 not in self._board:
            return 0  # draw
        return None

    @property
    def legal_plies(self) -> list[TicTacToePly]:
        return [
            TicTacToePly(i + 1)
            for i, val in enumerate(self._board)
            if val == 0
        ]

    @property
    def board(self) -> Board:
        return self._board
    
    def apply_ply(self, ply: TicTacToePly) -> TicTacToePosition:
        index = ply.square - 1
        if self._board[index] != 0:
            raise ValueError(f"Square {ply.square} is already occupied")
        new_board = list(self._board)
        new_board[index] = self._active_player_id
        return TicTacToePosition(
            board=tuple(new_board),
            active_player_id=-self._active_player_id
        )
    
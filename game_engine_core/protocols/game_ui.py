from typing import Protocol, Any

from .game_ply import GamePly
from .game_position import GamePosition

class GameUI[TPly: GamePly, TPosition: GamePosition[Any]](Protocol):

    def text_board(self, position: TPosition) -> str:
        """Simple text-based representation of board"""
        ...

    def render_board(self, position: TPosition) -> None:
        """Called to display board (can be same as text_board())"""
        ...

    def get_next_ply(self, position: TPosition) -> TPly:
        """Gets selection of next ply from human player"""
        ...
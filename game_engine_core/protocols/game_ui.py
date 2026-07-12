from typing import Any, Protocol

from .game_ply import GamePly
from .game_position import GamePosition


class GameUI[TPly: GamePly, TPosition: GamePosition[Any]](Protocol):
    """Interactive display and input. Logging concerns live on GameLogging."""

    def render_board(self, position: TPosition) -> None:
        """Called to display board"""
        ...

    def get_next_ply(self, position: TPosition) -> TPly:
        """Gets selection of next ply from human player"""
        ...
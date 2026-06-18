from typing import Protocol

from .game_ply import GamePly
from .game_position import GamePosition

class Player[TPly: GamePly, TPosition: GamePosition[GamePly]](Protocol):
    @property
    def name(self) -> str:
        """Display name of the player."""
        ...
    
    @property
    def render_before_ply(self) -> bool:
        """Flag determining whether to render board before this player's turn"""
        ...

    def select_ply(self, position: TPosition) -> TPly:
        """Select the next ply given the current position."""
        ...
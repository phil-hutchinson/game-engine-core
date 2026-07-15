from typing import Any, Protocol

from .game_ply import GamePly
from .game_position import GamePosition


class Player[TPly: GamePly, TPosition: GamePosition[Any]](Protocol):
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

    def observe_ply(self, position: TPosition, ply: TPly, new_position: TPosition) -> None:
        """Notifies that a ply has been applied to the position: for engines that retain state """
        ...

    def reset(self) -> None:
        """Notifies that a new game has started and state can be reset"""
        ...


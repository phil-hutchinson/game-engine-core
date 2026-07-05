from typing import Any, Protocol

from .game_ply import GamePly
from .game_position import GamePosition


class GameEngine[TGamePly: GamePly, TGamePosition: GamePosition[Any]](Protocol):

    def select_ply(self, game_position: TGamePosition) -> TGamePly:
        """
            Select the next ply given a specific position.
        """
        ...
    

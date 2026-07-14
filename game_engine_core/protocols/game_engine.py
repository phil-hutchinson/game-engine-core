from typing import Any, Protocol

from .game_ply import GamePly
from .game_position import GamePosition


class GameEngine[TGamePly: GamePly, TGamePosition: GamePosition[Any]](Protocol):

    def select_ply(self, game_position: TGamePosition) -> TGamePly:
        """
            Select the next ply given a specific position.
        """
        ...

    def observe_ply(self, position: TGamePosition, ply: TGamePly, new_position: TGamePosition) -> None:
        """
            Notifies that a ply has been applied to the position: for engines that retain state
        """
        ...

    def reset(self) -> None:
        """
            Notifies that a new game has started, engine should clear any state
        """
        ...

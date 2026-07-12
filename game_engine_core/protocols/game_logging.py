from typing import Any, Protocol

from .game_ply import GamePly
from .game_position import GamePosition


class GameLogging[TPly: GamePly, TPosition: GamePosition[Any]](Protocol):
    """Game-specific rendering for the game record: board snapshots and ply annotations.

    Distinct from GameUI (interactive display/input) so headless play needs no UI.
    A game may implement both protocols on one class.
    """

    def text_board(self, position: TPosition) -> str:
        """Simple text-based representation of board"""
        ...

    def ply_annotation(self, from_position: TPosition, ply: TPly, to_position: TPosition) -> str:
        """The string to log for an executed ply.

        May be richer than str(ply) (captures, disambiguation, end-of-game markers
        need surrounding context), and str(ply) is a valid trivial implementation.
        Invoked once per executed ply — never during search — so it may be
        expensive. Must not replace str(ply) as the ply's identity key.
        """
        ...

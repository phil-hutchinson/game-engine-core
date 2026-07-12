from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, Self

from .game_ply import GamePly


class GamePosition[TGamePly: GamePly]  (Protocol):
    @property
    def outcome(self) -> Literal[1, 0, -1] | None:
        """Game outcome (current-player-relative): 1 (active player wins), 0 (draw), -1 (active player loses), None (ongoing)."""
        ...

    @property
    def outcome_reason(self) -> str | None:
        """Why the game ended: a short game-specific string (e.g. "Stalemate") once outcome is non-None, else None."""
        ...

    @property
    def active_player_id(self) -> Literal[1, -1]:
        """Which player's turn it is: 1 for player 1, -1 for player 2."""
        ...

    @property 
    def legal_plies(self) -> Sequence[TGamePly]:
        """Get all legal plies from this position."""
        ...
    
    def apply_ply(self, ply: TGamePly) -> Self:
        """Apply a ply and return the resulting position with next player to move."""
        ...

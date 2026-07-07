from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ..models.game_result import GameResult


@dataclass(frozen=True)
class GameRecord:
    """One completed tournament game: who held which side, and how it ended."""

    players: Mapping[Literal[1, -1], str]
    """Player names keyed by the side they held (side 1 moves first)."""

    result: GameResult

    def points_for_side(self, side: Literal[1, -1]) -> float:
        """Points earned by the given side: 1 for a win, 0.5 for a draw, 0 for a loss.

        GameResult.outcome is absolute (1 means side 1 won), so a side scored a
        win exactly when the outcome equals its side id. Keeping this sign logic
        here means the aggregations never touch outcome conventions directly.
        """
        if self.result.outcome == 0:
            return 0.5
        return 1.0 if self.result.outcome == side else 0.0

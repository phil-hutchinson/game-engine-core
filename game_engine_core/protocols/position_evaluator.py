from collections.abc import Sequence
from typing import Any, Protocol

from ..models.position_evaluation import PositionEvaluation
from .game_ply import GamePly
from .game_position import GamePosition


class PositionEvaluator[TPly: GamePly, TPosition: GamePosition[Any]](Protocol):
    """Protocol for evaluating game positions."""

    def evaluate_positions(self, positions: Sequence[TPosition]) -> Sequence[PositionEvaluation]:
        """Evaluate each position from its active player's perspective (current-player-relative).

        Index-aligned with ``positions``: evaluation ``i`` corresponds to
        ``positions[i]``.
        """
        ...

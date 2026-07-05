from typing import Any, Protocol

from ..models.position_evaluation import PositionEvaluation
from .game_ply import GamePly
from .game_position import GamePosition


class PositionEvaluator[TPly: GamePly, TPosition: GamePosition[Any]](Protocol):
    """Protocol for evaluating game positions."""
    
    def evaluate_position(self, position: TPosition) -> PositionEvaluation:
        """Evaluate position from the active player's perspective (current-player-relative)."""
        ...

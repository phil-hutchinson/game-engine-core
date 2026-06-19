from typing import Protocol, Any

from .game_position import GamePosition
from .game_ply import GamePly
from ..models.position_evaluation import PositionEvaluation

class PositionEvaluator[TPly: GamePly, TPosition: GamePosition[Any]](Protocol):
    """Protocol for evaluating game positions."""
    
    def evaluate_position(self, position: TPosition) -> PositionEvaluation:
        """Evaluate position from the active player's perspective (current-player-relative)."""
        ...

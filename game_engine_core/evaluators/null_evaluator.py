from typing import Any

from ..models.position_evaluation import PositionEvaluation
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition


class NullEvaluator[TPly: GamePly, TPosition: GamePosition[Any]]:
    def evaluate_position(self, position: TPosition) -> PositionEvaluation:
        return PositionEvaluation(value=0.0)

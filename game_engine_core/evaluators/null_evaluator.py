from typing import Any

from ..models.position_evaluation import PositionEvaluation
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition


class NullEvaluator[TPly: GamePly, TPosition: GamePosition[Any]]:
    """Evaluator with no knowledge: every position scores 0 with uniform priors.

    The uniform policy is built here rather than defaulted in the engine, so the
    search keeps a single expansion path and the cost of having no policy head
    falls on the evaluator that lacks one. Note this makes MCTS behave as PUCT
    with equal priors, not as UCT: an unvisited ply scores on its exploration
    term alone, so siblings are not each forced to a visit before the search
    begins exploiting.
    """

    def evaluate_position(self, position: TPosition) -> PositionEvaluation:
        legal_plies = position.legal_plies
        uniform_prior = 1.0 / len(legal_plies) if legal_plies else 0.0
        return PositionEvaluation(
            value=0.0,
            policy={str(ply): uniform_prior for ply in legal_plies},
        )

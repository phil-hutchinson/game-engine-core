from game_engine_core.models.position_evaluation import PositionEvaluation
from game_engine_core.protocols.position_evaluator import PositionEvaluator
from .tictactoe_ply import TicTacToePly
from .tictactoe_position import TicTacToePosition


class TicTacToeEvaluator(PositionEvaluator[TicTacToePly, TicTacToePosition]):

    def evaluate_position(self, position: TicTacToePosition) -> PositionEvaluation:
        return PositionEvaluation(value=0.0, evaluation_cost=0.0)

from game_engine_core.models.position_evaluation import PositionEvaluation
from .tictactoe_position import TicTacToePosition, WINNING_LINES, Board


def _count_threats(board: Board, player: int) -> int:
    return sum(
        1 for line in WINNING_LINES
        if sum(board[i] == player for i in line) == 2
        and sum(board[i] == 0 for i in line) == 1
    )


class TicTacToeHeuristicEvaluator:

    def evaluate_position(self, position: TicTacToePosition) -> PositionEvaluation:
        board = position.board
        me = position.active_player_id
        opp = -me

        my_threats = _count_threats(board, me)
        opp_threats = _count_threats(board, opp)

        score = 0.0
        score += 0.6 if my_threats >= 2 else (0.25 if my_threats == 1 else 0.0)
        score -= 0.6 if opp_threats >= 2 else (0.25 if opp_threats == 1 else 0.0)

        score += 0.05 if board[4] == me else (-0.05 if board[4] == opp else 0.0)
        for i in (0, 2, 6, 8):
            score += 0.025 if board[i] == me else (-0.025 if board[i] == opp else 0.0)

        return PositionEvaluation(value=max(-1.0, min(1.0, score)))

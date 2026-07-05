from game_engine_core.models.position_evaluation import PositionEvaluation

from .tictactoe_position import WINNING_LINES, Board, TicTacToePosition


def _count_threats(board: Board, player: int) -> int:
    return sum(
        1 for line in WINNING_LINES
        if sum(board[i] == player for i in line) == 2
        and sum(board[i] == 0 for i in line) == 1
    )


def _score_move(board: Board, player: int, idx: int) -> float:
    """Heuristic score for a candidate move, used to build the policy head output."""
    opp = -player
    test: list[int] = list(board)
    test[idx] = player

    if any(all(test[i] == player for i in line) for line in WINNING_LINES):
        return 0.9

    test2: list[int] = list(board)
    test2[idx] = opp
    if any(all(test2[i] == opp for i in line) for line in WINNING_LINES):
        return 0.8

    # Creates two or more winning threats (fork)
    threats = sum(
        1 for line in WINNING_LINES
        if sum(test[i] == player for i in line) == 2
        and sum(test[i] == 0 for i in line) == 1
    )
    if threats >= 2:
        return 0.6

    if idx == 4:
        return 0.3
    if idx in (0, 2, 6, 8):
        return 0.2
    return 0.1


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

        legal = position.legal_plies
        scores = [_score_move(board, me, p.square - 1) for p in legal]
        total = sum(scores)
        policy = {str(p): s / total for p, s in zip(legal, scores, strict=True)}

        return PositionEvaluation(value=max(-1.0, min(1.0, score)), policy=policy)

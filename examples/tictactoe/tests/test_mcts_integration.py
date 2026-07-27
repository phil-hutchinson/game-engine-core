"""Integration tests: the MCTS engine playing a real TicTacToe board.

This is the composition a consumer of the library relies on — their position class
driven by the shipped engine — so it is tested here rather than in the package suite.

The win-in-one case is a fast smoke test but a weak strength signal: a search
weak enough to blunder in one position out of eight still passes it. The
full-game test below is the strength check, scoring every engine decision against
a negamax solver rather than only the outcome.
"""

from functools import cache

import pytest

from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import Board, TicTacToePosition
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator

TicTacToeMCTSEngine = MCTSEngine[
    TicTacToePly, TicTacToePosition, NullEvaluator[TicTacToePly, TicTacToePosition]
]

# Above the strength cliff with margin: the engine blunders against perfect play
# at 500 iterations and is clean from 1000 up (measured over every position
# reachable in five plies, story 21 step 3).
STRENGTH_ITERATIONS = 2000


def _engine(iterations: int) -> TicTacToeMCTSEngine:
    return MCTSEngine(evaluator=NullEvaluator(), iterations=iterations)


@cache
def _negamax(board: Board, player: int) -> int:
    """Game-theoretic value of the position, from the mover's perspective."""
    position = TicTacToePosition(board, player)  # type: ignore[arg-type]
    if position.outcome is not None:
        return position.outcome
    return max(
        -_negamax(position.apply_ply(ply).board, -player)
        for ply in position.legal_plies
    )


def _perfect_plies(position: TicTacToePosition) -> set[str]:
    """Every ply that preserves the value of the position — the non-blunders."""
    best = _negamax(position.board, position.active_player_id)
    return {
        str(ply)
        for ply in position.legal_plies
        if -_negamax(position.apply_ply(ply).board, -position.active_player_id) == best
    }


def test_engine_finds_the_win_in_one() -> None:
    # P1 to move with 1, 2 on the top row (P2 on 4, 5): square 3 wins
    # immediately, while most alternatives let P2 play 6 and complete the middle
    # row, so the search signal for square 3 is unambiguous.
    position = TicTacToePosition.new_game()
    for square in (1, 4, 2, 5):
        position = position.apply_ply(TicTacToePly(square))

    chosen = _engine(iterations=500).select_ply(position)

    assert chosen.square == 3
    # And the ply does what the engine thinks it does: the game ends, read as a
    # loss by the player now facing the completed line.
    assert position.apply_ply(chosen).outcome == -1


@pytest.mark.parametrize("engine_seat", [1, -1])
def test_engine_never_blunders_against_perfect_play(engine_seat: int) -> None:
    """The real strength check: a correct TicTacToe engine cannot be beaten.

    Every engine decision is scored against the solver, not just the result, so
    a search that stumbles into a draw by luck still fails. Both seats are
    covered because moving first and second exercise different search shapes:
    P1 must find a line that keeps the draw available, P2 must refute one.
    """
    engine = _engine(STRENGTH_ITERATIONS)
    position = TicTacToePosition.new_game()

    while position.outcome is None:
        if position.active_player_id == engine_seat:
            ply = engine.select_ply(position)
            assert str(ply) in _perfect_plies(position), (
                f"blundered with {ply} on board {position.board}; "
                f"value-preserving plies were {sorted(_perfect_plies(position))}"
            )
        else:
            # A perfect opponent, so the engine is never handed a free win.
            ply = next(
                p for p in position.legal_plies if str(p) in _perfect_plies(position)
            )
        new_position = position.apply_ply(ply)
        engine.observe_ply(position, ply, new_position)
        position = new_position

    # Perfect play by both sides draws. Stated in absolute terms: the outcome is
    # relative to the player to move at the final position, and 0 is 0 either way.
    assert position.outcome == 0

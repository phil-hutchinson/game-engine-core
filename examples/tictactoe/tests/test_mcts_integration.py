"""Integration tests: the MCTS engine playing a real TicTacToe board.

This is the composition a consumer of the library relies on — their position class
driven by the shipped engine — so it is tested here rather than in the package suite.

The win-in-one case is a fast smoke test but a weak strength signal: a search
weak enough to blunder in one position out of eight still passes it. The
full-game test below is the strength check, playing out all 72 two-ply openings
and scoring every engine decision against a negamax solver rather than only the
outcome — roughly 210 decisions, over openings that are drawn and openings the
engine must convert to a win.
"""

from functools import cache
from itertools import permutations
from typing import Literal

import pytest

from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import Board, TicTacToePosition
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator

TicTacToeMCTSEngine = MCTSEngine[
    TicTacToePly, TicTacToePosition, NullEvaluator[TicTacToePly, TicTacToePosition]
]

# Above the strength cliff. Measured over every position reachable in five plies
# (story 21 step 3), blunders persist at 1000 iterations (0.3%) and 2000 is the
# first clean budget. The opening sweep below is easier — it is clean from 1000
# up and blunders 8 times in 228 decisions at 500 — so 2000 also carries margin.
STRENGTH_ITERATIONS = 2000

# Every distinct position two plies in: nine first plies by eight replies. Each
# is a separate parametrised game, so the sweep covers openings that are still
# drawn and openings where the reply lost and the engine has to convert.
TWO_PLY_OPENINGS = list(permutations(range(1, 10), 2))


def _engine(iterations: int) -> TicTacToeMCTSEngine:
    return MCTSEngine(evaluator=NullEvaluator(), iterations=iterations)


def _opponent(player: Literal[1, -1]) -> Literal[1, -1]:
    """Negation that keeps the seat type, which `-player` would widen to int."""
    return -1 if player == 1 else 1


@cache
def _negamax(board: Board, player: Literal[1, -1]) -> int:
    """Game-theoretic value of the position, from the mover's perspective."""
    position = TicTacToePosition(board, player)
    if position.outcome is not None:
        return position.outcome
    return max(
        -_negamax(position.apply_ply(ply).board, _opponent(player))
        for ply in position.legal_plies
    )


def _perfect_plies(position: TicTacToePosition) -> set[str]:
    """Every ply that preserves the value of the position — the non-blunders."""
    mover = position.active_player_id
    best = _negamax(position.board, mover)
    return {
        str(ply)
        for ply in position.legal_plies
        if -_negamax(position.apply_ply(ply).board, _opponent(mover)) == best
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


@pytest.mark.parametrize(("first_ply", "reply"), TWO_PLY_OPENINGS)
def test_engine_never_blunders_against_perfect_play(first_ply: int, reply: int) -> None:
    """The real strength check: a correct TicTacToe engine cannot be beaten.

    Every engine decision is scored against the solver, not just the result, so
    a search that stumbles into the right outcome by luck still fails. Playing
    from all 72 two-ply openings rather than deepening one line is what gives
    this coverage: a single game scores about three decisions along one fixed
    path, which a strength regression can easily miss, while the sweep scores
    every decision the engine makes from every distinct position two plies in.
    """
    engine = _engine(STRENGTH_ITERATIONS)
    opening = TicTacToePosition.new_game()
    for square in (first_ply, reply):
        opening = opening.apply_ply(TicTacToePly(square))

    # The engine takes the seat to move at the opening — always P1, since two
    # plies have been played — and meets a perfect opponent, so it is never
    # handed a free win and must find the value of the opening itself.
    engine_seat = opening.active_player_id
    position = opening
    while position.outcome is None:
        if position.active_player_id == engine_seat:
            ply = engine.select_ply(position)
            assert str(ply) in _perfect_plies(position), (
                f"blundered with {ply} on board {position.board}; "
                f"value-preserving plies were {sorted(_perfect_plies(position))}"
            )
        else:
            ply = next(
                p for p in position.legal_plies if str(p) in _perfect_plies(position)
            )
        new_position = position.apply_ply(ply)
        engine.observe_ply(position, ply, new_position)
        position = new_position

    # Neither side blundered, so the game ends on the opening's solved value.
    # `outcome` is relative to the player to move at the final position, so it
    # is restated from the engine's seat before comparing.
    assert position.outcome * position.active_player_id * engine_seat == _negamax(
        opening.board, engine_seat
    )

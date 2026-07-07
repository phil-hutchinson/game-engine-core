"""Integration test: the MCTS engine finds a winning ply on a real TicTacToe board.

This is the composition a consumer of the library relies on — their position class
driven by the shipped engine — so it is tested here rather than in the package suite.
"""

from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import TicTacToePosition
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator


def test_engine_finds_the_win_in_one() -> None:
    # P1 to move with 1, 2 on the top row (P2 on 4, 5): square 3 wins
    # immediately, while most alternatives let P2 play 6 and complete the middle
    # row, so the search signal for square 3 is unambiguous.
    position = TicTacToePosition.new_game()
    for square in (1, 4, 2, 5):
        position = position.apply_ply(TicTacToePly(square))

    engine: MCTSEngine[
        TicTacToePly, TicTacToePosition, NullEvaluator[TicTacToePly, TicTacToePosition]
    ] = MCTSEngine(evaluator=NullEvaluator(), iterations=500)
    chosen = engine.select_ply(position)

    assert chosen.square == 3
    # And the ply does what the engine thinks it does: the game ends, read as a
    # loss by the player now facing the completed line.
    assert position.apply_ply(chosen).outcome == -1

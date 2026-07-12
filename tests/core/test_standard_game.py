"""StandardGame game-flow tests, centred on the relative-to-absolute outcome conversion.

The terminal position's outcome is current-player-relative; GameResult.outcome is
absolute (1 = player 1 won). Forced-line Nim (takes of exactly 1) makes the winner a
parity function of the starting pile, so both conversion directions can be asserted.
"""

from game_engine_core.game.standard_game import StandardGame
from game_engine_core.models.game_result import GameResult

from .nim_fixture import FirstLegalPlayer, NimPly, NimPosition, NimStubUI


def _run_forced_line_game(pile: int) -> GameResult:
    game: StandardGame[NimPly, NimPosition] = StandardGame(
        initial_position=NimPosition(pile=pile, takes=(1,)),
        players={1: FirstLegalPlayer("P1"), -1: FirstLegalPlayer("P2")},
        game_ui=NimStubUI(),
    )
    return game.run()


def test_player_one_win_reports_absolute_outcome_1() -> None:
    # Pile 3, one token per ply: P1, P2, P1 — player 1 takes the last token.
    result = _run_forced_line_game(pile=3)
    assert result.outcome == 1


def test_player_two_win_reports_absolute_outcome_minus_1() -> None:
    # Pile 4: P1, P2, P1, P2 — player 2 takes the last token.
    result = _run_forced_line_game(pile=4)
    assert result.outcome == -1


def test_result_reason_comes_from_the_terminal_position() -> None:
    result = _run_forced_line_game(pile=3)
    assert result.result_reason == "Last token taken"


def test_game_log_records_every_ply() -> None:
    result = _run_forced_line_game(pile=3)
    assert result.opening_board == "pile=3"
    assert list(result.game_log) == [
        ("1", "pile=2"),
        ("1", "pile=1"),
        ("1", "pile=0"),
    ]

"""StandardGame game-flow tests, centred on the relative-to-absolute outcome conversion.

The terminal position's outcome is current-player-relative; GameResult.outcome is
absolute (1 = player 1 won). Forced-line Nim (takes of exactly 1) makes the winner a
parity function of the starting pile, so both conversion directions can be asserted.
"""

from typing import Literal

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator
from game_engine_core.game.standard_game import StandardGame
from game_engine_core.models.game_result import GameResult
from game_engine_core.players.ai_player import AIPlayer
from game_engine_core.protocols.player import Player

from .nim_fixture import FirstLegalPlayer, NimPly, NimPosition, NimStubUI


def _run_forced_line_game(pile: int) -> GameResult:
    game: StandardGame[NimPly, NimPosition] = StandardGame(
        initial_position=NimPosition(pile=pile, takes=(1,)),
        players={1: FirstLegalPlayer("P1"), -1: FirstLegalPlayer("P2")},
        game_logging=NimStubUI(),
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


class ContextAnnotatingLogging(NimStubUI):
    """ply_annotation distinguishable from str(ply), built from both positions."""

    def ply_annotation(
        self, from_position: NimPosition, ply: NimPly, to_position: NimPosition
    ) -> str:
        return f"take {ply.take}: {from_position.pile}->{to_position.pile}"


def test_game_log_annotations_come_from_ply_annotation_not_str() -> None:
    game: StandardGame[NimPly, NimPosition] = StandardGame(
        initial_position=NimPosition(pile=2, takes=(1,)),
        players={1: FirstLegalPlayer("P1"), -1: FirstLegalPlayer("P2")},
        game_logging=ContextAnnotatingLogging(),
    )
    result = game.run()
    assert [annotation for annotation, _ in result.game_log] == [
        "take 1: 2->1",
        "take 1: 1->0",
    ]


def test_reused_engine_players_stay_isolated_across_games() -> None:
    # As a tournament does, the same AIPlayer/engine instances play a second
    # game. Without StandardGame.run resetting the engines, the retained tree
    # from game one's terminal (pile=0) position would still be the root going
    # into game two's pile=5 opening, and select_ply would hand back a ply
    # that's illegal for the real position - raising here instead of playing on.
    engine_one: MCTSEngine[NimPly, NimPosition, NullEvaluator[NimPly, NimPosition]] = MCTSEngine(
        evaluator=NullEvaluator(), iterations=20
    )
    engine_two: MCTSEngine[NimPly, NimPosition, NullEvaluator[NimPly, NimPosition]] = MCTSEngine(
        evaluator=NullEvaluator(), iterations=20
    )
    players: dict[Literal[1, -1], Player[NimPly, NimPosition]] = {
        1: AIPlayer(engine_one, "P1"),
        -1: AIPlayer(engine_two, "P2"),
    }

    def _run_game() -> GameResult:
        game: StandardGame[NimPly, NimPosition] = StandardGame(
            initial_position=NimPosition(pile=5),
            players=players,
            game_logging=NimStubUI(),
        )
        return game.run()

    first_result = _run_game()
    second_result = _run_game()

    assert first_result.outcome in (1, -1)
    assert second_result.outcome in (1, -1)

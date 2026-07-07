"""Reporting tests: a small Nim tournament written to disk and read back."""

import json
from pathlib import Path

import pytest

from game_engine_core.tournament.reporting import write_tournament_report
from game_engine_core.tournament.tournament import Tournament
from game_engine_core.tournament.tournament_result import TournamentResult

from .nim_fixture import NimPly, NimPosition
from .test_tournament import FirstLegalPlayer, NimStubUI


@pytest.fixture
def result() -> TournamentResult:
    tournament: Tournament[NimPly, NimPosition] = Tournament(
        players=[FirstLegalPlayer(name) for name in ["A", "B", "C"]],
        position_factory=lambda: NimPosition(pile=3, takes=(1,)),
        game_ui=NimStubUI(),
        games_per_pairing=2,
    )
    return tournament.run()


def test_standings_file_names_every_player(
    result: TournamentResult, tmp_path: Path
) -> None:
    write_tournament_report(result, tmp_path / "report")
    standings = (tmp_path / "report" / "standings.md").read_text()
    for name in ["A", "B", "C"]:
        assert name in standings
    assert "## Standings" in standings
    assert "## Cross-table" in standings


def test_one_parseable_json_per_game_matching_its_record(
    result: TournamentResult, tmp_path: Path
) -> None:
    write_tournament_report(result, tmp_path / "report")
    game_files = sorted((tmp_path / "report" / "games").iterdir())
    assert len(game_files) == len(result.records)

    for game_file, record in zip(game_files, result.records, strict=True):
        game = json.loads(game_file.read_text())
        assert game["side_one"] == record.players[1]
        assert game["side_two"] == record.players[-1]
        assert game["outcome"] == record.result.outcome
        assert [(p["ply"], p["board"]) for p in game["plies"]] == list(
            record.result.game_log
        )


def test_summary_is_printed_to_console(
    result: TournamentResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_tournament_report(result, tmp_path / "report")
    captured = capsys.readouterr().out
    assert "## Standings" in captured
    for name in ["A", "B", "C"]:
        assert name in captured

"""Reporting tests: a small Nim tournament written to disk and read back."""

import json
from pathlib import Path

import pytest

from game_engine_core.models.game_result import GameResult
from game_engine_core.tournament.cross_table import compute_cross_table
from game_engine_core.tournament.game_record import GameRecord
from game_engine_core.tournament.reporting import (
    render_summary,
    write_tournament_report,
)
from game_engine_core.tournament.standings import compute_standings
from game_engine_core.tournament.tournament import Tournament
from game_engine_core.tournament.tournament_result import TournamentResult

from .nim_fixture import FirstLegalPlayer, NimPly, NimPosition, NimStubUI


@pytest.fixture
def result() -> TournamentResult:
    tournament: Tournament[NimPly, NimPosition] = Tournament(
        players=[FirstLegalPlayer(name) for name in ["A", "B", "C"]],
        position_factory=lambda: NimPosition(pile=3, takes=(1,)),
        game_logging=NimStubUI(),
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


def test_rejects_an_existing_output_folder(
    result: TournamentResult, tmp_path: Path
) -> None:
    # One fresh folder per tournament: reuse would mix reports from different runs.
    write_tournament_report(result, tmp_path / "report")
    with pytest.raises(FileExistsError, match="fresh folder"):
        write_tournament_report(result, tmp_path / "report")


def test_notes_are_rendered_under_the_title(
    result: TournamentResult, tmp_path: Path
) -> None:
    write_tournament_report(
        result, tmp_path / "report", notes=["Checkpoints: weights/runs/x"]
    )
    standings = (tmp_path / "report" / "standings.md").read_text()
    assert standings.index("Checkpoints: weights/runs/x") < standings.index("## Standings")


def test_player_names_are_escaped_in_markdown_tables() -> None:
    # Names are arbitrary Player strings: pipes and newlines must not be able
    # to break the table layout. JSON game files keep names verbatim.
    names = ["pipe|name", "line\nname"]
    record = GameRecord(
        players={1: names[0], -1: names[1]},
        result=GameResult(outcome=1, result_reason="", opening_board="", game_log=[]),
    )
    summary = render_summary(
        TournamentResult(
            player_names=names,
            records=[record],
            standings=compute_standings(names, [record]),
            cross_table=compute_cross_table(names, [record]),
        )
    )
    assert "pipe\\|name" in summary
    assert "line name" in summary
    # Every table line still has a well-formed cell count: 8 pipes for the
    # 6-column standings rows, none of them introduced by a player name.
    standings_rows = [line for line in summary.splitlines() if line.startswith("| 1 ")]
    assert standings_rows and standings_rows[0].count("|") - standings_rows[0].count("\\|") == 7

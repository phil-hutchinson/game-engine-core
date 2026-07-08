"""Aggregation tests over fabricated GameRecords.

Records are built by hand rather than played: aggregation is pure, and
fabrication covers cases a real fixture game cannot (Nim has no draws).
"""

from typing import Literal

import pytest

from game_engine_core.models.game_result import GameResult
from game_engine_core.tournament.cross_table import compute_cross_table
from game_engine_core.tournament.game_record import GameRecord
from game_engine_core.tournament.standings import compute_standings


def _record(side_one: str, side_two: str, outcome: Literal[1, 0, -1]) -> GameRecord:
    return GameRecord(
        players={1: side_one, -1: side_two},
        result=GameResult(outcome=outcome, opening_board="", game_log=[]),
    )


def test_points_for_side_covers_win_draw_loss() -> None:
    win_for_side_one = _record("A", "B", outcome=1)
    assert win_for_side_one.points_for_side(1) == 1.0
    assert win_for_side_one.points_for_side(-1) == 0.0

    draw = _record("A", "B", outcome=0)
    assert draw.points_for_side(1) == 0.5
    assert draw.points_for_side(-1) == 0.5


def test_relative_outcome_flips_with_the_side() -> None:
    win_for_side_two = _record("A", "B", outcome=-1)
    assert win_for_side_two.relative_outcome_for_side(-1) == 1
    assert win_for_side_two.relative_outcome_for_side(1) == -1
    assert _record("A", "B", outcome=0).relative_outcome_for_side(1) == 0


def test_standings_tallies_across_sides() -> None:
    # A beats B once holding side 1, once holding side -1, and draws once.
    records = [
        _record("A", "B", outcome=1),
        _record("B", "A", outcome=-1),
        _record("A", "B", outcome=0),
    ]
    standings = compute_standings(["A", "B"], records)

    entry_a, entry_b = standings
    assert (entry_a.player_name, entry_a.wins, entry_a.draws, entry_a.losses) == ("A", 2, 1, 0)
    assert entry_a.points == 2.5
    assert entry_a.games_played == 3
    assert (entry_b.player_name, entry_b.wins, entry_b.draws, entry_b.losses) == ("B", 0, 1, 2)
    assert entry_b.points == 0.5


def test_standings_sorted_by_points_then_wins_then_name() -> None:
    # A: 2 wins (2.0). C: 1 win (1.0). B: 2 draws (1.0) — C outranks B on wins.
    records = [
        _record("A", "B", outcome=1),
        _record("A", "C", outcome=1),
        _record("C", "D", outcome=1),
        _record("B", "D", outcome=0),
        _record("D", "B", outcome=0),
    ]
    standings = compute_standings(["A", "B", "C", "D"], records)
    assert [entry.player_name for entry in standings] == ["A", "C", "B", "D"]


def test_standings_includes_players_without_games() -> None:
    standings = compute_standings(["A", "B", "C"], [_record("A", "B", outcome=1)])
    entry_c = next(entry for entry in standings if entry.player_name == "C")
    assert (entry_c.wins, entry_c.draws, entry_c.losses) == (0, 0, 0)
    assert entry_c.points == 0.0


def test_standings_rejects_unknown_player_in_record() -> None:
    with pytest.raises(ValueError, match="unknown player"):
        compute_standings(["A", "B"], [_record("A", "X", outcome=1)])


def test_standings_rejects_duplicate_player_names() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        compute_standings(["A", "A"], [])


def test_cross_table_accumulates_points_per_opponent() -> None:
    # A vs B: A wins on each side plus one draw -> A 2.5, B 0.5.
    records = [
        _record("A", "B", outcome=1),
        _record("B", "A", outcome=-1),
        _record("A", "B", outcome=0),
    ]
    table = compute_cross_table(["A", "B", "C"], records)

    assert table["A"]["B"] == 2.5
    assert table["B"]["A"] == 0.5
    # Complete grid: unplayed pairings are present at 0.0, self cells absent.
    assert table["A"]["C"] == 0.0
    assert table["C"]["B"] == 0.0
    assert "A" not in table["A"]


def test_cross_table_opposing_cells_sum_to_games_played() -> None:
    records = [
        _record("A", "B", outcome=1),
        _record("B", "A", outcome=1),
        _record("A", "B", outcome=0),
        _record("B", "A", outcome=0),
    ]
    table = compute_cross_table(["A", "B"], records)
    assert table["A"]["B"] + table["B"]["A"] == len(records)


def test_cross_table_rejects_unknown_player_in_record() -> None:
    with pytest.raises(ValueError, match="unknown player"):
        compute_cross_table(["A", "B"], [_record("X", "B", outcome=1)])

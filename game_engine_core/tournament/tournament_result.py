from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .game_record import GameRecord
from .standings import StandingsEntry


@dataclass(frozen=True)
class TournamentResult:
    """A completed tournament: every game played plus its aggregations."""

    player_names: Sequence[str]
    """All entrants, in the order they were supplied to the tournament."""

    records: Sequence[GameRecord]
    """Every game played, in playing order."""

    standings: Sequence[StandingsEntry]
    """Sorted standings (points, then wins, then name)."""

    cross_table: Mapping[str, Mapping[str, float]]
    """Points scored by each row player against each column player."""

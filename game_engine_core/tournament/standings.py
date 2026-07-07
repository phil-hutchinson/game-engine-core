"""Standings aggregation: per-player win/draw/loss tallies over game records.

Pure record-in/tables-out so it can be tested (and reused) without running
games. The player list is supplied explicitly rather than inferred from the
records so that players who have not scored — or not played at all — still
appear, and so the caller controls identity: a record naming an unknown player
is an error, not a new row.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .game_record import GameRecord

_SIDES: tuple[Literal[1, -1], Literal[1, -1]] = (1, -1)


@dataclass(frozen=True)
class StandingsEntry:
    player_name: str
    wins: int
    draws: int
    losses: int

    @property
    def points(self) -> float:
        """Tournament points: 1 per win, 0.5 per draw."""
        return self.wins + 0.5 * self.draws

    @property
    def games_played(self) -> int:
        return self.wins + self.draws + self.losses


def compute_standings(
    player_names: Sequence[str], records: Iterable[GameRecord]
) -> list[StandingsEntry]:
    """Tally records into standings sorted by points, then wins, then name.

    Wins and name break points ties: more decisive results rank higher, and the
    name comparison makes the ordering deterministic for reporting.
    """
    if len(set(player_names)) != len(player_names):
        raise ValueError(f"Duplicate player names in {player_names}")

    tallies: dict[str, dict[float, int]] = {
        name: {1.0: 0, 0.5: 0, 0.0: 0} for name in player_names
    }
    for record in records:
        for side in _SIDES:
            name = record.players[side]
            if name not in tallies:
                raise ValueError(f"Record names unknown player {name!r}")
            tallies[name][record.points_for_side(side)] += 1

    entries = [
        StandingsEntry(
            player_name=name,
            wins=tally[1.0],
            draws=tally[0.5],
            losses=tally[0.0],
        )
        for name, tally in tallies.items()
    ]
    entries.sort(key=lambda entry: (-entry.points, -entry.wins, entry.player_name))
    return entries

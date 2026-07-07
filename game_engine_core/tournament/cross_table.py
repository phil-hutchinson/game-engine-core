"""Cross-table aggregation: each player's aggregate score against each opponent.

The table is a complete grid over the supplied player list (0.0 for pairings
with no games, no self cells), so reports can render it without special cases.
Cells are points, not game counts: opposing cells for one pairing always sum to
the number of games played between the two, since every game awards exactly one
point split between its players.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from .game_record import GameRecord

_SIDES: tuple[Literal[1, -1], Literal[1, -1]] = (1, -1)


def compute_cross_table(
    player_names: Sequence[str], records: Iterable[GameRecord]
) -> dict[str, dict[str, float]]:
    """Points scored by each row player against each column player."""
    if len(set(player_names)) != len(player_names):
        raise ValueError(f"Duplicate player names in {player_names}")

    table: dict[str, dict[str, float]] = {
        row: {col: 0.0 for col in player_names if col != row} for row in player_names
    }
    for record in records:
        for side in _SIDES:
            name = record.players[side]
            opponent = record.players[-side]
            if name not in table or opponent not in table:
                raise ValueError(
                    f"Record names unknown player {name!r} or {opponent!r}"
                )
            table[name][opponent] += record.points_for_side(side)

    return table

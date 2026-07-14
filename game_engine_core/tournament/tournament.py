"""Round-robin tournament runner over the core game/player protocols.

Game-agnostic and ML-agnostic: anything implementing Player can enter, so
training checkpoints, classical engines, and scripted baselines mix freely.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any

from ..game.standard_game import StandardGame
from ..protocols.game_logging import GameLogging
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition
from ..protocols.player import Player
from .cross_table import compute_cross_table
from .game_record import GameRecord
from .standings import compute_standings
from .tournament_result import TournamentResult


class Tournament[TPly: GamePly, TPosition: GamePosition[Any]]:
    """Round-robin: every pair of players meets games_per_pairing times.

    Side 1 (first to move) alternates within each pairing so first-move
    advantage cancels out — exactly with an even games_per_pairing, to within
    one game with an odd one. Games run headless (no GameUI); the GameLogging
    feeds the game logs.

    position_factory is called once per game with the two participants in side
    order — the first argument holds side 1 (first to move), the second side
    -1 — with the within-pairing alternation already applied. This is the seam
    for games whose starting position depends on per-player state (the factory
    may downcast to its concrete player type); games that don't need it ignore
    the arguments, e.g. ``lambda p1, p2: TicTacToePosition.new_game()``.
    """

    def __init__(
        self,
        players: Sequence[Player[TPly, TPosition]],
        position_factory: Callable[
            [Player[TPly, TPosition], Player[TPly, TPosition]], TPosition
        ],
        game_logging: GameLogging[TPly, TPosition],
        games_per_pairing: int = 2,
    ):
        if len(players) < 2:
            raise ValueError("A tournament needs at least 2 players")
        names = [player.name for player in players]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate player names in {names}")
        if games_per_pairing < 1:
            raise ValueError(f"games_per_pairing must be >= 1, got {games_per_pairing}")
        self._players = players
        self._position_factory = position_factory
        self._game_logging = game_logging
        self._games_per_pairing = games_per_pairing

    def run(self) -> TournamentResult:
        records: list[GameRecord] = []
        for first, second in combinations(self._players, 2):
            for game_index in range(self._games_per_pairing):
                first.reset()
                second.reset()
                # Alternate who moves first within the pairing.
                if game_index % 2 == 0:
                    records.append(self._play_game(first, second))
                else:
                    records.append(self._play_game(second, first))
        player_names = [player.name for player in self._players]
        return TournamentResult(
            player_names=player_names,
            records=records,
            standings=compute_standings(player_names, records),
            cross_table=compute_cross_table(player_names, records),
        )

    def _play_game(
        self,
        side_one: Player[TPly, TPosition],
        side_other: Player[TPly, TPosition],
    ) -> GameRecord:
        game: StandardGame[TPly, TPosition] = StandardGame(
            initial_position=self._position_factory(side_one, side_other),
            players={1: side_one, -1: side_other},
            game_logging=self._game_logging,
            game_ui=None,
        )
        return GameRecord(
            players={1: side_one.name, -1: side_other.name},
            result=game.run(),
        )

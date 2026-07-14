"""Minimal fixture game for testing the core protocols and engines.

Subtraction Nim: a pile of tokens, each ply removes one of the permitted take
sizes, and taking the last token wins. Players strictly alternate, matching the
framework's current alternation assumption (general-cleanup finding #3).

Why Nim: its game theory is trivial to state in a test. With takes of 1-2, a
pile that is a multiple of 3 is lost for the player to move; otherwise taking
``pile % 3`` is the winning ply. With takes restricted to ``(1,)`` every
position has exactly one legal ply, so a whole game is a forced line whose
winner is a parity function of the starting pile — used where a test needs a
fully deterministic game with a known outcome.
"""

from __future__ import annotations

from typing import Literal

from game_engine_core.protocols.game_ply import GamePly
from game_engine_core.protocols.game_position import GamePosition


class NimPly(GamePly):

    def __init__(self, take: int):
        self._take = take

    @property
    def take(self) -> int:
        """Number of tokens this ply removes from the pile."""
        return self._take

    def __str__(self) -> str:
        return str(self._take)


class NimPosition(GamePosition[NimPly]):

    def __init__(
        self,
        pile: int,
        active_player_id: Literal[1, -1] = 1,
        takes: tuple[int, ...] = (1, 2),
    ):
        # Require a take of 1 so no reachable non-empty pile can become a
        # dead end (a position with no legal plies but ``outcome is None``),
        # which the framework treats as impossible.
        if 1 not in takes:
            raise ValueError(f"takes must include 1 to avoid dead ends, got {takes}")
        self._pile = pile
        self._active_player_id: Literal[1, -1] = active_player_id
        self._takes = takes

    @property
    def pile(self) -> int:
        return self._pile

    @property
    def active_player_id(self) -> Literal[1, -1]:
        return self._active_player_id

    @property
    def outcome(self) -> Literal[1, 0, -1] | None:
        # The previous player took the last token and won, so an empty pile is
        # a loss from the perspective of the player now to move. Nim has no draws.
        if self._pile == 0:
            return -1
        return None

    @property
    def outcome_reason(self) -> str | None:
        # Taking the last token is Nim's only way to end a game.
        if self._pile == 0:
            return "Last token taken"
        return None

    @property
    def legal_plies(self) -> list[NimPly]:
        return [NimPly(take) for take in self._takes if take <= self._pile]

    def apply_ply(self, ply: NimPly) -> NimPosition:
        if ply.take not in self._takes or ply.take > self._pile:
            raise ValueError(
                f"Illegal take {ply.take} (pile {self._pile}, permitted takes {self._takes})"
            )
        return NimPosition(
            pile=self._pile - ply.take,
            active_player_id=-self._active_player_id,
            takes=self._takes,
        )


class NimStubUI:
    """Minimal GameUI + GameLogging over the pile count; no human input in tests."""

    def text_board(self, position: NimPosition) -> str:
        return f"pile={position.pile}"

    def ply_annotation(
        self, from_position: NimPosition, ply: NimPly, to_position: NimPosition
    ) -> str:
        return str(ply)

    def render_board(self, position: NimPosition) -> None:
        pass

    def get_next_ply(self, position: NimPosition) -> NimPly:
        raise NotImplementedError("Tests use scripted players only")


class FirstLegalPlayer:
    """Scripted Player: always takes the first legal ply."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def render_before_ply(self) -> bool:
        return False

    def select_ply(self, position: NimPosition) -> NimPly:
        return position.legal_plies[0]

    def observe_ply(self, position: NimPosition, ply: NimPly, new_position: NimPosition) -> None:
        pass

    def reset(self) -> None:
        pass

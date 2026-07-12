"""Tournament runner tests over forced-line Nim.

With takes of exactly 1, a game's winner is a parity function of the starting
pile: pile 3 is a win for the player holding side 1, pile 4 a win for side -1.
That makes outcome attribution — mapping the absolute GameResult back to
whichever player held each side — exactly checkable.
"""

import pytest

from game_engine_core.tournament.tournament import Tournament

from .nim_fixture import FirstLegalPlayer, NimPly, NimPosition


class LoggingOnly:
    """GameLogging with no UI members: tournaments are headless by construction,
    so an object with nothing to render (or prompt with) must be sufficient."""

    def text_board(self, position: NimPosition) -> str:
        return f"pile={position.pile}"

    def ply_annotation(
        self, from_position: NimPosition, ply: NimPly, to_position: NimPosition
    ) -> str:
        return str(ply)


def _players(*names: str) -> list[FirstLegalPlayer]:
    return [FirstLegalPlayer(name) for name in names]


def _tournament(
    names: list[str], pile: int, games_per_pairing: int
) -> Tournament[NimPly, NimPosition]:
    return Tournament(
        players=_players(*names),
        position_factory=lambda: NimPosition(pile=pile, takes=(1,)),
        game_logging=LoggingOnly(),
        games_per_pairing=games_per_pairing,
    )


def test_every_pairing_plays_the_requested_games() -> None:
    result = _tournament(["A", "B", "C"], pile=3, games_per_pairing=2).run()
    assert len(result.records) == 6  # 3 pairings x 2 games
    assert result.player_names == ["A", "B", "C"]


def test_sides_alternate_within_a_pairing() -> None:
    result = _tournament(["A", "B"], pile=3, games_per_pairing=4).run()
    assert [record.players[1] for record in result.records] == ["A", "B", "A", "B"]


def test_odd_games_per_pairing_splits_sides_within_one() -> None:
    result = _tournament(["A", "B"], pile=3, games_per_pairing=3).run()
    side_one_counts = [record.players[1] for record in result.records]
    assert side_one_counts.count("A") == 2
    assert side_one_counts.count("B") == 1


def test_side_one_wins_are_attributed_to_the_holder() -> None:
    # Pile 3: whoever moves first takes the last token. With 2 games per
    # pairing each player wins exactly the game they held side 1 in.
    result = _tournament(["A", "B", "C"], pile=3, games_per_pairing=2).run()
    for record in result.records:
        assert record.points_for_side(1) == 1.0
    for entry in result.standings:
        assert (entry.wins, entry.draws, entry.losses) == (2, 0, 2)
    assert result.cross_table["A"]["B"] == 1.0
    assert result.cross_table["B"]["A"] == 1.0


def test_side_two_wins_are_attributed_to_the_holder() -> None:
    # Pile 4: the second player takes the last token, so the winner is
    # whichever player did NOT hold side 1 — the sign-flip case.
    result = _tournament(["A", "B"], pile=4, games_per_pairing=2).run()
    for record in result.records:
        assert record.points_for_side(-1) == 1.0
    entry_a = next(e for e in result.standings if e.player_name == "A")
    assert (entry_a.wins, entry_a.losses) == (1, 1)


def test_each_game_starts_from_a_fresh_position() -> None:
    calls = 0

    def counting_factory() -> NimPosition:
        nonlocal calls
        calls += 1
        return NimPosition(pile=3, takes=(1,))

    Tournament(
        players=_players("A", "B", "C"),
        position_factory=counting_factory,
        game_logging=LoggingOnly(),
        games_per_pairing=2,
    ).run()
    assert calls == 6


def test_rejects_duplicate_player_names() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        _tournament(["A", "A"], pile=3, games_per_pairing=2)


def test_rejects_fewer_than_two_players() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _tournament(["A"], pile=3, games_per_pairing=2)


def test_rejects_nonpositive_games_per_pairing() -> None:
    with pytest.raises(ValueError, match="games_per_pairing"):
        _tournament(["A", "B"], pile=3, games_per_pairing=0)

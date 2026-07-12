# Story: Widen Tournament Position Factory (issue #11)

## Goal

Widen `Tournament`'s `position_factory` from `Callable[[], TPosition]` to

```python
Callable[[Player[TPly, TPosition], Player[TPly, TPosition]], TPosition]
```

invoked once per game with the two participants **in side order** — `(side_one, side_other)`, i.e. already reflecting the alternation `Tournament` applies within each pairing. This gives games whose starting position depends on per-player state a seam to build it, without `Tournament` learning anything game-specific.

This is a **breaking signature change with no compatibility shim** (same posture as story #10): existing zero-arg factories become two-parameter callables that ignore their arguments, e.g. `lambda p1, p2: TicTacToePosition.new_game()`. One seam, not two — deliberately *not* an additional `pre_play` parameter alongside the existing factory, where one parameter would silently ignore the other.

## Motivation

The driving case is the downstream Capture the Flag repo: CtF opens with a secret, simultaneous phase-1 placement — each player commits pieces without seeing the other's choices — and only then does ply-by-ply play begin. The starting `GamePosition` is therefore a function of *both players' state*. `Tournament`'s bare zero-arg factory has no seam for that, so the downstream repo currently hand-rolls its own `play_match` (consulting each player for placements) and a `batch_runner` that reimplements the tallying `Tournament`/`TournamentResult`/standings/cross-table already provide.

With the widened factory, the CtF factory consults its own players (downcasting to its concrete player type for game-specific state — the `Player` protocol is untouched) and returns the assembled opening position; all scheduling, alternation, aggregation, and reporting come back for free.

Side order matters: side 1 alternates within a pairing, so a factory must not assume a fixed player→side mapping. Passing `(side_one, side_other)` makes the mapping explicit per game.

## Scope

- `Tournament.__init__`: `position_factory` type widened as above; `_play_game` passes `(side_one, side_other)`. Docstring documents the side-order contract and the ignore-the-arguments pattern for games that don't need it.
- In-repo callers updated to two-parameter factories: `examples/tictactoe_learning/tournament.py` and the tournament tests.
- Test coverage: the factory is called once per game and receives the players in the same side order the resulting `GameRecord` reports (i.e. alternation is observable through the factory's arguments).
- README updated if it shows a `Tournament` construction.

## Non-goals

- `SelfPlayCollector.position_factory` stays zero-arg: self-play has no `Player` objects — a single engine drives both sides — so a `(Player, Player)` signature has nothing to receive.
- No changes to `StandardGame` or the `Player` protocol: the pre-play phase is entirely the factory's business, and game-specific player state is reached by downcasting in game code, not by widening the protocol.
- No generalized multi-phase game support (e.g. mid-game auctions or negotiation phases); this story only covers building the *initial* position.

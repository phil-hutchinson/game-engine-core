# Story: SelfPlayCollector as fleet driver (issue #24)

Part of the **Fleet Play** epic (#20). Depends on #23 (fleet engine) and #22
(batch-first protocols).

## Goal

Turn `SelfPlayCollector` from a sequential game runner into the **fleet driver**.
Today `collect(n_games)` is a plain `for _ in range(n_games): self._play_game()`
loop, each game played to completion on its own before the next starts. Instead,
run all N games as one fleet: advance them in lockstep, one fleet-ply at a time,
so the underlying engine can batch every game's search into shared evaluations.

The public result is unchanged — `collect` still returns the same
`list[TrainingSample]`, with the same per-game value back-fill (the sign
alternation over each game's reversed step records). Only the driving changes.

## The fleet loop

1. **Bootstrap** N games from the position factory (bulk where the factory
   supports it), each with its own running list of step records.
2. Each fleet-turn, hand the engine the current position of every *live* game,
   in slot order, via `select_plies_with_policy(positions)` (#23), and get back
   one `(ply, policy)` per game.
3. For each live game: capture its step (batched `encode_positions` and batched
   `policy_transform`, #22), then apply its ply.
4. **Drop** any game that has reached a terminal position; back-fill its values
   and emit its `TrainingSample`s.
5. Repeat until all games are done.

Game identity is the slot index throughout: positions go in in slot order, plies
come back in slot order, and a finished game leaves the fleet, shrinking the
batch for subsequent turns.

## Motivation

The batching win only materialises end-to-end if something keeps N games'
searches in flight simultaneously — the engine can batch a wave only across the
games it is handed at once. The collector is the natural owner of that: it
already decides how many games to play and holds their per-step records for
training. Making it the driver is what connects the fleet engine to real
self-play data collection.

## The long tail

Games finish at different plies, so the live fleet drains over time and the
final turns run at low batch occupancy. The skateboard **accepts this**: it runs
a fixed fleet of N games with no midstream refill and tolerates the tail. This
keeps the driver simple and is correct; it is only suboptimal on wall-clock near
the end of a batch. The mitigations — refilling the fleet with fresh games
(epic backlog P4) and speculative batch backfill to keep the GPU saturated
through the tail (P3) — are deferred.

## Scope

- Replace the sequential `collect` loop with the fleet loop above.
- Bulk-bootstrap N starting positions from the position factory.
- Drive per-turn search through `select_plies_with_policy`, aligned by slot.
- Batch capture: `encode_positions` and the batched `policy_transform`.
- Per-game termination, value back-fill, and `TrainingSample` emission unchanged
  in meaning, adapted to games finishing at different turns.

## Non-goals

- No midstream refill / queueing of new games (epic backlog P4) — fixed fleet.
- No min-batch floor or speculative prefetch for the tail (P3).
- No change to `TrainingSample`, to the value back-fill semantics, or to
  `TrainingLoop`.
- Non-learning fleet play (tournaments / `StandardGame`) is out of scope (P6).

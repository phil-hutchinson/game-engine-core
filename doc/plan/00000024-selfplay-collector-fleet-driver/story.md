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

1. **Bootstrap** N games by calling the position factory N times — it stays
   scalar (#22 does not widen it: a game's starting position is a per-game
   concern, not a batch one), each with its own running list of step records.
2. Each fleet-turn, hand the engine the current position of every *live* game,
   in slot order, via `select_plies_for_training(positions)` (#23), and get back
   one `(ply, policy)` per game.
3. For each live game: capture its step (batched `encode_positions` and batched
   `policy_transform`, #22), then apply its ply (batched via
   `batch_ops.apply_plies`, #22).
4. **Drop** any game that has reached a terminal position (via
   `batch_ops.outcomes`, #22); back-fill its values and emit its
   `TrainingSample`s.
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
- Bootstrap N starting positions by calling the (scalar) position factory N
  times.
- Drive per-turn search through `select_plies_for_training`, aligned by slot.
- Batch capture: `encode_positions` and the batched `policy_transform`; batch
  the terminal test and ply application through `batch_ops`.
- Per-game termination, value back-fill, and `TrainingSample` emission unchanged
  in meaning, adapted to games finishing at different turns.

## Non-goals

- No midstream refill / queueing of new games (epic backlog P4) — fixed fleet.
- No min-batch floor or speculative prefetch for the tail (P3).
- No change to `TrainingSample`, to the value back-fill semantics, or to
  `TrainingLoop`.
- Non-learning fleet play (tournaments / `StandardGame`) is out of scope (P6).

## Decisions taken during implementation

These amend the story above, which left them open.

1. **One engine per `collect`, not per game.** The fleet hands *the* engine every
   live game's position at once, so a per-game engine has no meaning.
   `engine_factory` keeps its signature and is called once per `collect`. Safe
   because the training surface retains nothing between calls and holds no fleet
   state (#23's roots are call-scoped); required because lockstep depends on every
   game sharing one iteration budget. Per-game engine *configuration* is therefore
   no longer possible — but it was already incompatible with the fleet.
2. **Samples are returned in slot order, not completion order.** Games retire at
   different turns, so the natural emission order would be shortest-game-first.
   Instead each slot accumulates its own samples and the buckets are concatenated in
   slot order once the fleet drains, which keeps `collect`'s output identical to what
   playing the same games one at a time produced. This refines "emit its
   `TrainingSample`s" in the fleet loop above to mean *emit into the slot's bucket*;
   the back-fill still happens at retirement, while the outcome is in hand.
3. **One `outcomes` call per fleet-turn, at the top of the turn.** It decides which
   games leave the fleet *and* supplies the retiring game's back-fill value, so the
   sequential driver's two calls per game collapse into one per turn. It also
   enforces the engine's precondition, since a terminal slot would raise and forfeit
   every other game's search, and it makes a game whose starting position is already
   decided fall out correctly, contributing no samples.
4. **The batch is dense and compacts as games retire.** Game identity is fixed for
   the game's life, but its position within a batch shrinks as earlier games leave.
   The batch cannot carry holes: every seam on this path is typed
   `Sequence[TPosition]` with no optional. Compaction is an order-preserving filter,
   so "positions go in in slot order" still holds — the live set stays ascending in
   original slot, merely with gaps.

## Obligations on caller-supplied collaborators

- **The engine must retain no per-game state on the training path.** `MCTSEngine`
  satisfies this by construction, but `engine_factory` is typed on the concrete
  class, so a subclass relying on per-game construction to clear state would now
  carry it across the whole fleet.
- **The policy transform receives one batch spanning several different games.** A
  correct index-aligned transform is unaffected; one that assumed a call's positions
  all came from the same game would break. The evaluator and `batch_ops` need no new
  guarantee — both were already shared instances, and #23's wave already hands the
  evaluator leaves from N games in a single call.

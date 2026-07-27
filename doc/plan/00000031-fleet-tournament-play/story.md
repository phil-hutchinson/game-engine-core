# Story: Fleet tournament play (issue #31)

Part of the **Fleet Play Phase Two** epic (#25).

## Goal

Extend the fleet path beyond self-play so that **non-learning** play — running
many games for tournament / evaluation purposes — can also batch its searches
across games and saturate the GPU. Bring `StandardGame` (or a fleet counterpart)
onto the fleet engine's `select_plies` path.

## Motivation

The skateboard aims the fleet engine at self-play only: `SelfPlayCollector` is
the driver (#24), and `StandardGame.run` remains a single-game loop calling
`player.select_ply` one position at a time. But the batching win is not specific
to learning — running a tournament of many games (a new checkpoint against a
baseline, a round-robin, strength sampling) has exactly the same shape and the
same idle-GPU problem when played one game at a time. This story generalises the
fleet driver so evaluation runs benefit too.

## Scope

- A fleet driver for tournament play that advances N games in lockstep through
  the engine's `select_plies`, analogous to `SelfPlayCollector` but emitting game
  results rather than training samples.
- Reconcile with `StandardGame`'s concerns (players, logging, UI, result
  reasons) under batched play — likely a headless fleet variant, since per-move
  UI rendering does not fit lockstep advancement.
- Slot alignment and long-tail handling as in self-play (and composable with #29
  refill / #28 backfill if present).

## Non-goals

- No interactive / UI-driven play in the fleet path — fleet tournament play is
  headless by nature.
- No new engine capability — this consumes the phase-1 fleet engine (#23), it
  does not extend it.
- No change to the self-play driver (#24).

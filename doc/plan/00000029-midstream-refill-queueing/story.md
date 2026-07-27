# Story: Midstream refill / queueing (issue #29)

Part of the **Fleet Play Phase Two** epic (#25).

## Goal

Stop the fleet from draining. When a game finishes mid-fleet, replace it with a
fresh game so the live batch stays at (or near) its target size, instead of
shrinking turn by turn until the last game runs alone.

## Motivation

The skateboard runs a fixed fleet of N games and accepts the long tail: games
finish at different plies, so the live batch drains and the final turns run at
low GPU occupancy. Refill attacks that tail at its source — hold fleet size
steady by feeding new games into the slots vacated by finished ones, keeping the
batch wide until the total game budget is exhausted.

This is the more direct counterpart to speculative backfill (#28): #28 keeps a
draining fleet's GPU busy with speculative work; #29 keeps the fleet from
draining in the first place. They compose — refill holds size steady during the
main run, and backfill mops up the unavoidable tail once the game budget is spent
and no more refills are available.

## The drain-vs-refill decision

Because games in a wave finish at different plies, the driver owns a scheduling
question: when a slot frees, refill it immediately (maximising occupancy, at the
cost of more bookkeeping and a fleet whose games are at wildly different plies),
or let it drain. This story is where that policy lives — the driver
(`SelfPlayCollector`, from #24) gains a queue of pending games and a refill
policy, allocating a fresh bootstrapped game into any freed slot while games
remain in the budget.

## Scope

- Give the fleet driver a target fleet size and a backlog of games still to play.
- On a game finishing, emit its samples and bootstrap a replacement into the
  freed slot until the backlog is exhausted, then let the fleet drain.
- Keep slot alignment intact as games enter and leave.

## Non-goals

- No change to per-game search or to `TrainingSample` semantics.
- No speculative prefetch (#28); this story changes *which games* occupy the
  fleet, not what the GPU does with spare capacity.
- No cross-process / distributed queueing — a single in-process fleet.

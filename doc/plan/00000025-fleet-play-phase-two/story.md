# Epic: Fleet Play Phase Two (issue #25)

Follows the **Fleet Play** epic (#20), which delivers the skateboard: true PUCT
expansion (#21), batch-first game protocols (#22), the fleet/wave MCTS engine
(#23), and `SelfPlayCollector` as fleet driver (#24). Phase Two is the backlog
of optimisations and open decisions deliberately parked out of that skateboard.

## Goal

Take fleet self-play from "works and batches the network forward" to "saturates
the GPU and stays saturated," and settle the design questions left open by the
skateboard. Each story here stands alone and sits on top of the phase-1 engine.

## Context: where the skateboard leaves off

The skateboard batches the network forward across N games and batches the
game-facing touchpoints (#22), but it deliberately stops short in three places:

- The **tree internals** (PUCT descent, backpropagation) stay per-tree CPU
  loops, and children are still `MCTSNode` objects with a scalar `prior`.
- There is **no evaluation cache** and no cross-ply reuse — each `select_plies`
  runs from bare roots.
- The fleet is **fixed** with no midstream refill, so the live batch drains as
  games finish and the long tail runs at low GPU occupancy.

And it defers two decisions outright: what (if anything) to retain across plies,
and whether the fleet path should serve non-learning tournament play.

## Stories

| # | Story | Addresses |
|---|-------|-----------|
| #26 | Vectorised node representation | Tree internals still per-node; PUCT is a Python loop |
| #27 | Position-keyed evaluation cache | No eval reuse; retention-as-performance has no home |
| #28 | Speculative batch backfill (tail trimming) | Long tail runs at low GPU occupancy |
| #29 | Midstream refill / queueing | Fixed fleet drains over time |
| #30 | Retention during training decision | Open: what to retain across plies |
| #31 | Fleet tournament play | Fleet path is self-play only |

### Dependencies and ordering

- **#26** is independent — a self-contained refactor of the node representation
  that vectorises selection. It makes the internals cheaper but changes no
  results.
- **#27** introduces the position hash/equality contract and an LRU cache in
  front of `evaluate_positions`. It is the prerequisite for #28 and the cleaner
  home for any retention-as-performance value (see #30).
- **#28** depends on **#27** — speculative backfill is pure cache-fill, so it
  needs the cache to fill.
- **#29** is independent of the above; it changes the driver, not the engine.
- **#30** is a decision story (a spike + a call), not a feature; its outcome may
  feed back into #23's retention stance and into #27.
- **#31** is independent; it extends the fleet path to `StandardGame`.

## The retention thread (why #27 and #30 are separate)

Tree retention carries two distinct kinds of value, and Phase Two splits them:

- **Retention-as-performance** (reuse cached evaluations) — better served by a
  position-keyed cache (#27) than by tree-node retention, because a cache
  survives across games, catches transpositions, and is exactly what the tail
  prefetch (#28) would populate. Under the wave its value is muted anyway (a
  wave pays one forward pass the moment any single game needs an evaluation), so
  it mainly narrows the batch rather than skipping the call.
- **Retention-as-strength** (carry visit statistics forward so a new search
  continues the old) — a genuine search-strength lever, and an open question in
  learning mode. That is the decision in #30.

## Non-goals (epic-wide)

- Anything already delivered by the phase-1 skateboard (#20–#24).
- Distributed / multi-GPU fleets — Phase Two targets saturating a single
  device, not scaling across many.

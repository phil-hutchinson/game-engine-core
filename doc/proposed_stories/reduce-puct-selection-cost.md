# Proposed: reduce PUCT selection cost at low branching factors

**Status:** proposed, not scheduled. Deliberately deferred — see *When to pick
this up*.

Follows #26 (vectorised node representation). All the measurements referenced
here are recorded in
[`doc/plan/00000026-vectorized-node-representation/benchmark.md`](../plan/00000026-vectorized-node-representation/benchmark.md),
including the machine they were taken on.

## The situation #26 left

#26 moved child statistics onto the parent as numpy arrays and made PUCT
selection a single array computation. Search results did not move, and the
`apply_plies` seam improved as intended — 4.5x fewer position constructions,
call width bounded by fleet size instead of fleet x branching factor.

Selection got slower. A sweep of the isolated selection kernel across branching
factors put the cost at a flat ~5.6 us per node scored regardless of width,
against a scalar loop's ~285 ns + 182 ns/slot. Almost all of it is numpy's
per-call dispatch across the ten array operations in `child_puct_values`, not
arithmetic. The crossover — where the array form starts winning — is at a
**branching factor of about 30**.

That is why the four benchmark cells regress by 29–78%: both games in this repo
are illustrative examples of library usage, and at branching 2 and 9 they sit
far below the crossover. They are not evidence about real consuming games and
should not be read as such.

## The two candidates

Both were measured as part of #26's Step 6. Neither is implemented.

**D — keep slot addressing, drop numpy.** Hold the per-slot statistics in
Python `list`s rather than `ndarray`s and score them with a scalar loop. Costs
within 5% of the pre-#26 child-object representation at every width tested, so
it recovers the regression outright. Crucially it keeps *slot addressing*, which
is what #26's lazy materialisation and fleet-batched `apply_plies` are written
against — those steps are untouched by this change. numpy stops being a runtime
dependency of `game_engine_core`.

Confined to `MCTSNode`: `expand` builds lists, `child_puct_value(slot)` returns
(without the `int()`/`float()` conversions numpy needed), `child_puct_values`
and its `np.argmax` call site go away.

**E — keep numpy, cut the op count.** Fold the scalar coefficient into one
multiply, replace the `zeros_like` + masked `divide` guard with a clamped
denominator, accumulate in place. Measured at a consistent 1.5x over the shipped
kernel, dropping the floor from ~5.6 us to ~3.7 us and moving the crossover from
~30 to ~19.

E depends on an invariant that is currently implicit: `record_visit` only ever
increments a slot's visit count and its total value together, so a slot with
zero visits has a total value of exactly 0.0 and `maximum(visits, 1)` is an
exact substitute for the mask. Adopting E means stating that where both `expand`
and `record_visit` can see it — a future path that updated one array without the
other would silently change selection rather than fail.

## Recommendation, if forced today: E

Not because it is faster in this repo — D is, at both example games' widths —
but because **its cost is bounded and D's is not**. E is flat: ~3.7 us per node
scored at width 2, ~4.7 us at width 400. D is linear at ~188 ns/slot forever:
15 us per node at width 100, 71 us at width 400, and rising. Choosing E caps the
worst case at roughly 3 us per node scored against the best available option;
choosing D leaves the worst case unbounded in the one direction a general
library cannot rule out.

The counterweight is that 1.5x is a modest return for the work, and that E
commits `game_engine_core` to a runtime dependency it did not have, justified by
the boundedness argument rather than by any measured cell.

## When to pick this up

The choice needs the branching factor **and depth** of a game that actually
consumes this library, and no such game exists yet. Implementing one is what
will supply it. Specifically:

- **Branching factor below ~19** — D is the better representation, and the
  numpy dependency is not earning its place.
- **Branching factor above ~30** — the shipped kernel is already correct; E is
  a straightforward 1.5x on top and D would be a regression.
- **Between the two, or spread across both** — E, on the bounding argument
  above.

Depth matters as much as width, because selection runs at every node of a
descent: a deep narrow game pays the per-node cost ~10 times per iteration where
a shallow wide one pays it ~4 times. Neither figure is knowable from the example
games.

Waiting is cheap precisely because the shipped approach is the bounded one. The
cost of deferring is capped at ~3 us per node scored over the best alternative,
which is why this is a proposal rather than a blocker.

## The larger option this defers past

Vectorising across the **fleet** rather than within a node scores all N trees'
current nodes in one array operation, paying numpy's dispatch once per wave
instead of once per node scored. That puts every game above the crossover
regardless of its own branching factor — the only option here that helps a
narrow game at all. It is out of scope for both D and E because the trees sit at
different depths with different branching, so the batch is ragged and needs
padding or a flattened layout. #26's plan already names it as the better axis;
if selection is still the bottleneck after a real game exists, this is the story
to write instead of either option above.

## Non-goals

- No change to search results. Whichever option is taken, the selected slot must
  be identical for identical statistics — the signatures in `benchmark.md` are
  the check.
- No change to the seam. #26's lazy materialisation and fleet-batched
  `apply_plies` are kept as they are; both D and E are confined to how a node's
  slots are stored and scored.
- Not a strength change.

## Instruments that already exist

All beside `doc/plan/00000026-vectorized-node-representation/benchmark.md`:

- `selection_sweep.py` — the kernel sweep that produced the crossover figures.
  D and E are already implemented in it as variants, so measuring a change to
  either starts from working code.
- `benchmark.py` — the four-cell search benchmark. Whichever option is taken,
  its four result signatures must not move.
- `seam_call_counts.py`, `profile_cell.py`, `ply_sequences.py` — seam widths,
  time attribution, and the fixed-seed re-rooting check.

What does not exist, and would be worth building before this is decided: a
synthetic position with configurable branching factor *and depth*. The sweep
isolates the kernel, so it is silent on the ~34–57% of runtime outside
selection — including `record_visit` and the `visits` property, which also moved
onto ndarrays in Step 2. Measuring a *whole search* at arbitrary width and depth
is what would turn the projections here into numbers.

# Story: Speculative batch backfill / tail trimming (issue #28)

Part of the **Fleet Play Phase Two** epic (#25). Depends on #27 (evaluation
cache).

## Goal

Keep the GPU saturated through the long tail by **backfilling the wave's batch
with speculative evaluations**. When the number of live games drops below a
configurable minimum-batch floor, spend the otherwise-idle batch slots
pre-evaluating leaves the search is likely to want soon — the highest-prior
frontier of each live game — so that later iterations find them already in the
cache (#27).

## Motivation

A GPU forward pass is near-flat in cost below its saturation width: a batch of 3
costs almost as much as a batch of 64. In the fleet's long tail, as games finish
at different plies the live batch drains, and the final turns run many waves at a
handful of positions each — wall-clock diverges sharply from the
GPU-saturated ideal. Those empty batch slots are free, so filling them with
speculative work the search will probably reach converts wasted capacity into
cache hits.

The effect is strongest exactly where it is needed. Down to the last game, each
wave can pre-evaluate a leaf plus its siblings and go deep, so a search that digs
twenty levels into one favoured line makes only a handful of real forward passes
and hits the cache for the rest — collapsing the tail's sequential single-game
calls into a few wide, free ones.

## The correctness line

Speculation must fill the cache **only**. The key property:

- **Prefetching an evaluation is safe** — the network is a pure function of
  position, so pre-computing and caching its value/policy changes nothing; when
  the search legitimately arrives it reads the cached answer, bit-identical to
  computing on demand.
- **A speculative node must not touch `visits` / backpropagation** until the real
  descent reaches it. Backpropagating a leaf PUCT did not actually select would
  alter the statistics — that is leaf-parallel-with-virtual-loss, an
  *approximation* of the sequential search, and a strength change rather than a
  free optimisation. This story stays on the safe side of that line: conceptually
  one new node still enters each tree per iteration; speculation only warms the
  cache.

## Scope

- A configurable minimum-batch floor.
- A frontier-selection heuristic (highest-prior unexpanded leaves per live game)
  to choose speculative positions, sized to fill the batch up to the floor
  without exceeding the GPU's efficient width.
- Route speculative positions through the cache-fill path (#27) without any
  effect on visit statistics or backpropagation.

## Non-goals

- No speculative backpropagation / virtual loss — no change to search results.
- No refilling the fleet with new games — that is #29; this story trims the tail
  of a fixed fleet.
- Not a substitute for #29; the two are complementary tail strategies.

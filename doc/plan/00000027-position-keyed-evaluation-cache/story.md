# Story: Position-keyed evaluation cache (issue #27)

Part of the **Fleet Play Phase Two** epic (#25). Prerequisite for #28.

## Goal

Put an LRU cache in front of `evaluate_positions`, keyed by position, so a
position already evaluated is served from the cache instead of the network. The
network is a pure function of position, so this changes no results — it only
avoids recomputing evaluations.

## Motivation

Two Phase Two needs share this one mechanism:

- **Retention-as-performance.** The skateboard drops the retained tree; the
  cleaner home for its performance value is a position-keyed cache. Unlike
  tree-node retention it survives across games and catches transpositions, and it
  does so without carrying stale visit statistics (the search-strength question
  stays separate — see #30).
- **Tail trimming (#28).** Speculative batch backfill is pure cache-fill: it
  pre-evaluates likely-soon frontier leaves so later iterations hit the cache.
  That requires a cache to fill.

Under the wave, a cache's value is real but bounded: a wave pays one forward pass
the moment *any* game needs an evaluation, so cache hits mostly *narrow* the
batch (nearly free) rather than skip the call — a whole call is skipped only on
waves where the entire fleet is cached-or-terminal. The cache earns its keep
mainly in combination with #28 and near the endgame.

## The contract cost

This is the story that introduces the **position hash/equality contract**: to key
a cache by position, `GamePosition` must support hashing and equality. The
skateboard deliberately avoids this so simple games pay nothing; it is paid here,
where the cache needs it. Pre-release, with a single downstream consumer, this
breaking widening is acceptable.

## Scope

- Define the position hash/equality contract on `GamePosition`.
- An LRU eval cache in front of `evaluate_positions`, with a configurable
  capacity, that partitions a batch into hits (served from cache) and misses
  (sent to the network), then records the misses.
- Wire the cache into the fleet engine's evaluation step so waves consult it
  before batching.

## Non-goals

- No change to search results — caching a pure function is transparent.
- No speculative population here — that is #28; this story only serves and
  records real evaluations.
- No visit-statistic reuse — this is retention-as-performance only; the strength
  question is #30.

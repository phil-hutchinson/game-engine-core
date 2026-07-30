# Epic: Fleet Play (issue #20)

## Goal

Make MCTS self-play run as a **fleet** — N games advanced in lockstep so that a
single batched network evaluation serves all N at once, instead of N games each
making their own sequential evaluator calls. A normal single game is just the
fleet at `N = 1`.

The performance thesis is a single fact about the network forward pass: it
dominates self-play wall-clock (≈70% on CPU), and a GPU forward is near-flat in
cost below its saturation width — a batch of 256 costs little more than a batch
of 1. Today every game evaluates one position per MCTS iteration on its own, so
the GPU is fed one position at a time and never saturates. If N games are
synchronized at **iteration granularity**, each MCTS iteration gathers one leaf
from every game into a single forward pass, and the GPU is fed N positions at
once.

Amdahl's law then sets the ceiling: once the forward is batched, the residual
per-game work (position encoding, the policy transform, legality generation)
becomes the wall. So the epic also widens the game-facing protocols to be
**batch-first**, letting a game vectorise those touchpoints too.

## The "wave" structure

One MCTS iteration is turned sideways. Instead of running select → expand →
evaluate → backpropagate to completion on one tree, each phase sweeps all N
trees before the next begins, with the batched evaluation as the single
synchronisation point:

```
for each game:  select a leaf           →  N leaves
partition:      terminal vs non-terminal
                evaluate_positions(non-terminal leaves)   →  ONE batched call
for each game:  scatter value + backpropagate
```

Terminal leaves carry a known outcome and never touch the network — they are
partitioned out of the batch and backpropagate directly. Every live
non-terminal game contributes exactly one position per wave, keeping the games
in lockstep (equal iteration budgets ⇒ equal wave counts, so synchronisation is
automatic). We explicitly do **not** let a game run ahead to fill a bigger
batch — preserving ply/iteration lockstep is worth more than a marginally wider
batch.

The terminal partition and, at expansion, the legality and successor calls all
go through the `BatchPositionProcessor` seam (`batch_ops`, #22) rather than the
position directly — #22 lands that seam with every call still batch-of-one; #23
is what widens the calls it makes through it to width N.

## Stories

### Phase 1 — the skateboard

| # | Story | Depends on |
|---|-------|------------|
| #21 | True PUCT / full node expansion | — |
| #22 | Batch-first game protocols | — |
| #23 | Fleet / wave MCTS engine | #21, #22 |
| #24 | SelfPlayCollector as fleet driver | #22, #23 |
| #26 | Vectorised node representation | #21, #23 |

#21 and #22 are independent and may land in either order. #23 consumes both;
#24 drives #23. #26 lands last, and was pulled forward from Phase Two (#25):
once the forward pass is batched, the residual per-node CPU work is the wall
this epic's own Amdahl argument predicts, and the tree internals are the
largest remaining piece of it. Capture the Flag (the downstream consumer)
adopts the whole set in a single release.

Two decisions shape the cut:

- **#21 is deliberately separate from the fleet.** Fixing expansion to true
  PUCT is a single-game, search-strength change with almost no intersection with
  mass play — but the wave *assumes* AlphaZero-style expansion (one evaluation
  per newly-reached leaf, all children attached at once). Doing it first means
  #23 inherits search machinery already in the right shape, rather than
  re-litigating expansion inside the harder story.
- **#22 batches only the game-connected touchpoints**, not the tree internals.
  PUCT descent and backpropagation chase parent pointers and depend on each
  step's argmax; vectorising them is a separate lever, taken separately in #26
  rather than folded into the protocol change.

### Phase 2 — backlog (deferred / open)

Tracked as its own epic, Fleet Play Phase Two (#25). Vectorised node
representation was originally P1 here; it has been pulled into Phase 1 as #26
and the remaining refs renumbered.

| Ref | Story | Note |
|-----|-------|------|
| P1 | Position-keyed evaluation cache | LRU cache in front of `evaluate_positions`; introduces a position hash/equality contract. Subsumes retention-as-performance and catches transpositions. |
| P2 | Speculative batch backfill (tail trimming) | When live games drop below a min-batch floor, prefetch high-prior frontier leaves to keep the GPU saturated during the long tail. Pure cache-fill, **no** speculative backpropagation. Depends on P1. |
| P3 | Midstream refill / queueing | Start new games as others finish, holding fleet size steady instead of letting it drain. |
| P4 | Retention decision (open) | Whether anything is retained across plies — eval-cache only vs. visit-stat carryover (a search-strength choice). Genuinely undecided; the skateboard runs from bare roots. |
| P5 | Fleet tournament play | Extend the fleet path to non-learning tournament / `StandardGame` use. |

## Why the retained tree is out of the skateboard

Tree retention (issue #14) carries two kinds of value: retained **visit
statistics** (a search-strength lever) and retained **evaluations** (a
performance lever). The skateboard drops both:

- The strength lever is a separate axis (P4), left open pending a decision on
  what AlphaZero should do in learning mode.
- The performance lever largely evaporates under the wave. A wave's cost floor
  is one GPU forward pass, paid the moment a *single* game needs an evaluation
  that iteration; cached evaluations for the other games only *narrow* the
  batch, which is nearly free anyway. So eval-caching saves a whole call only on
  waves where the entire fleet is cached-or-terminal at once. If that lever is
  wanted back, its cleaner home is a position-keyed cache in front of the
  evaluator (P1), not tree-node retention.

## Non-goals (epic-wide)

- An evaluation cache and the position hash/equality contract it needs (P1).
- Speculative prefetch / batch backfill for the long tail (P2).
- Refilling the fleet with new games midstream (P3) — the skateboard runs a
  fixed fleet and accepts the long tail.
- Mass-play tournaments (P5) — the fleet targets self-play only.
- A DAG / transposition-sharing tree — #26 changes how children are stored, not
  that nodes form a tree.

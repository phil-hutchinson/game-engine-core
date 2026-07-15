# Story: Retain the MCTS tree between plies (issue #14)

## Goal

Stop `MCTSEngine` from discarding its search tree after every ply. Currently
each `select_ply` builds a fresh tree from scratch, re-expanding and (with a
neural evaluator) re-evaluating positions that the previous search already
explored. Instead, the engine retains the subtree rooted at the node
representing the ply that was actually played, carrying its visit counts, values,
priors, cached policies, and children forward into the next search.

Concretely: in position *n* the engine holds a whole tree. When a ply is
selected and applied, the engine re-roots onto the child node representing that
ply, discarding the rest of the tree but keeping everything beneath the chosen
node. The next search grows from that warm-started root.

## Motivation

Rebuilding the tree every ply throws away work that is still valid. The retained
subtree's statistics remain correct after re-rooting — values are stored from
each node's own perspective and back-propagation negates per level, so no sign
adjustment is needed — and retained nodes never need re-evaluating, which is the
dominant cost under a neural-network evaluator. Warm-starting each search from
the retained tree lets a caller reach the same playing strength with a smaller
`iterations` budget, or greater strength at the same budget.

## Design

**Re-rooting is driven by observed plies, not inferred from position diffs.** The
engine is *told* every ply as it is applied, and descends its tree one node per
observed ply. This is agnostic to turn structure: it makes no assumption about
how many plies pass between the engine's own turns (strict alternation,
move-again rules, passing, or more than two players are all just sequences of
observed plies). A position-diff approach at a fixed depth would silently bake in
a strict-alternation assumption; observation avoids that.

The seam is three methods added to the `GameEngine` and `Player` protocols:

- **`observe_ply(position, ply, new_position)`** — notification that a ply has
  been applied. The MCTS engine re-roots onto the matching child (matched by
  `str(ply)`, the framework's established ply identity key); if no matching child
  exists (the ply was still unexplored, or there is no tree yet) it clears the
  retained root so the next search rebuilds from the true position. Non-retaining
  players/engines implement it as a no-op.
- **`reset()`** — notification that a new game has started; the engine clears its
  retained tree. Required because engine instances are reused across games (a
  `Tournament` reuses the same `AIPlayer`/engine for every game in the
  round-robin), so a stale tree must not leak between games.

`StandardGame.run` calls `reset()` on both players at the start of each game and
broadcasts every applied ply to both players via `observe_ply`. Because the
retained root is trusted without re-validating it against the incoming position
(the `GamePosition` protocol has no value equality), the reset-on-miss and
reset-on-new-game paths are load-bearing, not optional — they are what keep a
stale root from ever driving a search.

**`select_ply` retains; `select_ply_with_policy` does not.** This split follows
their intended usage:

- `select_ply` is the gameplay/tournament path and uses the retained root
  (building a fresh one only when none is held).
- `select_ply_with_policy` is the self-play data-collection path
  (`SelfPlayCollector`). It always builds a fresh tree and never touches the
  retained root. Retention would change the semantics of the visit-count
  distribution used as the policy training target — retained visits were
  allocated under a different root context and one level deeper — so self-play
  deliberately keeps the clean per-position search.

## Scope

- `MCTSEngine`: retained `_root_node` field; `select_ply` warm-starts from it;
  `observe_ply` re-roots (or clears on miss); `reset()` clears it;
  `select_ply_with_policy` unchanged in behaviour (always fresh). Internal
  refactor separating root creation from tree growth. Reduce default iterations
  to 1000
- `GameEngine` and `Player` protocols: add `observe_ply` and `reset`.
- `AIPlayer`: forward `observe_ply`/`reset` to its engine.
- `HumanPlayer`: no-op `observe_ply`/`reset`.
- `RandomEngine`: no-op `observe_ply`/`reset`.
- `StandardGame.run`: reset both players at game start; broadcast each applied
  ply to both players.
- Test coverage for re-rooting, reset, the fresh-root fallback on a miss, and
  statistic preservation.

## Non-goals

- **No iteration top-up.** The retained search still runs the full `iterations`
  count on top of the warm-started root. Reducing iterations to exploit the
  warm start is left to the caller for now; a dynamic/adaptive budget is a
  possible future enhancement.
- **No transposition handling.** Positions reached by different ply orders remain
  distinct nodes; re-rooting keeps a single subtree, it does not merge
  transpositions.
- **No value equality on `GamePosition`.** Re-rooting is driven by observed
  plies, not by comparing positions, so the position protocol is untouched.
- **No change to self-play training semantics.** `select_ply_with_policy` and
  `SelfPlayCollector` behave exactly as before.

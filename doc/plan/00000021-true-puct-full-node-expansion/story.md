# Story: True PUCT / full node expansion (issue #21)

Part of the **Fleet Play** epic (#20).

## Goal

Replace the current one-child-per-iteration expansion with **AlphaZero-style
full expansion**: on first reaching a leaf, evaluate it once, attach *all* of
its children at once with their priors, and let PUCT govern every descent from
then on.

One MCTS iteration becomes:

1. **Select** — descend from the root by PUCT until reaching a node that has not
   yet been evaluated (a leaf).
2. **Evaluate** — call the evaluator on that leaf, yielding value and policy.
   For a terminal leaf, use the outcome and skip the network.
3. **Expand** — create a child for every legal ply, seeding each with its prior
   from the policy.
4. **Backpropagate** — push the value up to the root.

The root is simply the first leaf: on a fresh tree, iteration 1 evaluates and
expands the root; subsequent iterations descend to deeper leaves. Exactly one
evaluation per iteration, always.

## Motivation

The current engine is UCT with priors bolted on, not true PUCT. `_select_leaf`
descends only `while current.is_fully_expanded` (`mcts_engine.py:153`), and
`_expand_node` adds a single child per visit via `pop()` (`mcts_engine.py:173`).
So a node's first *K* visits are spent expanding one sibling at a time, and PUCT
only governs descent *after* every child exists — the search explores all
siblings once before exploitation can begin at that node. True PUCT instead lets
a high-prior ply be selected repeatedly while a low-prior sibling stays at zero
visits, because an unvisited child's score is its exploration term (∝ prior)
alone.

This also fixes a latent quirk: because expansion always evaluates the *child*,
a fresh root is never evaluated, so its `policy` stays `None` and its children
fall back to uniform priors (`prior = 1.0`, `mcts_engine.py:176`). Only deeper
nodes ever receive network priors today. Under full expansion the root is
evaluated on its first visit and its children get real priors.

Beyond correctness, this is the expansion shape the fleet wave requires: the
wave wants exactly one evaluation per game per iteration, and "evaluate a
newly-reached leaf, attach all its children" delivers precisely that. Landing it
here — as a self-contained single-game change — keeps it out of the harder fleet
story (#23).

## Scope

- Rework `_select_leaf` / `_expand_node` / `_mcts_iteration` for the
  evaluate-then-expand-all flow.
- Evaluate the root as the first leaf; remove the uniform-prior fallback.
- Remove `unexplored_plies`, the expansion `random.shuffle`, and the
  `is_fully_expanded` gate — full expansion makes them vestigial.
- Simplify `_visit_distribution`: with every legal ply already a child (possibly
  0-visit), the "include unexplored plies with 0 visits" special case
  (`mcts_engine.py:134`) is no longer needed.
- `observe_ply` misses become near-impossible — after a root is evaluated all
  legal plies are children, and visit-count selection never plays into a
  0-visit child — so the null-the-tree branch becomes a rare edge, not a common
  path.

## Non-goals

- No mass play / batching of any kind — single-game engine only (that is #23).
- No change to the node's storage representation: children stay as
  `MCTSNode` objects with a scalar `prior`; the `policy` dict is consumed at
  expansion and may be discarded, but the array-based representation is deferred
  (epic backlog P1).
- No change to **retention policy**. The retained tree keeps its current
  semantics; the `observe_ply` simplification above falls out naturally and does
  not decide the open retention question (epic backlog P5).
- No change to the evaluator protocol (that is #22).

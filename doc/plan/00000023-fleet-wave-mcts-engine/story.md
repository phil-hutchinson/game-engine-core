# Story: Fleet / wave MCTS engine (issue #23)

Part of the **Fleet Play** epic (#20). Depends on #21 (true PUCT) and #22
(batch-first protocols).

## Goal

Make `MCTSEngine` fleet-capable: search N independent games in lockstep so that
each MCTS iteration produces a single batched evaluation across all N. A normal
single game is the fleet at `N = 1`.

New primitives:

```python
def select_plies(self, positions: Sequence[TPosition]) -> Sequence[TPly]: ...
def select_plies_with_policy(
    self, positions: Sequence[TPosition]
) -> Sequence[tuple[TPly, dict[str, float]]]: ...
```

`select_ply` / `select_ply_with_policy` become thin `N = 1` wrappers over these,
preserving the existing single-game API.

## The wave

Each engine holds N trees, one per game, addressed by slot index — the position
at index *i* belongs to game *i*, and the ply returned at index *i* is game
*i*'s. With equal iteration budgets, "wave *w*" is just "iteration *w* of every
game," so the games stay synchronised with no explicit coordination.

One wave:

1. **Select** one leaf in every tree by PUCT descent.
2. **Partition** the leaves: terminal (outcome known, via `batch_ops.outcomes`)
   vs. non-terminal.
3. **Evaluate** the non-terminal leaves in a single `evaluate_positions(...)`
   call (#22).
4. **Scatter + backpropagate**: give each non-terminal leaf its returned value
   and expand its children (full expansion, #21, via `batch_ops.legal_plies` /
   `batch_ops.apply_plies`); backpropagate terminal leaves directly from their
   outcome. Every live game advances exactly one iteration.

`batch_ops` (`BatchPositionProcessor`, #22) is the seam for all of this — the
engine already reaches every position through it as of #22, one call per
position; this story is what widens those calls from N calls of width 1 to one
call of width N.

Two consequences drive the implementation:

- **Terminals leave the batch.** A game whose selected leaf is terminal
  contributes nothing to the forward pass but still advances its wave (it
  backpropagates its known value), so lockstep is preserved.
- **The evaluation side effect becomes an explicit scatter.** Today
  `_evaluate_node` both returns a value and mutates `node.policy` in place. In a
  wave the values arrive as a batched result, so assigning "row *i* → game *i*'s
  leaf" (value, and the priors for full expansion) is a post-evaluation step,
  not a side effect buried inside evaluation.

## Motivation

The batched evaluation only happens if every game's per-iteration evaluator call
is collected into one forward pass, and that requires the games to be
synchronised at **iteration granularity** — not ply granularity. A single
`select_ply` runs the full iteration budget (e.g. 200 evaluations)
sequentially, so interleaving N independent `select_ply` calls at the ply level
would not line their evaluations up. The wave loop therefore has to live *inside*
a fleet method that holds all N trees at once. This is that method.

## Scope

- Hold N trees (`list` of roots) instead of a single `_root_node`; index games
  by slot.
- Implement the wave: per-wave leaf selection, terminal partition (via
  `batch_ops`), one batched `evaluate_positions`, scatter of values + full
  expansion (via `batch_ops`), per-tree backpropagation.
- `select_ply` / `select_ply_with_policy` delegate to the plural forms at
  `N = 1`, keeping single-game behaviour identical.
- Build fresh (bare) roots from the supplied positions on each call — no
  cross-call retention (see non-goals).

## Non-goals

- **No retention** across calls or plies; each `select_plies` starts from bare
  roots (epic backlog P5). Under the wave, retention's performance value largely
  disappears anyway — see the epic story.
- No evaluation cache / position hashing (P2), no speculative prefetch (P3).
- No vectorising of selection or backpropagation — the wave batches the
  game-touchpoint calls, but descent and backprop stay per-tree loops (P1).
- No fleet driving: deciding *which* games are in flight, bootstrapping them,
  and handling the long tail is the collector's job (#24). This story owns only
  the synchronised search across whatever N it is handed.

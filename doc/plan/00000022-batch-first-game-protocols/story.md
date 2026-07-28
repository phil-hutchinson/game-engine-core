# Story: Batch-first game protocols (issue #22)

Part of the **Fleet Play** epic (#20).

## Goal

Give the search a **plural, whole-batch** seam for every game touchpoint it
uses, so the library always calls the batch form and the game decides whether to
service it with a real tensor path or a sequential loop.

The seam for position operations is a new concrete class,
`BatchPositionProcessor`, whose default implementations loop the existing
`GamePosition` methods. A game that wants vectorisation subclasses it and
overrides selectively; a game that does not ignores it and gets the base
instance. The evaluator seams widen in place, plural-only.

Touchpoints, by their new home:

- **`BatchPositionProcessor`** (new): `outcomes(positions)`,
  `legal_plies(positions)`, `apply_plies(positions, plies)`.
- **`PositionEvaluator`**: `evaluate_position` → `evaluate_positions(positions)`.
- **`NeuralNetworkEvaluator`**: `encode_position` → `encode_positions`;
  `decode_policy` → `decode_policies`; a real batched forward replaces the
  current batch-of-one `unsqueeze(0)` / `squeeze(0)` dance.
- **`SelfPlayCollector`**: `policy_transform` becomes batch-shaped.

**`GamePosition` is untouched.** Its scalar members stay exactly as they are —
they are what `BatchPositionProcessor`'s defaults are built from, and what
`StandardGame`, `RandomEngine`, the UI and the players keep using. The batch
seam serves the search path (`MCTSEngine`, and `SelfPlayCollector` driving it);
single-game play has no reason to pay for it.

`TrainingLoop`'s `policy_loss_fn` / `value_loss_fn` are already batch-shaped
(they score a whole batch of predictions against targets in
`training_loop.py`), so they need no change — noted here only to record that
they are the same kind of seam.

The package is alpha with a single implementing repository, and the README
already states that breaking changes land in any release. **Backwards
compatibility is not a goal of this story.** Where a scalar form is replaced, it
is replaced outright rather than kept alongside: one touchpoint per concept, no
implementation left with half its methods dead.

## Motivation

Amdahl's law. The network forward pass is ≈70% of self-play wall-clock on CPU;
moving only that to a batched GPU call caps the achievable speed-up, because the
residual per-game work then dominates. Profiling puts `encode_position` and the
`policy_transform` next after the forward, with legality generation behind them.
Batching those removes most of the residual overhead.

The wave (issue #23) reinforces this: each wave phase already gathers N of
everything (N leaves to test for terminal, N positions to expand, N to
evaluate). With batch-first seams each phase is a single bulk call instead of a
Python loop of N calls:

- select leaves → one `outcomes(...)` to partition terminal vs. non-terminal
- evaluate → one `evaluate_positions(...)`
- expand → one `legal_plies(...)` + one `apply_plies(...)`
- capture → one `encode_positions(...)`, one batched `policy_transform`

Should #26 later materialise children lazily, `apply_plies` moves from expansion
to leaf materialisation — one call per wave either way, which is why its shape
matters more than its placement (see below).

The design principle is **batch everything at the search boundary, let the game
decide what it processes en masse or sequentially.** A position already carries
small CPU tensors, so a game like Capture the Flag can stack them and compute
legality / encodings for the whole fleet at once; tic-tac-toe can ignore the
opportunity and inherit the loop. Offering the batch entry point everywhere —
even where a given game will not use it — is what lets the fast games go fast
without taxing the simple ones.

### Why a separate class rather than a wider `GamePosition`

A position is one position. The batch operations are functions of N positions,
so a position instance is not a natural receiver for them, and neither
`GamePosition` nor `PositionEvaluator` is the right home: `NullEvaluator` is
game-agnostic library code that could not implement legality for an arbitrary
game, and a tournament runs two players with two different evaluators over one
shared set of rules.

A concrete class rather than a protocol-plus-default-implementation pair,
because a correct game-agnostic default *exists* for every method here — looping
the `GamePosition` scalars. That default is not a stub for toy games: if
`outcome` is a stored field, the loop is already the optimal implementation, and
a fully vectorised game is expected to inherit some methods and override others.
The general rule this story adopts, and which the evaluator seams follow in the
other direction: **ship a default where a game-agnostic one exists; make it
abstract where it does not.**

### `apply_plies` is flat and pairwise

`apply_plies(positions, plies)` pairs its arguments element-by-element rather
than taking one position and its many legal plies. Both eager expansion (each
leaf repeated once per legal ply, flattened into a single call) and lazy
materialisation (the one leaf per tree that a wave actually descended into) are
expressible in that shape, so this story does not decide between them —
#26 can adopt lazy child materialisation without another protocol change.

Folding successor positions into `legal_plies` — returning `(ply, position)`
pairs — was considered and rejected for the same reason: it bakes eager
expansion into the seam permanently, and it forces successor construction on
the callers that want legality alone (`RandomEngine`, the UI,
`_visit_distribution`'s zero-total fallback).

### `decode_policies` and `policy_transform` — the alignment discipline

Both deal with the perspective-relative action space, at opposite ends. During
search, `decode_policy` maps the head's logits (laid out in the mover's frame)
back to global-frame `str(ply)` priors. At capture, `policy_transform`
re-expresses the MCTS visit distribution into the head's column layout so it can
serve as a training target. Both need `position.active_player_id`, so their
batched forms must carry the N positions alongside the N tensors and pair them
element-by-element (`zip(rows, positions)`) — the same positional-identity
alignment the fleet relies on throughout, and the same contract `apply_plies`
carries.

## Scope

- Add `BatchPositionProcessor` with looping defaults over the `GamePosition`
  scalars, and document the one rule an implementer has to hold: for each
  operation, exactly one side is the real implementation — a scalar method may
  delegate batch-of-one to an *overridden* batch method, but delegating to an
  inherited default recurses.
- `MCTSEngine` takes a processor (defaulting to the base instance) and routes
  **every** position operation through it, including the fallback paths in
  `_visit_distribution` and `_select_best_ply`, so a vectorised game and a
  sequential one cannot diverge.
- Widen `PositionEvaluator` to `evaluate_positions`; `NullEvaluator` services it
  itself.
- Make `NeuralNetworkEvaluator` genuinely batched: its abstract surface becomes
  `encode_positions` / `decode_policies`, and `evaluate_positions` stacks the
  encodings, runs one forward, splits the rows and decodes them.
- Migrate the library's own call sites (engine, collector) to the plural forms,
  and make `policy_transform` batch-shaped.
- Update the examples and fixtures (tic-tac-toe, Nim) to the new surfaces.
  `TicTacToeNNEvaluator` implements `encode_positions` by genuine stacking, as
  the worked example of the pattern; `NimNNEvaluator` stays a plain
  comprehension for contrast. Neither subclasses `BatchPositionProcessor`,
  demonstrating the intended default path.

## Non-goals

- No vectorising of the tree internals — PUCT descent and backpropagation are
  not game touchpoints and stay per-tree CPU work (epic backlog P1 / #26).
- No lazy child materialisation. `apply_plies` is shaped so #26 can adopt it,
  but expansion keeps building every child's position eagerly here.
- No position hash/equality requirement — that is the evaluation cache's
  contract (epic backlog P2), not this story's.
- No obligation on any game to actually vectorise.
- No `GamePosition` changes, and therefore none to `StandardGame`,
  `RandomEngine`, the players, the UI or the tournament runner.
- No behaviour change in `MCTSEngine`: it routes existing single-game work
  through the batch seam at N = 1. The wave itself is #23. This includes
  expansion, which issues one batch-of-one `apply_plies` call per child rather
  than a single call of width B — the width-B call is available without the
  wave and is behaviour-identical, but widening any call is #23's business, not
  this story's (peer review #3).
- No bulk position factory. Bootstrapping N starting positions happens once per
  `collect()`, not once per wave, so `position_factory` stays as it is (#24).
- The fleet engine and driver that consume these (issues #23, #24) are separate.
